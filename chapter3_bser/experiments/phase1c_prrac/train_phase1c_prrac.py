"""Independent PRRAC trainer for Chapter 3 Phase 1C.

Long experiments are user-launched only. Worker episodes use immutable actor
snapshots and return transitions to the parent learner for all optimization.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import socket
import statistics
import time
import traceback
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from chapter3_bser.controllers.state_provider import OnlinePlanningStateProvider
from chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.train_phase1c_v2 import (
    _make_base_env,
    _public_context,
    _seed_all,
)
from chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.training_env import (
    Phase1CV2TrainingEnv,
)
from chapter3_bser.experiments.phase1c_prrac import (
    ARCHITECTURE_VERSION,
    CHECKPOINT_SCHEMA,
    IMPLEMENTATION_VERSION,
    METHOD,
)
from chapter3_bser.experiments.phase1c_prrac.diagnostics import PRRACDiagnostics
from chapter3_bser.experiments.phase1c_prrac.replay_adapter import PRRACReplayAdapter
from chapter3_bser.experiments.phase1c_prrac.runtime_factory import (
    CONTROLLER_FACTORY_VERSION,
    NATIVE_B1_RUNTIME_REVISION,
    build_prrac_online_controller,
    runtime_contract,
)
from chapter3_bser.experiments.phase1c_prrac.search_continuity import (
    SearchContinuityDiagnostics,
)
from chapter3_bser.experiments.phase1c_prrac.training_env import PRRACTrainingEnv
from chapter3_bser.experiments.phase1c_prrac.transition_protocol import (
    PRRACTransitionMetadata,
)
from chapter3_bser.integration.guided_env import GuidedEnv
from chapter3_bser.integration.rmaddpg_bridge import RMADDPGGuidanceBridge
from chapter3_bser.models.prrac.prrac_maddpg import PRRACMADDPG
from chapter3_bser.models.prrac.stage_mapping import STAGE_MAPPING
from chapter3_bser.online.config import execution_runtime_config, load_phase1b2_config
from core.registry.experiment_registry import assert_registered_ch3_method
from core.scenarios.ch3_generator_impl import build_scenario_manifests


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "chapter3" / "bser_phase1c_prrac_train.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "chapter3" / "phase1c_prrac" / "training"
INCOMPATIBLE_MESSAGE = (
    "Phase 1C-v1/v2 checkpoints cannot be resumed under PRRAC architecture semantics."
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _json_safe(float(value.detach().cpu().item()))
        return _json_safe(value.detach().cpu().tolist())
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_json_safe(rows))


def _config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _json_safe(dict(config)), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema": "bser.phase1c.prrac.training.v1",
        "method": METHOD,
        "implementation_version": IMPLEMENTATION_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "observation_dim": 28,
        "action_dim": 3,
        "critic_dim": 124,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"invalid PRRAC config {key}: {config.get(key)!r}")
    if config.get("resume_from_v1_allowed") is not False or config.get(
        "resume_from_v2_allowed"
    ) is not False:
        raise ValueError("PRRAC must reject Phase 1C-v1/v2 resume")
    if config.get("guidance_enabled") is not True or config.get(
        "training_update"
    ) is not True:
        raise ValueError("PRRAC requires BSER guidance and real training updates")
    if config.get("profile") != "M20_MOVING_UNKNOWN_MULTI":
        raise ValueError("PRRAC formal profile changed")
    contract = runtime_contract(config)
    if contract.checkpoint_runtime_revision == NATIVE_B1_RUNTIME_REVISION:
        if config.get("execution_variant") != "B1_ATOMIC_LAST_VALID":
            raise ValueError("native PRRAC training requires B1_ATOMIC_LAST_VALID")
        if config.get("runtime_integration_mode") != "native":
            raise ValueError("native PRRAC training requires runtime_integration_mode=native")
    config["execution_runtime"] = execution_runtime_config(config)
    return config


def _parameter_counts(learner: PRRACMADDPG) -> dict[str, int]:
    return {
        "router_parameter_count": sum(
            p.numel() for agent in learner.agents for p in agent.actor.router.parameters()
        ),
        "expert_parameter_count": sum(
            p.numel()
            for agent in learner.agents
            for p in agent.actor.residual_experts.parameters()
        ),
        "gate_parameter_count": sum(
            p.numel()
            for agent in learner.agents
            for p in agent.actor.trust_gate_module.parameters()
        ),
        "critic_parameter_count": sum(
            p.numel()
            for agent in learner.agents
            for module in (agent.critic1, agent.critic2)
            for p in module.parameters()
        ),
    }


def _checkpoint_metadata(
    config: Mapping[str, Any], completed_episode: int, learner: PRRACMADDPG
) -> dict[str, Any]:
    contract = runtime_contract(config)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "method": METHOD,
        "implementation_version": IMPLEMENTATION_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "config_hash": _config_hash(config),
        "completed_episode": int(completed_episode),
        "seed": int(config["seed"]),
        "profile": str(config["profile"]),
        "observation_dim": 28,
        "action_dim": 3,
        "critic_dim": 124,
        "stage_mapping": dict(STAGE_MAPPING),
        "architecture": copy.deepcopy(config["architecture"]),
        "loss": copy.deepcopy(config["loss"]),
        "reward": copy.deepcopy(config["reward"]),
        "replay": copy.deepcopy(config["replay"]),
        "execution_runtime_revision": contract.checkpoint_runtime_revision,
        "execution_variant": contract.execution_variant.value,
        "runtime_integration_mode": contract.runtime_integration_mode,
        "controller_factory_version": CONTROLLER_FACTORY_VERSION,
        **_parameter_counts(learner),
    }


def _save_checkpoint(
    learner: PRRACMADDPG,
    replay: PRRACReplayAdapter,
    directory: Path,
    config: Mapping[str, Any],
    completed_episode: int,
    *,
    global_step: int,
    update_step: int,
    replay_sample_count: int,
    optimizer_update_count: int,
    episode_rows: list[dict[str, Any]],
    execution_rows: list[dict[str, Any]],
    prrac_rows: list[dict[str, Any]],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"phase1c_prrac_episode_{int(completed_episode):04d}.pt"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "metadata": _checkpoint_metadata(config, completed_episode, learner),
        "prrac_training_state": learner.training_state_dict(),
        "prrac_replay_state": replay.state_dict(),
        "completed_episode": int(completed_episode),
        "global_step": int(global_step),
        "update_step": int(update_step),
        "replay_sample_count": int(replay_sample_count),
        "optimizer_update_count": int(optimizer_update_count),
        "episode_metrics": [dict(row) for row in episode_rows],
        "execution_diagnostics": [dict(row) for row in execution_rows],
        "prrac_diagnostics": [dict(row) for row in prrac_rows],
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary checkpoint exists: {temporary}")
    torch.save(payload, temporary)
    temporary.replace(path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if loaded.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("saved PRRAC checkpoint schema validation failed")
    if loaded.get("metadata", {}).get("config_hash") != _config_hash(config):
        raise RuntimeError("saved PRRAC checkpoint config hash validation failed")
    return path


def _load_checkpoint(
    path: Path,
    learner: PRRACMADDPG,
    replay: PRRACReplayAdapter,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    schema = payload.get("schema")
    if schema in {
        "bser.phase1c.training_state.v1",
        "bser.phase1c.training_state.v2",
    }:
        raise ValueError(INCOMPATIBLE_MESSAGE)
    if schema != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported PRRAC checkpoint schema: {schema!r}")
    metadata = dict(payload.get("metadata", {}))
    if metadata.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("PRRAC checkpoint architecture mismatch")
    expected_runtime = runtime_contract(config)
    actual_revision = str(metadata.get("execution_runtime_revision", ""))
    if actual_revision != expected_runtime.checkpoint_runtime_revision:
        raise ValueError("checkpoint execution runtime revision mismatch")
    actual_variant = str(
        metadata.get("execution_variant", "B0_LEGACY_V2_1")
    )
    actual_integration = str(metadata.get("runtime_integration_mode", "legacy"))
    actual_factory = str(
        metadata.get("controller_factory_version", CONTROLLER_FACTORY_VERSION)
    )
    if actual_variant != expected_runtime.execution_variant.value:
        raise ValueError("checkpoint execution variant mismatch")
    if actual_integration != expected_runtime.runtime_integration_mode:
        raise ValueError("checkpoint runtime integration mode mismatch")
    if actual_factory != CONTROLLER_FACTORY_VERSION:
        raise ValueError("checkpoint controller factory version mismatch")
    if metadata.get("architecture") != dict(config["architecture"]):
        raise ValueError("PRRAC checkpoint architecture config mismatch")
    if metadata.get("config_hash") != _config_hash(config):
        raise ValueError("PRRAC checkpoint config hash mismatch")
    for key, expected in (("observation_dim", 28), ("action_dim", 3), ("critic_dim", 124)):
        if int(metadata.get(key, -1)) != expected:
            raise ValueError(f"PRRAC checkpoint {key} mismatch")
    learner.load_training_state_dict(payload["prrac_training_state"])
    replay.load_state_dict(payload["prrac_replay_state"])
    return payload


def _verify_checkpoint_roundtrip(
    path: Path,
    config: Mapping[str, Any],
) -> bool:
    """Load a checkpoint into newly constructed learner and replay objects."""

    checkpoint = Path(path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    replay_state = payload["prrac_replay_state"]["base_replay"]
    rl = dict(config.get("rl", {}))
    learner = PRRACMADDPG(
        architecture=config["architecture"],
        loss=config["loss"],
        gamma=float(rl.get("gamma", 0.95)),
        tau=float(rl.get("tau", 0.01)),
        lr_actor=float(rl.get("lr_actor", 0.001)),
        lr_critic=float(rl.get("lr_critic", 0.001)),
    )
    replay = PRRACReplayAdapter(
        max_steps=int(replay_state["max_steps"]),
        config=config.get("replay", {}),
        generator_seed=int(config["seed"]) + 31,
    )
    restored = _load_checkpoint(checkpoint, learner, replay, config)
    if int(restored["completed_episode"]) != int(
        restored["metadata"]["completed_episode"]
    ):
        raise RuntimeError("PRRAC checkpoint episode metadata mismatch after restore")
    if len(replay) != int(replay_state["filled_i"]):
        raise RuntimeError("PRRAC checkpoint replay length mismatch after restore")
    replay_roundtrip = replay.state_dict()
    for name in ("stage_before", "stage_after"):
        if not torch.equal(
            replay_roundtrip[name], payload["prrac_replay_state"][name]
        ):
            raise RuntimeError(f"PRRAC checkpoint {name} mismatch after restore")
    return True


def _build_learner(config: Mapping[str, Any]):
    learner = PRRACMADDPG(
        architecture=config["architecture"],
        loss=config["loss"],
        gamma=float(config["rl"]["gamma"]),
        tau=float(config["rl"]["tau"]),
        lr_actor=float(config["rl"]["lr_actor"]),
        lr_critic=float(config["rl"]["lr_critic"]),
    )
    replay = PRRACReplayAdapter(
        max_steps=int(config["rl"]["replay_size"]),
        config=config["replay"],
        generator_seed=int(config["seed"]) + 31,
    )
    return learner, replay


def _clone_parameters(parameters):
    return tuple(parameter.detach().cpu().clone() for parameter in parameters)


def _changed(before, after) -> int:
    return sum(int(not torch.equal(left, right.detach().cpu())) for left, right in zip(before, after))


def _numpy_copy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _contains_tensor(payload: Any) -> bool:
    if torch.is_tensor(payload):
        return True
    if isinstance(payload, Mapping):
        return any(
            _contains_tensor(key) or _contains_tensor(value)
            for key, value in payload.items()
        )
    if isinstance(payload, (list, tuple, set, frozenset)):
        return any(_contains_tensor(value) for value in payload)
    if is_dataclass(payload) and not isinstance(payload, type):
        return any(
            _contains_tensor(getattr(payload, field.name)) for field in fields(payload)
        )
    return False


def _combine_episode_diagnostics(
    rollout: Mapping[str, Any], update: Mapping[str, Any]
) -> dict[str, Any]:
    """Keep rollout actor diagnostics authoritative and update data separate."""

    rollout_values = copy.deepcopy(dict(rollout))
    update_values = copy.deepcopy(dict(update))
    combined = dict(rollout_values)
    combined.update(
        {
            "router_rollout_accuracy": rollout_values.get("router_accuracy"),
            "router_rollout_confusion_matrix": rollout_values.get(
                "router_confusion_matrix"
            ),
            "router_update_accuracy": update_values.get("router_accuracy"),
            "router_update_confusion_matrix": update_values.get(
                "router_confusion_matrix"
            ),
            "rollout_diagnostics": rollout_values,
            "update_diagnostics": update_values,
            "stage_critic_losses": update_values.get("stage_critic_losses", {}),
            "stage_td_errors": update_values.get("stage_td_errors", {}),
        }
    )
    return combined


def _sum_confusion_matrices(
    rows: list[dict[str, Any]],
    field: str,
    *,
    fallback_field: str | None = None,
) -> list[list[int]]:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for row in rows:
        value = row.get(field)
        if value is None and fallback_field is not None:
            value = row.get(fallback_field)
        if value is not None:
            candidate = np.asarray(value, dtype=np.int64)
            if candidate.shape != (3, 3):
                raise ValueError(f"{field} must be a 3x3 confusion matrix")
            matrix += candidate
    return [[int(value) for value in row] for row in matrix.tolist()]


def _confusion_accuracy(matrix: list[list[int]]) -> float | None:
    values = np.asarray(matrix, dtype=np.int64)
    total = int(values.sum())
    return None if total == 0 else float(np.trace(values) / total)


def _router_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rollout_matrix = _sum_confusion_matrices(
        rows,
        "router_rollout_confusion_matrix",
        fallback_field="router_confusion_matrix",
    )
    update_matrix = _sum_confusion_matrices(rows, "router_update_confusion_matrix")
    rollout_accuracy = _confusion_accuracy(rollout_matrix)
    update_accuracy = _confusion_accuracy(update_matrix)
    stage_coverage = {
        name: bool(sum(rollout_matrix[index]) > 0)
        for index, name in enumerate(("search", "intercept", "hold"))
    }
    return {
        "router_accuracy": rollout_accuracy,
        "router_confusion_matrix": rollout_matrix,
        "router_rollout_accuracy": rollout_accuracy,
        "router_rollout_confusion_matrix": rollout_matrix,
        "router_update_accuracy": update_accuracy,
        "router_update_confusion_matrix": update_matrix,
        "stage_coverage": stage_coverage,
        "all_stages_observed": all(stage_coverage.values()),
    }


def _runtime_metadata(
    requested_device: str, resolved_device: torch.device
) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_name = (
        str(torch.cuda.get_device_name(resolved_device))
        if resolved_device.type == "cuda"
        else None
    )
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_available": cuda_available,
        "requested_device": str(requested_device),
        "resolved_device": str(resolved_device),
        "cuda_device_name": cuda_device_name,
        "hostname": socket.gethostname(),
    }


def _apply_transitions(
    learner: PRRACMADDPG,
    replay: PRRACReplayAdapter,
    transitions,
    episode_summary: Mapping[str, Any],
    rl: Mapping[str, Any],
    *,
    global_step: int,
    update_step: int,
    device: str,
):
    diagnostics = PRRACDiagnostics()
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    replay_sample_count = 0
    optimizer_update_count = 0
    actor_update_count = 0
    for obs, actions, rewards, next_obs, dones, success_flags, metadata in transitions:
        replay.push(
            tuple(torch.as_tensor(value, dtype=torch.float32) for value in obs),
            torch.as_tensor(actions, dtype=torch.float32),
            torch.as_tensor(rewards, dtype=torch.float32),
            tuple(torch.as_tensor(value, dtype=torch.float32) for value in next_obs),
            tuple(bool(value) for value in dones),
            tuple(bool(value) for value in success_flags),
            PRRACTransitionMetadata.from_dict(metadata)
            if isinstance(metadata, Mapping)
            else metadata,
        )
        global_step += 1
        if (
            global_step < int(rl["warmup_steps"])
            or global_step % int(rl["update_frequency"]) != 0
            or len(replay) < int(rl["batch_size"])
        ):
            continue
        learner.prep_training(device)
        for _ in range(int(rl["updates_per_train"])):
            batch = replay.sample(int(rl["batch_size"]), norm_rews=False, device=device)
            replay_sample_count += 1
            errors = []
            update_actor = update_step % int(rl["policy_delay"]) == 0
            for agent_i in range(4):
                if update_actor:
                    result = learner.update(batch, agent_i)
                    actor_losses.append(float(result["actor_loss"]))
                    actor_update_count += 1
                else:
                    result = learner.update_critic_only(batch, agent_i)
                critic_losses.append(float(result["critic_loss"]))
                errors.append(result["td_error"])
                diagnostics.observe_update(result)
                optimizer_update_count += 1
                with torch.no_grad():
                    output = learner.agents[agent_i].actor(
                        torch.as_tensor(batch.obs[agent_i], device=learner.device)
                    )
                    diagnostics.observe_actor(output, batch.stage_before)
            replay.update_priorities(
                batch.indices,
                torch.stack([torch.as_tensor(value) for value in errors]).mean(dim=0),
                batch.success_tail_flags,
            )
            learner.update_all_targets()
            update_step += 1
        learner.prep_rollouts(device)
    marked = replay.finalize_episode(
        int(episode_summary.get("episode_id", -1)),
        success=bool(episode_summary.get("success", False)),
    )
    return {
        "global_step": int(global_step),
        "update_step": int(update_step),
        "replay_sample_count": int(replay_sample_count),
        "optimizer_update_count": int(optimizer_update_count),
        "actor_update_count": int(actor_update_count),
        "actor_loss": None if not actor_losses else float(statistics.fmean(actor_losses)),
        "critic_loss": None if not critic_losses else float(statistics.fmean(critic_losses)),
        "success_tail_marked": int(marked),
        "diagnostics": diagnostics.summary(),
    }


def _collect_episode(job: dict[str, Any]):
    """Collect one episode from a fixed snapshot; no optimizer exists in worker."""

    torch.set_num_threads(1)
    episode_index = int(job["episode_index"])
    scenario = copy.deepcopy(job["scenario"])
    _seed_all(int(scenario["scenario_seed"]))
    base_env = _make_base_env(job, device="cpu")
    guided = GuidedEnv(base_env, enabled=True)
    v2_env = Phase1CV2TrainingEnv(guided, reward_config=job["reward"])
    env = PRRACTrainingEnv(v2_env)
    started = time.perf_counter()
    try:
        env.reset(
            scenario=scenario,
            episode_id=episode_index,
            episode_index=episode_index,
        )
        phase1b_config = load_phase1b2_config()
        runtime_config = execution_runtime_config(job)
        phase1b_config["execution_runtime"] = copy.deepcopy(runtime_config)
        provider = OnlinePlanningStateProvider(
            env,
            refresh_interval=int(phase1b_config["online"]["state_refresh_interval"]),
            refresh_on_executor_handoff=bool(runtime_config["refresh_on_executor_handoff"]),
            refresh_on_public_target_shift=bool(runtime_config["refresh_on_public_target_shift"]),
            public_target_update_distance=float(runtime_config["public_target_update_distance"]),
            public_target_update_min_steps=int(runtime_config["public_target_update_min_steps"]),
        )
        state = provider.initialize()
        context = _public_context(env, state)
        controller = build_prrac_online_controller(phase1b_config, job)
        initialized = controller.initialize(state, context)
        bridge = RMADDPGGuidanceBridge()
        guidance = bridge.compile_guidance(
            initialized.allocation, state, context, decision_reason="INITIALIZE"
        )
        env.install_guidance(guidance)
        observations = env.refresh_observation_after_guidance()
        actor = PRRACMADDPG(
            architecture=job["architecture"],
            loss=job["loss"],
            gamma=float(job["rl"]["gamma"]),
            tau=float(job["rl"]["tau"]),
            lr_actor=float(job["rl"]["lr_actor"]),
            lr_critic=float(job["rl"]["lr_critic"]),
        )
        actor.load_policy_snapshot(job["policy_snapshot"])
        actor.prep_rollouts("cpu")
        actor.reset_noise()
        transitions = []
        rollout_diagnostics = PRRACDiagnostics()
        search_diagnostics = SearchContinuityDiagnostics()
        search_diagnostics_enabled = bool(
            dict(job.get("search_continuity_diagnostics", {})).get("enabled", False)
        )
        if search_diagnostics_enabled:
            search_diagnostics.begin_episode(state)
        reward_total = 0.0
        collision_count = 0
        event_count = 0
        accepted_replans = 0
        action_norms: list[float] = []
        final_step = 0
        for _ in range(int(job["max_steps"])):
            state_before = state
            installed_guidance = guidance
            with torch.no_grad():
                action_parts = actor.step(observations, explore=True)
                actions = torch.stack([item.squeeze(0) for item in action_parts])
                actor_outputs = [
                    actor.agents[agent_i].actor(
                        torch.as_tensor(observation).reshape(1, 28)
                    )
                    for agent_i, observation in enumerate(observations)
                ]
            _, rewards, dones = env.step(actions)
            metadata = env.last_prrac_transition_metadata
            if metadata is None:
                raise RuntimeError("PRRAC training wrapper did not emit stage metadata")
            for output in actor_outputs:
                rollout_diagnostics.observe_actor(output, [int(metadata.stage_before)])
            state = provider.snapshot(force=False)
            if search_diagnostics_enabled:
                search_diagnostics.observe_transition(
                    stage_before=metadata.stage_before,
                    stage_after=metadata.stage_after,
                    installed_guidance=installed_guidance,
                    planning_state_before=state_before,
                    planning_state_after=state,
                    collision_flags=env.unwrapped.collision_flags,
                    raw_actions=actions,
                    applied_actions=actions,
                    actor_outputs=actor_outputs,
                    residual_contribution_ratios=(
                        env.unwrapped.last_residual_contribution_ratio_search
                    ),
                )
            context = _public_context(env, state)
            result = controller.step(state, context)
            env.observe_controller_result(result, controller=controller, state_provider=provider)
            guidance = bridge.compile_guidance(
                result.allocation, state, context, decision_reason=result.decision_reason
            )
            env.install_guidance(guidance)
            next_observations = env.refresh_observation_after_guidance()
            event_count += len(result.events)
            accepted_replans += int(bool(result.replanned))
            collision_count += int(env.unwrapped._collision_flags.sum().item())
            action_norms.append(float(torch.linalg.vector_norm(actions, dim=1).mean().item()))
            reward_tensor = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1)
            reward_total += float(reward_tensor.sum().item())
            task = env.get_task_state()
            final_step = int(task.step)
            transitions.append(
                (
                    tuple(_numpy_copy(value) for value in observations),
                    _numpy_copy(actions),
                    _numpy_copy(reward_tensor),
                    tuple(_numpy_copy(value) for value in next_observations),
                    tuple(bool(value) for value in dones),
                    tuple(bool(task.mission_complete) for _ in range(4)),
                    metadata.to_dict(),
                )
            )
            observations = next_observations
            if all(bool(value) for value in dones):
                break
        task = env.get_task_state()
        metrics = {
            "method": METHOD,
            "implementation_version": IMPLEMENTATION_VERSION,
            "architecture_version": ARCHITECTURE_VERSION,
            "episode": episode_index + 1,
            "episode_index": episode_index,
            "episode_id": episode_index,
            "scenario_id": str(scenario.get("scenario_id", "")),
            "scenario_seed": int(scenario["scenario_seed"]),
            "success": bool(task.mission_complete),
            "found": bool(task.target_found),
            "contact_episode": bool(getattr(env.unwrapped, "capture_contact_step_count", 0)),
            "hold_episode": bool(getattr(env.unwrapped, "capture_full_hold_step_count", 0)),
            "collision": int(collision_count),
            "episode_length": int(final_step),
            "event_count": int(event_count),
            "accepted_replans": int(accepted_replans),
            "reward": float(reward_total),
            "action_norm": 0.0 if not action_norms else float(statistics.fmean(action_norms)),
            "wall_seconds": float(time.perf_counter() - started),
        }
        if search_diagnostics_enabled:
            metrics.update(
                search_diagnostics.summary(
                    found=bool(task.target_found), max_steps=int(job["max_steps"])
                )
            )
        payload = (
            metrics,
            transitions,
            env.finalize_episode(),
            rollout_diagnostics.summary(),
        )
        if _contains_tensor(payload):
            raise RuntimeError("PRRAC worker payload must not contain torch.Tensor")
        return payload
    finally:
        env.close()


def _rolling(values, window: int):
    return [
        float(statistics.fmean(values[max(0, index + 1 - window) : index + 1]))
        for index in range(len(values))
    ]


def _save_line_plot(path: Path, title: str, series: Mapping[str, list[float]]) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for label, values in series.items():
        axis.plot(range(1, len(values) + 1), values, label=label)
    axis.set(xlabel="Episode", title=title)
    if len(series) > 1:
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_outputs(rows, prrac_rows, metrics_dir: Path, window: int) -> None:
    if not rows:
        return
    _save_line_plot(
        metrics_dir / "loss_curves.png",
        "PRRAC training losses",
        {
            "actor": [float(row.get("actor_loss") or 0.0) for row in rows],
            "critic": [float(row.get("critic_loss") or 0.0) for row in rows],
        },
    )
    _save_line_plot(
        metrics_dir / "success_found_trend.png",
        "Success / found trend",
        {
            "success": _rolling([float(row["success"]) for row in rows], window),
            "found": _rolling([float(row["found"]) for row in rows], window),
        },
    )
    _save_line_plot(
        metrics_dir / "phase_replay_curve.png",
        "Phase replay counts",
        {
            name: [float(row.get(f"replay_count_{name}", 0)) for row in rows]
            for name in ("pre_found", "post_found", "contact_hold", "success_tail")
        },
    )
    _save_line_plot(
        metrics_dir / "router_probability_curve.png",
        "Router probabilities",
        {
            name: [float(row.get(f"router_probability_{name}") or 0.0) for row in prrac_rows]
            for name in ("search", "intercept", "hold")
        },
    )
    _save_line_plot(
        metrics_dir / "trust_gate_curve.png",
        "Trust gate",
        {"mean": [float(row.get("gate_mean") or 0.0) for row in prrac_rows]},
    )
    _save_line_plot(
        metrics_dir / "expert_action_norm_curve.png",
        "Expert action norm",
        {
            name: [float(row.get(f"expert_action_norm_{name}") or 0.0) for row in prrac_rows]
            for name in ("search", "intercept", "hold")
        },
    )
    matrix = np.zeros((3, 3), dtype=np.int64)
    for row in prrac_rows:
        matrix += np.asarray(row.get("router_confusion_matrix", matrix), dtype=np.int64)
    figure, axis = plt.subplots(figsize=(5.5, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(xlabel="Predicted", ylabel="Actual", title="Router confusion matrix")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(metrics_dir / "router_confusion_matrix.png", dpi=180)
    plt.close(figure)


def _group_parameters(learner: PRRACMADDPG) -> dict[str, tuple[torch.Tensor, ...]]:
    return {
        "router": _clone_parameters(
            p for agent in learner.agents for p in agent.actor.router.parameters()
        ),
        "expert": _clone_parameters(
            p for agent in learner.agents for p in agent.actor.residual_experts.parameters()
        ),
        "gate": _clone_parameters(
            p for agent in learner.agents for p in agent.actor.trust_gate_module.parameters()
        ),
        "critic_head": _clone_parameters(
            p
            for agent in learner.agents
            for critic in (agent.critic1, agent.critic2)
            for head in critic.heads
            for p in head.parameters()
        ),
    }


def _current_group_parameters(learner: PRRACMADDPG, name: str):
    if name == "router":
        return (p for agent in learner.agents for p in agent.actor.router.parameters())
    if name == "expert":
        return (p for agent in learner.agents for p in agent.actor.residual_experts.parameters())
    if name == "gate":
        return (p for agent in learner.agents for p in agent.actor.trust_gate_module.parameters())
    return (
        p
        for agent in learner.agents
        for critic in (agent.critic1, agent.critic2)
        for head in critic.heads
        for p in head.parameters()
    )


def _dry_run_requirements_met(
    requirements: Mapping[str, Any],
    update_counts: Mapping[str, int],
    *,
    parameter_update_count: int,
    critic_head_parameter_count: int,
    checkpoint_count: int,
    checkpoint_load_verified: bool,
) -> bool:
    return bool(
        (
            not requirements["require_parameter_update"]
            or parameter_update_count > 0
        )
        and (
            not requirements["require_router_update"]
            or update_counts["router_parameter_update_count"] > 0
        )
        and (
            not requirements["require_expert_update"]
            or update_counts["expert_parameter_update_count"] > 0
        )
        and (
            not requirements["require_gate_update"]
            or update_counts["gate_parameter_update_count"] > 0
        )
        and (
            not requirements["require_all_critic_heads_update"]
            or update_counts["critic_head_parameter_update_count"]
            == critic_head_parameter_count
        )
        and (
            not requirements["require_checkpoint"]
            or (checkpoint_count > 0 and checkpoint_load_verified)
        )
    )


def run_training(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path | None = None,
    dry_run: bool = False,
    seed_override: int | None = None,
    resume: Path | None = None,
    episodes_override: int | None = None,
    max_steps_override: int | None = None,
    workers_override: int | None = None,
    device_override: str | None = None,
) -> dict[str, Any]:
    assert_registered_ch3_method(METHOD)
    config = copy.deepcopy(_load_config(config_path))
    overrides = {
        "seed": seed_override,
        "episodes": episodes_override,
        "max_steps": max_steps_override,
        "workers": workers_override,
        "device": device_override,
    }
    if dry_run:
        config["episodes"] = int(config["dry_run"]["episodes"])
        config["max_steps"] = int(config["dry_run"]["max_steps"])
        config["checkpoint_interval"] = int(config["episodes"])
    for key, value in overrides.items():
        if value is not None:
            config[key] = int(value) if key not in {"device"} else str(value)
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    if int(config["episodes"]) <= 0 or int(config["max_steps"]) <= 0:
        raise ValueError("episodes and max_steps must be positive")
    configured_output = Path(config["output_dir"])
    if not configured_output.is_absolute():
        configured_output = ROOT / configured_output
    output = Path(output_dir) if output_dir is not None else configured_output
    if dry_run:
        output = output / "dry_run"
    directories = {name: output / name for name in ("checkpoints", "metrics", "logs")}
    protected = (
        output / "resolved_training_config.json",
        directories["metrics"] / "training_summary.json",
    )
    if resume is None and (
        any(path.exists() for path in protected)
        or any(directories["checkpoints"].glob("*.pt"))
    ):
        raise FileExistsError(f"PRRAC output exists; use --resume or a new directory: {output}")
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    resolved = output / "resolved_training_config.json"
    if not resolved.exists():
        _write_json(resolved, config)
    _seed_all(int(config["seed"]))
    learner, replay = _build_learner(config)
    learner.prep_rollouts(str(device))
    rows: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []
    prrac_rows: list[dict[str, Any]] = []
    checkpoints: list[str] = []
    global_step = update_step = replay_sample_count = optimizer_update_count = 0
    start_episode = 0
    if resume is not None:
        payload = _load_checkpoint(resume, learner, replay, config)
        start_episode = int(payload["completed_episode"])
        global_step = int(payload["global_step"])
        update_step = int(payload["update_step"])
        replay_sample_count = int(payload["replay_sample_count"])
        optimizer_update_count = int(payload["optimizer_update_count"])
        rows = [dict(row) for row in payload.get("episode_metrics", ())]
        execution_rows = [dict(row) for row in payload.get("execution_diagnostics", ())]
        prrac_rows = [dict(row) for row in payload.get("prrac_diagnostics", ())]
        checkpoints.append(str(Path(resume).resolve()))
    if start_episode >= int(config["episodes"]):
        raise ValueError("resume checkpoint already reached configured episode count")
    initial_groups = _group_parameters(learner)
    manifest = build_scenario_manifests(
        count=int(config["episodes"]),
        generator_seed=int(config["seed"]),
        split="train",
        profiles=(str(config["profile"]),),
    )[str(config["profile"])]
    scenarios = list(manifest["scenarios"])
    started = time.perf_counter()
    workers = max(1, min(int(config["workers"]), int(config["episodes"]) - start_episode))
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        for batch_start in range(start_episode, int(config["episodes"]), workers):
            indices = range(batch_start, min(batch_start + workers, int(config["episodes"])))
            snapshot = learner.policy_snapshot()
            jobs = [
                {
                    "episode_index": index,
                    "scenario": scenarios[index],
                    "base_candidate": config["base_candidate"],
                    "profile": config["profile"],
                    "max_steps": config["max_steps"],
                    "rl": config["rl"],
                    "reward": config["reward"],
                    "architecture": config["architecture"],
                    "loss": config["loss"],
                    "execution_runtime": config["execution_runtime"],
                    "execution_runtime_revision": config["execution_runtime_revision"],
                    "execution_variant": config.get(
                        "execution_variant", "B0_LEGACY_V2_1"
                    ),
                    "runtime_integration_mode": config.get(
                        "runtime_integration_mode", "legacy"
                    ),
                    "controller_factory_version": config.get(
                        "controller_factory_version", CONTROLLER_FACTORY_VERSION
                    ),
                    "search_continuity_diagnostics": config.get(
                        "search_continuity_diagnostics", {"enabled": False}
                    ),
                    "policy_snapshot": snapshot,
                }
                for index in indices
            ]
            try:
                results = list(executor.map(_collect_episode, jobs))
            except Exception as exc:
                failure = directories["logs"] / f"worker_failure_{batch_start + 1:04d}.json"
                _write_json(
                    failure,
                    {
                        "schema": "bser.phase1c.prrac.worker_failure.v1",
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
            for metrics, transitions, execution, rollout_diag in sorted(
                results, key=lambda item: int(item[0]["episode_index"])
            ):
                update = _apply_transitions(
                    learner,
                    replay,
                    transitions,
                    {"episode_id": metrics["episode_id"], "success": metrics["success"]},
                    config["rl"],
                    global_step=global_step,
                    update_step=update_step,
                    device=str(device),
                )
                global_step = update["global_step"]
                update_step = update["update_step"]
                replay_sample_count += update["replay_sample_count"]
                optimizer_update_count += update["optimizer_update_count"]
                metrics.update(
                    actor_loss=update["actor_loss"],
                    critic_loss=update["critic_loss"],
                    replay_size=len(replay),
                    replay_sample_count=update["replay_sample_count"],
                    optimizer_update_count=update["optimizer_update_count"],
                )
                for name, count in replay.phase_counts().items():
                    metrics[f"replay_count_{name}"] = int(count)
                combined_diag = _combine_episode_diagnostics(
                    rollout_diag, update["diagnostics"]
                )
                combined_diag["episode"] = int(metrics["episode"])
                rows.append(metrics)
                execution_rows.append(execution)
                prrac_rows.append(combined_diag)
                episode = int(metrics["episode"])
                if episode % int(config["checkpoint_interval"]) == 0 or episode == int(config["episodes"]):
                    checkpoint = _save_checkpoint(
                        learner,
                        replay,
                        directories["checkpoints"],
                        config,
                        episode,
                        global_step=global_step,
                        update_step=update_step,
                        replay_sample_count=replay_sample_count,
                        optimizer_update_count=optimizer_update_count,
                        episode_rows=rows,
                        execution_rows=execution_rows,
                        prrac_rows=prrac_rows,
                    )
                    checkpoints.append(str(checkpoint.resolve()))
            _write_csv(directories["metrics"] / "episode_metrics.csv", rows)
            _write_csv(directories["metrics"] / "execution_diagnostics.csv", execution_rows)
            _write_csv(directories["metrics"] / "prrac_diagnostics.csv", prrac_rows)
    update_counts = {
        f"{name}_parameter_update_count": _changed(
            before, _current_group_parameters(learner, name)
        )
        for name, before in initial_groups.items()
    }
    _plot_outputs(rows, prrac_rows, directories["metrics"], int(config["rolling_window"]))
    _write_json(directories["checkpoints"] / "checkpoint_list.json", checkpoints)
    found_count = sum(bool(row["found"]) for row in rows)
    success_count = sum(bool(row["success"]) for row in rows)
    contact_count = sum(bool(row["contact_episode"]) for row in rows)
    hold_count = sum(bool(row["hold_episode"]) for row in rows)
    collision_count = sum(int(row["collision"]) > 0 for row in rows)
    router_summary = _router_summary(prrac_rows)

    def scalar_mean(name):
        values = [float(row[name]) for row in prrac_rows if row.get(name) is not None]
        return None if not values else float(statistics.fmean(values))
    def nested_mean(field, name):
        values = [
            float(row[field][name])
            for row in prrac_rows
            if isinstance(row.get(field), Mapping)
            and row[field].get(name) is not None
        ]
        return None if not values else float(statistics.fmean(values))
    parameter_update_count = sum(update_counts.values())
    checkpoint_load_verified = bool(
        checkpoints
        and _verify_checkpoint_roundtrip(Path(checkpoints[-1]), config)
    )
    pipeline_passed = bool(
        len(rows) == int(config["episodes"])
        and replay_sample_count > 0
        and optimizer_update_count > 0
        and parameter_update_count > 0
        and checkpoints
    )
    dry_requirements = config["dry_run"]
    dry_run_passed = bool(
        dry_run
        and pipeline_passed
        and _dry_run_requirements_met(
            dry_requirements,
            update_counts,
            parameter_update_count=parameter_update_count,
            critic_head_parameter_count=len(initial_groups["critic_head"]),
            checkpoint_count=len(checkpoints),
            checkpoint_load_verified=checkpoint_load_verified,
        )
    )
    summary = {
        "schema": "bser.phase1c.prrac.training.summary.v1",
        "method": METHOD,
        "implementation_version": IMPLEMENTATION_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "pipeline_passed": pipeline_passed,
        "dry_run_passed": dry_run_passed,
        "performance_passed": None,
        "completed_episode_count": len(rows),
        "global_step": int(global_step),
        "replay_sample_count": int(replay_sample_count),
        "optimizer_update_count": int(optimizer_update_count),
        "parameter_update_count": int(parameter_update_count),
        **update_counts,
        **router_summary,
        "router_stage_counts": replay.stage_counts(),
        "gate_mean": scalar_mean("gate_mean"),
        "gate_p10": scalar_mean("gate_p10"),
        "gate_p90": scalar_mean("gate_p90"),
        "gate_saturation_low_rate": scalar_mean("gate_saturation_low_rate"),
        "gate_saturation_high_rate": scalar_mean("gate_saturation_high_rate"),
        "alignment_mean": scalar_mean("alignment_mean"),
        "alignment_negative_rate": scalar_mean("alignment_negative_rate"),
        "stage_critic_losses": {
            name: nested_mean("stage_critic_losses", name)
            for name in ("search", "intercept", "hold")
        },
        "stage_td_errors": {
            name: nested_mean("stage_td_errors", name)
            for name in ("search", "intercept", "hold")
        },
        "found_rate": 0.0 if not rows else found_count / len(rows),
        "success_rate": 0.0 if not rows else success_count / len(rows),
        "success_if_found_rate": 0.0 if found_count == 0 else success_count / found_count,
        "contact_episode_rate": 0.0 if not rows else contact_count / len(rows),
        "hold_episode_rate": 0.0 if not rows else hold_count / len(rows),
        "collision_episode_rate": 0.0 if not rows else collision_count / len(rows),
        "checkpoint_count": len(checkpoints),
        "checkpoint_load_verified": checkpoint_load_verified,
        "resumed_from": None if resume is None else str(Path(resume).resolve()),
        "wall_seconds": float(time.perf_counter() - started),
        **_runtime_metadata(str(config["device"]), device),
    }
    _write_json(directories["metrics"] / "training_summary.json", summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    summary = run_training(
        config_path=args.config,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        seed_override=args.seed,
        resume=args.resume,
        episodes_override=args.episodes,
        max_steps_override=args.max_steps,
        workers_override=args.workers,
        device_override=args.device,
    )
    print(json.dumps(_json_safe(summary), sort_keys=True, allow_nan=False))
    return 0 if summary["pipeline_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
