"""Independent synchronous-rollout trainer for Phase 1C-v2 execution learning."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
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
from chapter3_bser.experiments.phase1c_bser_rmaddpg_v2 import (
    CHECKPOINT_SCHEMA,
    IMPLEMENTATION_VERSION,
    METHOD,
)
from chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.phase_aware_replay import (
    PhaseAwareReplayBuffer,
)
from chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.training_env import (
    Phase1CV2TrainingEnv,
)
from chapter3_bser.integration.guided_env import GuidedEnv
from chapter3_bser.integration.rmaddpg_bridge import RMADDPGGuidanceBridge
from chapter3_bser.online.config import execution_runtime_config, load_phase1b2_config
from chapter3_bser.online.controller import OnlineBSERController
from chapter3_bser.online.mission_context import OnlineMissionContext
from core.algorithms.maddpg import MADDPG
from core.config.ch3_config import build_ch3_config
from core.env.mission_env import MissionCoreEnv, environment_kwargs_from_config
from core.registry.experiment_registry import assert_registered_ch3_method
from core.scenarios.ch3_generator_impl import build_scenario_manifests


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "chapter3" / "bser_phase1c_v2_train.json"
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "chapter3" / "phase1c_bser_rmaddpg_v2" / "training"
)
V1_SCHEMA = "bser.phase1c.training_state.v1"
LEGACY_EXECUTION_RUNTIME_REVISION = "legacy_static_target_v2"
DYNAMIC_EXECUTION_RUNTIME_REVISION = "dynamic_public_intercept_v2_1"


def _seed_all(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


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
        return [_json_safe(item) for item in value.detach().cpu().tolist()]
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


def _reward_config(config: Mapping[str, Any]) -> dict[str, Any]:
    primary = config.get("reward")
    alternate = config.get("reward_adapter")
    if primary is None and alternate is None:
        raise ValueError("Phase 1C-v2 config requires reward/reward_adapter settings")
    if primary is not None and alternate is not None and dict(primary) != dict(alternate):
        raise ValueError("reward and reward_adapter config aliases differ")
    return dict(primary if primary is not None else alternate)


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema") != "bser.phase1c.training.v2":
        raise ValueError("invalid Phase 1C-v2 training configuration")
    if config.get("method") != METHOD:
        raise ValueError("Phase 1C-v2 method mismatch")
    if config.get("implementation_version") != IMPLEMENTATION_VERSION:
        raise ValueError("Phase 1C-v2 implementation_version mismatch")
    if config.get("resume_from_v1_allowed") is not False:
        raise ValueError("Phase 1C-v2 must set resume_from_v1_allowed=false")
    if config.get("phase1c_enable_bser_guidance") is not True:
        raise ValueError("Phase 1C-v2 must explicitly enable BSER guidance")
    if config.get("guidance_enabled") is not True:
        raise ValueError("Phase 1C-v2 requires guidance_enabled=true")
    if config.get("training_enabled") is not True:
        raise ValueError("Phase 1C-v2 requires training_enabled=true")
    if config.get("training_update") is not True:
        raise ValueError("Phase 1C-v2 training requires updates")
    if config.get("profile") != "M20_MOVING_UNKNOWN_MULTI":
        raise ValueError("Phase 1C-v2 formal profile changed")
    for key, expected in (
        ("observation_dim", 28),
        ("action_dim", 3),
        ("critic_dim", 124),
    ):
        if int(config.get(key, -1)) != expected:
            raise ValueError(f"Phase 1C-v2 {key} changed from frozen contract")
    reward = _reward_config(config)
    if reward.get("schema") != "bser.phase1c.execution_reward.v2":
        raise ValueError("Phase 1C-v2 reward schema mismatch")
    replay = dict(config.get("replay", {}))
    if replay.get("schema") != "bser.phase1c.phase_aware_replay.v1":
        raise ValueError("Phase 1C-v2 replay schema mismatch")
    config["execution_runtime_revision"] = str(
        config.get(
            "execution_runtime_revision", LEGACY_EXECUTION_RUNTIME_REVISION
        )
    )
    config["execution_runtime"] = execution_runtime_config(config)
    return config


def _execution_runtime_revision(config: Mapping[str, Any]) -> str:
    return str(
        config.get(
            "execution_runtime_revision", LEGACY_EXECUTION_RUNTIME_REVISION
        )
    )


def _config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _json_safe(dict(config)),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _public_context(env: Any, state) -> OnlineMissionContext:
    return OnlineMissionContext.from_public_views(
        env.get_task_state(), env.get_search_execution_state(), state
    )


def _policy_snapshot(maddpg: MADDPG):
    return tuple(
        {
            key: value.detach().cpu().numpy().copy()
            for key, value in agent.policy.state_dict().items()
        }
        for agent in maddpg.agents
    )


def _resolve_training_device(device: str | torch.device) -> torch.device:
    device_name = str(device).lower()
    if device_name == "gpu":
        device_name = "cuda"
    resolved = torch.device(device_name)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device!r} was requested for Phase 1C-v2 training, "
            "but torch.cuda.is_available() is False"
        )
    return resolved


def _numpy_copy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _make_base_env(config: Mapping[str, Any], *, device: str = "cpu") -> MissionCoreEnv:
    env_config = build_ch3_config(
        str(config.get("base_candidate", "ch3_v3_full_reference")),
        str(config["profile"]),
    )
    return MissionCoreEnv(
        **environment_kwargs_from_config(
            env_config,
            device=device,
            max_steps=int(config["max_steps"]),
            return_numpy=False,
        )
    )


def _collect_episode(job: dict[str, Any]):
    """Collect one episode with a fixed actor snapshot; never update weights."""

    torch.set_num_threads(1)
    episode_index = int(job["episode_index"])
    scenario = copy.deepcopy(job["scenario"])
    scenario_seed = int(scenario["scenario_seed"])
    _seed_all(scenario_seed)
    base_config = {
        "base_candidate": job["base_candidate"],
        "profile": job["profile"],
        "max_steps": int(job["max_steps"]),
    }
    base_env = _make_base_env(base_config, device="cpu")
    guided = GuidedEnv(base_env, enabled=True)
    env = Phase1CV2TrainingEnv(guided, reward_config=job["reward"])
    started = time.perf_counter()
    try:
        env.reset(
            scenario=scenario,
            episode_id=episode_index,
            episode_index=episode_index,
        )
        phase1b_config = load_phase1b2_config()
        execution_runtime = execution_runtime_config(job)
        phase1b_config["execution_runtime"] = copy.deepcopy(execution_runtime)
        provider = OnlinePlanningStateProvider(
            env,
            refresh_interval=int(phase1b_config["online"]["state_refresh_interval"]),
            refresh_on_executor_handoff=bool(
                execution_runtime["refresh_on_executor_handoff"]
            ),
            refresh_on_public_target_shift=bool(
                execution_runtime["refresh_on_public_target_shift"]
            ),
            public_target_update_distance=float(
                execution_runtime["public_target_update_distance"]
            ),
            public_target_update_min_steps=int(
                execution_runtime["public_target_update_min_steps"]
            ),
        )
        state = provider.initialize()
        mission_context = _public_context(env, state)
        controller = OnlineBSERController(phase1b_config)
        initialized = controller.initialize(state, mission_context)
        bridge = RMADDPGGuidanceBridge()
        guidance = bridge.compile_guidance(
            initialized.allocation,
            state,
            mission_context,
            decision_reason="INITIALIZE",
        )
        env.install_guidance(guidance)
        observations = env.refresh_observation_after_guidance()

        rl = job["rl"]
        actor = MADDPG.init_from_env(
            env.unwrapped,
            gamma=float(rl["gamma"]),
            tau=float(rl["tau"]),
            lr_actor=float(rl["lr_actor"]),
            lr_critic=float(rl["lr_critic"]),
            hidden_dim=int(rl["hidden_dim"]),
            residual_action_reg=float(rl["residual_action_reg"]),
        )
        for agent, state_dict in zip(actor.agents, job["policy_states"]):
            agent.policy.load_state_dict(
                {key: torch.as_tensor(value) for key, value in state_dict.items()}
            )
        actor.prep_rollouts(device="cpu")
        actor.reset_noise()

        transitions = []
        allocation_versions = {guidance.allocation_version}
        event_count = 0
        optimizer_invocations = 0
        accepted_replans = 0
        rejected_replans = 0
        collision_count = 0
        action_norms: list[float] = []
        residual_ratios: list[float] = []
        reward_total = 0.0
        base_reward_total = 0.0
        final_step = 0

        for _ in range(int(job["max_steps"])):
            with torch.no_grad():
                parts = actor.step(observations, explore=True)
                actions = torch.stack(
                    [value.squeeze(0) for value in parts]
                ).to(env.unwrapped.device)
            action_norms.append(float(torch.norm(actions, dim=1).mean().item()))
            _, rewards, dones = env.step(actions)
            metadata = env.last_transition_metadata
            if metadata is None:
                raise RuntimeError("v2 training wrapper did not emit transition metadata")

            state = provider.snapshot(force=False)
            mission_context = _public_context(env, state)
            result = controller.step(state, mission_context)
            env.observe_controller_result(
                result,
                controller=controller,
                state_provider=provider,
            )
            guidance = bridge.compile_guidance(
                result.allocation,
                state,
                mission_context,
                decision_reason=result.decision_reason,
            )
            env.install_guidance(guidance)
            next_observations = env.refresh_observation_after_guidance()
            allocation_versions.add(guidance.allocation_version)

            diagnostics = result.diagnostics
            attempted = bool(
                diagnostics is not None
                and (
                    diagnostics.optimizer_invoked
                    or diagnostics.allocation_scope != "none"
                )
            )
            event_count += len(result.events)
            optimizer_invocations += int(
                diagnostics is not None and diagnostics.optimizer_invoked
            )
            accepted_replans += int(bool(result.replanned))
            rejected_replans += int(attempted and not result.replanned)
            collision_count += int(env.unwrapped._collision_flags.sum().item())
            residual_ratios.append(
                float(env.unwrapped.last_residual_contribution_ratio)
            )
            reward_tensor = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1)
            reward_total += float(reward_tensor.sum().item())
            if env.last_base_rewards is not None:
                base_reward_total += float(env.last_base_rewards.sum().item())
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
                    metadata,
                )
            )
            observations = next_observations
            if all(bool(value) for value in dones):
                break

        def mean(values: list[float]) -> float:
            return float(statistics.fmean(values)) if values else 0.0

        task = env.get_task_state()
        execution_row = env.finalize_episode()
        metrics = {
            "method": METHOD,
            "implementation_version": IMPLEMENTATION_VERSION,
            "execution_runtime_revision": str(
                job.get(
                    "execution_runtime_revision",
                    LEGACY_EXECUTION_RUNTIME_REVISION,
                )
            ),
            "episode": episode_index + 1,
            "episode_index": episode_index,
            "episode_id": episode_index,
            "scenario_id": str(scenario.get("scenario_id", "")),
            "scenario_seed": scenario_seed,
            "success": bool(task.mission_complete),
            "found": bool(task.target_found),
            "collision": int(collision_count),
            "episode_length": int(final_step),
            "event_count": int(event_count),
            "optimizer_invocation_count": int(optimizer_invocations),
            "accepted_replans": int(accepted_replans),
            "rejected_replans": int(rejected_replans),
            "allocation_version_count": len(allocation_versions),
            "reward": float(reward_total),
            "base_reward": float(base_reward_total),
            "action_norm": mean(action_norms),
            "residual_ratio": mean(residual_ratios),
            "initial_planner_endpoint_fallback_count": int(
                env.initial_endpoint_fallback_count
            ),
            "contact_bonus_count": int(env.reward_adapter.contact_bonus_count),
            "hold_bonus_count": int(env.reward_adapter.hold_bonus_count),
            "terminal_bonus_count": int(env.reward_adapter.terminal_bonus_count),
            "discovery_correction_count": int(
                env.reward_adapter.discovery_correction_count
            ),
            "searcher_zeroed_step_count": int(
                env.reward_adapter.searcher_zeroed_step_count
            ),
            "wall_seconds": float(time.perf_counter() - started),
        }
        return metrics, transitions, execution_row
    finally:
        env.close()


def _build_learner(config: Mapping[str, Any]):
    env = _make_base_env(config, device="cpu")
    rl = config["rl"]
    try:
        maddpg = MADDPG.init_from_env(
            env.unwrapped,
            gamma=float(rl["gamma"]),
            tau=float(rl["tau"]),
            lr_actor=float(rl["lr_actor"]),
            lr_critic=float(rl["lr_critic"]),
            hidden_dim=int(rl["hidden_dim"]),
            residual_action_reg=float(rl["residual_action_reg"]),
        )
    finally:
        env.close()
    replay = PhaseAwareReplayBuffer(
        max_steps=int(rl["replay_size"]),
        num_agents=4,
        obs_dims=(28, 28, 28, 28),
        ac_dims=(3, 3, 3, 3),
        config=config["replay"],
        storage_device="cpu",
        generator_seed=int(config.get("seed", 2729)) + 31,
    )
    return maddpg, replay


def _apply_transitions(
    maddpg: MADDPG,
    replay: PhaseAwareReplayBuffer,
    transitions,
    episode_summary: Mapping[str, Any],
    rl: Mapping[str, Any],
    *,
    global_step: int,
    update_step: int,
    device: str,
):
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    q_values: list[float] = []
    replay_sample_count = 0
    optimizer_update_count = 0
    sample_fraction_sums = {name: 0.0 for name in replay.STRATA}
    sample_diagnostic_count = 0

    for obs, actions, rewards, next_obs, dones, success_flags, metadata in transitions:
        replay.push(
            obs,
            actions,
            rewards,
            next_obs,
            dones,
            success_flags,
            metadata,
        )
        global_step += 1
        if (
            global_step < int(rl["warmup_steps"])
            or global_step % int(rl["update_frequency"]) != 0
            or len(replay) < int(rl["batch_size"])
        ):
            continue
        maddpg.prep_training(device=device)
        for _ in range(int(rl["updates_per_train"])):
            sample = replay.sample(
                int(rl["batch_size"]), norm_rews=False, device=device
            )
            replay_sample_count += 1
            details = replay.last_sample_diagnostics
            for name in replay.STRATA:
                sample_fraction_sums[name] += float(
                    details.get("effective_fractions", {}).get(name, 0.0)
                )
            sample_diagnostic_count += 1
            try:
                with torch.no_grad():
                    joint = torch.cat((*sample[0], *sample[1]), dim=1)
                    q_values.append(
                        float(
                            statistics.fmean(
                                float(agent.critic1(joint).mean().item())
                                for agent in maddpg.agents
                            )
                        )
                    )
            except (AttributeError, RuntimeError, ValueError):
                pass
            errors = []
            update_actor = update_step % int(rl["policy_delay"]) == 0
            for agent_i in range(4):
                if update_actor:
                    critic_loss, actor_loss, error = maddpg.update(sample, agent_i)
                    actor_losses.append(float(actor_loss))
                else:
                    critic_loss, error = maddpg.update_critic_only(sample, agent_i)
                critic_losses.append(float(critic_loss))
                errors.append(torch.as_tensor(error).detach())
                optimizer_update_count += 1
            replay.update_priorities(
                sample[6], torch.stack(errors).mean(dim=0), sample[7]
            )
            maddpg.update_all_targets(compute_diff=False)
            update_step += 1
        maddpg.prep_rollouts(device=device)

    success_tail_marked = replay.finalize_episode(
        int(episode_summary.get("episode_id", -1)),
        success=bool(episode_summary.get("success", False)),
    )

    def mean(values: list[float]) -> float:
        return float(statistics.fmean(values)) if values else 0.0

    return {
        "global_step": int(global_step),
        "update_step": int(update_step),
        "actor_loss": mean(actor_losses),
        "critic_loss": mean(critic_losses),
        "q_value": mean(q_values),
        "replay_sample_count": int(replay_sample_count),
        "optimizer_update_count": int(optimizer_update_count),
        "success_tail_marked": int(success_tail_marked),
        "sample_fractions": {
            name: (
                0.0
                if sample_diagnostic_count == 0
                else sample_fraction_sums[name] / sample_diagnostic_count
            )
            for name in replay.STRATA
        },
    }


def _checkpoint_metadata(config: Mapping[str, Any], episode: int) -> dict[str, Any]:
    return {
        "schema": "bser.phase1c.checkpoint.v2",
        "method": str(config.get("method", METHOD)),
        "implementation_version": str(
            config.get("implementation_version", IMPLEMENTATION_VERSION)
        ),
        "execution_runtime_revision": _execution_runtime_revision(config),
        "episode": int(episode),
        "completed_episode": int(episode),
        "seed": int(config.get("seed", 2729)),
        "profile": str(config.get("profile", "M20_MOVING_UNKNOWN_MULTI")),
        "observation_dim": int(config.get("observation_dim", 28)),
        "action_dim": int(config.get("action_dim", 3)),
        "critic_dim": int(config.get("critic_dim", 124)),
        "bser_integration_version": str(
            config.get("bser_integration_version", "bser.control_context.v1")
        ),
        "config_hash": _config_hash(config),
        "reward_adapter": _reward_config(config),
        "replay": dict(config["replay"]),
        "training_update": True,
    }


def _save_checkpoint(
    maddpg: MADDPG,
    replay: PhaseAwareReplayBuffer,
    directory: Path,
    config: Mapping[str, Any],
    episode: int,
    *,
    global_step: int,
    update_step: int,
    replay_sample_count: int,
    optimizer_update_count: int,
    episode_rows: list[dict[str, Any]],
    execution_rows: list[dict[str, Any]],
) -> Path:
    path = directory / f"phase1c_v2_episode_{int(episode):04d}.pt"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "metadata": _checkpoint_metadata(config, episode),
        "maddpg_training_state": maddpg.training_state_dict(),
        "replay_state": replay.state_dict(),
        "completed_episode": int(episode),
        "global_step": int(global_step),
        "update_step": int(update_step),
        "replay_sample_count": int(replay_sample_count),
        "optimizer_update_count": int(optimizer_update_count),
        "episode_metrics": [dict(row) for row in episode_rows],
        "execution_diagnostics": [dict(row) for row in execution_rows],
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary checkpoint already exists: {temporary}")
    torch.save(payload, temporary)
    temporary.replace(path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if loaded.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("saved Phase 1C-v2 checkpoint schema validation failed")
    metadata = loaded.get("metadata", {})
    if metadata.get("implementation_version") != str(
        config.get("implementation_version", IMPLEMENTATION_VERSION)
    ):
        raise RuntimeError("saved Phase 1C-v2 implementation metadata mismatch")
    if metadata.get("execution_runtime_revision") != _execution_runtime_revision(
        config
    ):
        raise RuntimeError("saved Phase 1C-v2 execution runtime revision mismatch")
    if metadata.get("config_hash") != _config_hash(config):
        raise RuntimeError("saved Phase 1C-v2 config hash validation failed")
    return path


def _load_checkpoint(
    path: Path,
    maddpg: MADDPG,
    replay: PhaseAwareReplayBuffer,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    schema = payload.get("schema")
    if schema == V1_SCHEMA:
        raise ValueError(
            "Phase 1C-v1 checkpoints are intentionally incompatible with "
            "Phase 1C-v2 reward/replay semantics and cannot be resumed."
        )
    if schema != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported Phase 1C-v2 checkpoint schema: {schema!r}")
    metadata = dict(payload.get("metadata", {}))
    if metadata.get("implementation_version") != str(
        config.get("implementation_version", IMPLEMENTATION_VERSION)
    ):
        raise ValueError("resume checkpoint implementation_version mismatch")
    checkpoint_revision = metadata.get("execution_runtime_revision")
    configured_revision = _execution_runtime_revision(config)
    if checkpoint_revision != configured_revision:
        if configured_revision == DYNAMIC_EXECUTION_RUNTIME_REVISION:
            raise ValueError(
                "Phase 1C-v2 legacy checkpoints cannot be resumed under "
                "dynamic-public-intercept v2.1 runtime semantics."
            )
        raise ValueError("resume checkpoint execution_runtime_revision mismatch")
    if metadata.get("config_hash") != _config_hash(config):
        raise ValueError("resume checkpoint config hash mismatch")
    for key, expected in (
        ("observation_dim", int(config.get("observation_dim", 28))),
        ("action_dim", int(config.get("action_dim", 3))),
        ("critic_dim", int(config.get("critic_dim", 124))),
    ):
        if int(metadata.get(key, expected)) != expected:
            raise ValueError(f"resume checkpoint {key} mismatch")
    if metadata.get("reward_adapter") != _reward_config(config):
        raise ValueError("resume checkpoint reward adapter config mismatch")
    if metadata.get("replay") != dict(config["replay"]):
        raise ValueError("resume checkpoint replay config mismatch")
    maddpg.load_training_state_dict(payload["maddpg_training_state"])
    replay.load_state_dict(payload["replay_state"])
    return payload


def _rolling(values, window: int):
    result = []
    for index in range(len(values)):
        start = max(0, index + 1 - int(window))
        result.append(float(statistics.fmean(values[start : index + 1])))
    return result


def _plot_curves(rows: list[dict[str, Any]], metrics_dir: Path, window: int) -> None:
    if not rows:
        return
    episodes = [int(row["episode_index"]) + 1 for row in rows]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(episodes, [float(row["actor_loss"]) for row in rows], label="actor")
    axis.plot(episodes, [float(row["critic_loss"]) for row in rows], label="critic")
    axis.set(xlabel="Episode", ylabel="Loss", title="Phase 1C-v2 training losses")
    axis.legend(); figure.tight_layout(); figure.savefig(metrics_dir / "loss_curves.png", dpi=180); plt.close(figure)

    rewards = [float(row["reward"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(episodes, rewards, alpha=0.35, label="episode")
    axis.plot(episodes, _rolling(rewards, window), label=f"rolling-{window}")
    axis.set(xlabel="Episode", ylabel="Adjusted reward", title="Phase 1C-v2 reward")
    axis.legend(); figure.tight_layout(); figure.savefig(metrics_dir / "reward_curve.png", dpi=180); plt.close(figure)

    success = _rolling([float(bool(row["success"])) for row in rows], window)
    found = _rolling([float(bool(row["found"])) for row in rows], window)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(episodes, success, label="success")
    axis.plot(episodes, found, label="found")
    axis.set(xlabel="Episode", ylabel="Rolling rate", title="Success / found trend", ylim=(0, 1))
    axis.legend(); figure.tight_layout(); figure.savefig(metrics_dir / "success_found_trend.png", dpi=180); plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.8))
    for name in PhaseAwareReplayBuffer.STRATA:
        axis.plot(
            episodes,
            [float(row.get(f"sample_fraction_{name}", 0.0)) for row in rows],
            label=name,
        )
    axis.set(xlabel="Episode", ylabel="Actual sample fraction", title="Phase-aware replay mix", ylim=(0, 1))
    axis.legend(); figure.tight_layout(); figure.savefig(metrics_dir / "phase_aware_replay_curve.png", dpi=180); plt.close(figure)


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
    if seed_override is not None:
        config["seed"] = int(seed_override)
    if dry_run:
        config["episodes"] = int(config["dry_run"]["episodes"])
        config["max_steps"] = int(config["dry_run"]["max_steps"])
        config["checkpoint_interval"] = min(
            int(config.get("checkpoint_interval", config["episodes"])),
            int(config["episodes"]),
        )
    if episodes_override is not None:
        config["episodes"] = int(episodes_override)
    if max_steps_override is not None:
        config["max_steps"] = int(max_steps_override)
    if workers_override is not None:
        config["workers"] = int(workers_override)
    if device_override is not None:
        config["device"] = str(device_override)
    if int(config["episodes"]) <= 0 or int(config["max_steps"]) <= 0:
        raise ValueError("episodes and max_steps must be positive")
    config["device"] = str(_resolve_training_device(config["device"]))

    configured_output = Path(config["output_dir"])
    if not configured_output.is_absolute():
        configured_output = ROOT / configured_output
    base_output = configured_output if output_dir is None else Path(output_dir)
    output = base_output / "dry_run" if dry_run else base_output
    directories = {
        name: output / name for name in ("checkpoints", "logs", "metrics")
    }
    protected = (
        output / "resolved_training_config.json",
        directories["metrics"] / "episode_metrics.csv",
        directories["metrics"] / "training_summary.json",
    )
    existing_checkpoints = tuple(directories["checkpoints"].glob("*.pt"))
    if resume is None and (any(path.exists() for path in protected) or existing_checkpoints):
        raise FileExistsError(
            f"Phase 1C-v2 output already exists; use --resume or a new output directory: {output}"
        )
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    resolved_path = output / "resolved_training_config.json"
    if not resolved_path.exists():
        _write_json(resolved_path, config)

    _seed_all(int(config["seed"]))
    maddpg, replay = _build_learner(config)
    maddpg.prep_rollouts(device=str(config["device"]))
    rows: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []
    checkpoints: list[str] = []
    global_step = 0
    update_step = 0
    replay_sample_count = 0
    optimizer_update_count = 0
    start_episode = 0
    if resume is not None:
        payload = _load_checkpoint(resume, maddpg, replay, config)
        start_episode = int(payload["completed_episode"])
        global_step = int(payload["global_step"])
        update_step = int(payload["update_step"])
        replay_sample_count = int(payload["replay_sample_count"])
        optimizer_update_count = int(payload["optimizer_update_count"])
        rows = [dict(row) for row in payload.get("episode_metrics", [])]
        execution_rows = [
            dict(row) for row in payload.get("execution_diagnostics", [])
        ]
        checkpoints = [str(Path(resume).resolve())]
        if len(rows) != start_episode:
            raise ValueError("checkpoint episode metrics do not match completed episode")
    if start_episode >= int(config["episodes"]):
        raise ValueError("resume checkpoint already reached configured episode count")

    initial_actor = tuple(
        parameter.detach().cpu().clone()
        for agent in maddpg.agents
        for parameter in agent.policy.parameters()
    )
    manifest = build_scenario_manifests(
        count=int(config["episodes"]),
        generator_seed=int(config["seed"]),
        split="train",
        profiles=(str(config["profile"]),),
    )[str(config["profile"])]
    scenarios = list(manifest["scenarios"])
    started = time.perf_counter()
    remaining = int(config["episodes"]) - start_episode
    workers = max(1, min(int(config["workers"]), remaining))

    mp_context = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp_context,
    ) as executor:
        for batch_start in range(start_episode, int(config["episodes"]), workers):
            batch_indices = range(
                batch_start,
                min(batch_start + workers, int(config["episodes"])),
            )
            policy_states = _policy_snapshot(maddpg)
            jobs = [
                {
                    "episode_index": index,
                    "scenario": scenarios[index],
                    "base_candidate": config["base_candidate"],
                    "profile": config["profile"],
                    "max_steps": config["max_steps"],
                    "rl": config["rl"],
                    "reward": _reward_config(config),
                    "execution_runtime": copy.deepcopy(config["execution_runtime"]),
                    "execution_runtime_revision": _execution_runtime_revision(
                        config
                    ),
                    "policy_states": policy_states,
                }
                for index in batch_indices
            ]
            try:
                results = list(executor.map(_collect_episode, jobs))
            except Exception as exc:
                failure_path = directories["logs"] / (
                    f"worker_failure_episode_{batch_start + 1:04d}.json"
                )
                _write_json(
                    failure_path,
                    {
                        "schema": "bser.phase1c.v2.worker_failure.v1",
                        "method": METHOD,
                        "implementation_version": IMPLEMENTATION_VERSION,
                        "batch_first_episode": batch_start + 1,
                        "batch_last_episode": max(batch_indices) + 1,
                        "scenarios": [
                            {
                                "episode": int(job["episode_index"]) + 1,
                                "scenario_id": str(job["scenario"].get("scenario_id")),
                                "scenario_seed": int(job["scenario"]["scenario_seed"]),
                            }
                            for job in jobs
                        ],
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
            for episode_metrics, transitions, execution_row in sorted(
                results, key=lambda value: int(value[0]["episode_index"])
            ):
                update = _apply_transitions(
                    maddpg,
                    replay,
                    transitions,
                    {
                        "episode_id": episode_metrics["episode_id"],
                        "success": episode_metrics["success"],
                    },
                    config["rl"],
                    global_step=global_step,
                    update_step=update_step,
                    device=str(config["device"]),
                )
                global_step = int(update["global_step"])
                update_step = int(update["update_step"])
                replay_sample_count += int(update["replay_sample_count"])
                optimizer_update_count += int(update["optimizer_update_count"])
                episode_metrics["actor_loss"] = float(update["actor_loss"])
                episode_metrics["critic_loss"] = float(update["critic_loss"])
                episode_metrics["q_value"] = float(update["q_value"])
                episode_metrics["replay_size"] = len(replay)
                episode_metrics["replay_sample_count"] = int(
                    update["replay_sample_count"]
                )
                episode_metrics["optimizer_update_count"] = int(
                    update["optimizer_update_count"]
                )
                episode_metrics["success_tail_marked"] = int(
                    update["success_tail_marked"]
                )
                for name, fraction in update["sample_fractions"].items():
                    episode_metrics[f"sample_fraction_{name}"] = float(fraction)
                counts = replay.phase_counts()
                for name, count in counts.items():
                    episode_metrics[f"replay_count_{name}"] = int(count)
                rows.append(episode_metrics)
                execution_rows.append(execution_row)

                episode_number = int(episode_metrics["episode_index"]) + 1
                if (
                    episode_number % int(config["checkpoint_interval"]) == 0
                    or episode_number == int(config["episodes"])
                ):
                    path = _save_checkpoint(
                        maddpg,
                        replay,
                        directories["checkpoints"],
                        config,
                        episode_number,
                        global_step=global_step,
                        update_step=update_step,
                        replay_sample_count=replay_sample_count,
                        optimizer_update_count=optimizer_update_count,
                        episode_rows=rows,
                        execution_rows=execution_rows,
                    )
                    checkpoints.append(str(path.resolve()))
            _write_csv(directories["metrics"] / "episode_metrics.csv", rows)
            _write_csv(
                directories["metrics"] / "execution_diagnostics.csv",
                execution_rows,
            )

    changed_actor_tensors = sum(
        int(not torch.equal(before, after.detach().cpu()))
        for before, after in zip(
            initial_actor,
            (
                parameter
                for agent in maddpg.agents
                for parameter in agent.policy.parameters()
            ),
        )
    )
    _plot_curves(rows, directories["metrics"], int(config["rolling_window"]))
    _write_json(directories["checkpoints"] / "checkpoint_list.json", checkpoints)
    found_count = sum(bool(row["found"]) for row in rows)
    success_count = sum(bool(row["success"]) for row in rows)
    phase_counts = replay.phase_counts()
    pipeline_passed = bool(
        len(rows) == int(config["episodes"])
        and len(execution_rows) == len(rows)
        and replay_sample_count > 0
        and optimizer_update_count > 0
        and changed_actor_tensors > 0
        and checkpoints
    )
    dry_run_passed = bool(pipeline_passed) if dry_run else False
    summary = {
        "schema": "bser.phase1c.training.summary.v2",
        "method": METHOD,
        "implementation_version": IMPLEMENTATION_VERSION,
        "execution_runtime_revision": _execution_runtime_revision(config),
        "dry_run": bool(dry_run),
        "pipeline_passed": pipeline_passed,
        "dry_run_passed": dry_run_passed,
        "performance_passed": None,
        "episodes": int(config["episodes"]),
        "max_steps": int(config["max_steps"]),
        "workers": workers,
        "requested_device": str(config["device"]),
        "actual_device": str(maddpg.device),
        "torch_version": str(torch.__version__),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(maddpg.device)
            if maddpg.device.type == "cuda"
            else None
        ),
        "training_update": True,
        "resumed_from": None if resume is None else str(Path(resume).resolve()),
        "start_episode": int(start_episode),
        "completed_episode_count": len(rows),
        "diagnostics_generated": bool(len(execution_rows) == len(rows)),
        "global_step": int(global_step),
        "replay_size": len(replay),
        "replay_sample_count": int(replay_sample_count),
        "optimizer_update_count": int(optimizer_update_count),
        "parameter_update_count": int(changed_actor_tensors),
        "checkpoint_count": len(checkpoints),
        "checkpoint_load_verified": bool(checkpoints),
        "found_rate": 0.0 if not rows else found_count / len(rows),
        "success_rate": 0.0 if not rows else success_count / len(rows),
        "success_if_found_rate": (
            0.0 if found_count == 0 else success_count / found_count
        ),
        "mean_reward": (
            None
            if not rows
            else float(statistics.fmean(float(row["reward"]) for row in rows))
        ),
        "mean_actor_loss": (
            None
            if not rows
            else float(
                statistics.fmean(float(row["actor_loss"]) for row in rows)
            )
        ),
        "mean_critic_loss": (
            None
            if not rows
            else float(
                statistics.fmean(float(row["critic_loss"]) for row in rows)
            )
        ),
        "replay_stratum_counts": phase_counts,
        "success_tail_mark_count": int(replay.success_tail_mark_count),
        "protocol_correction_counts": {
            "discovery": sum(int(row["discovery_correction_count"]) for row in rows),
            "searcher_zeroed_steps": sum(
                int(row["searcher_zeroed_step_count"]) for row in rows
            ),
            "contact_bonus": sum(int(row["contact_bonus_count"]) for row in rows),
            "hold_bonus": sum(int(row["hold_bonus_count"]) for row in rows),
            "terminal_bonus": sum(int(row["terminal_bonus_count"]) for row in rows),
        },
        "executor_invalid_reason_counts": {
            "ASSIGNMENT_UNREACHABLE": sum(
                int(row.get("executor_invalid_assignment_unreachable_count", 0))
                for row in execution_rows
            ),
            "QUERY_UNREACHABLE": sum(
                int(row.get("executor_invalid_query_unreachable_count", 0))
                for row in execution_rows
            ),
            "PLANNING_COST_INCREASE": sum(
                int(row.get("executor_invalid_cost_increase_count", 0))
                for row in execution_rows
            ),
            "STALE_ENDPOINT_SNAPSHOT_DEFERRED": sum(
                int(
                    row.get(
                        "executor_invalid_stale_snapshot_deferred_count", 0
                    )
                )
                for row in execution_rows
            ),
        },
        "executor_validity_deferred_count": sum(
            int(row.get("executor_validity_deferred_count", 0))
            for row in execution_rows
        ),
        "public_target_update_event_count": sum(
            int(row.get("public_target_update_event_count", 0))
            for row in execution_rows
        ),
        "public_target_update_accepted_count": sum(
            int(row.get("public_target_update_accepted_count", 0))
            for row in execution_rows
        ),
        "planning_refresh_counts": {
            "full": sum(
                int(row.get("full_planning_refresh_count", 0))
                for row in execution_rows
            ),
            "handoff_forced": sum(
                int(row.get("handoff_forced_refresh_count", 0))
                for row in execution_rows
            ),
            "target_shift_forced": sum(
                int(row.get("target_shift_forced_refresh_count", 0))
                for row in execution_rows
            ),
        },
        "wall_seconds": float(time.perf_counter() - started),
    }
    _write_json(directories["metrics"] / "training_summary.json", summary)
    _write_json(
        directories["logs"] / "training_log.json",
        {
            "summary": summary,
            "checkpoint_metadata": _checkpoint_metadata(config, len(rows)),
        },
    )
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
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0 if summary["pipeline_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
