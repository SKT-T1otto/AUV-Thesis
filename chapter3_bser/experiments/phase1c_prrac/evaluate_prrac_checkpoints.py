"""Read-only deterministic evaluation for independent PRRAC checkpoints."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
from dataclasses import replace
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from chapter3_bser.controllers.state_provider import OnlinePlanningStateProvider
from chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.train_phase1c_v2 import _seed_all
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
from chapter3_bser.experiments.phase1c_prrac.evaluation_metrics import (
    aggregate_checkpoint,
    failure_stage,
    paired_checkpoint_comparison,
    recommend_checkpoint,
    router_class_metrics,
)
from chapter3_bser.experiments.phase1c_prrac.evaluation_trace import (
    FailureTraceRecorder,
    failure_trace_row,
    vector3,
)
from chapter3_bser.experiments.phase1c_prrac.training_env import PRRACTrainingEnv
from chapter3_bser.integration.control_context import (
    AgentAssignmentContextV1,
    BSERControlContextV1,
    ExecutorAssignmentContextV1,
)
from chapter3_bser.integration.guided_env import GuidedEnv
from chapter3_bser.integration.rmaddpg_bridge import RMADDPGGuidanceBridge
from chapter3_bser.models.prrac.prrac_maddpg import PRRACMADDPG
from chapter3_bser.models.prrac.stage_mapping import PRRACStage
from chapter3_bser.online.config import execution_runtime_config, load_phase1b2_config
from chapter3_bser.online.controller import OnlineBSERController
from chapter3_bser.online.mission_context import OnlineMissionContext
from core.config.ch3_config import build_ch3_config
from core.env.mission_env import MissionCoreEnv, environment_kwargs_from_config
from core.registry.experiment_registry import assert_registered_ch3_method
from core.scenarios.ch3_generator_impl import build_scenario_manifests


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "chapter3" / "bser_phase1c_prrac_eval.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "chapter3" / "phase1c_prrac" / "evaluation_v1"
EVALUATION_SCHEMA = "bser.phase1c.prrac.evaluation.v1"
SUMMARY_SCHEMA = "bser.phase1c.prrac.evaluation.summary.v1"
OLD_CHECKPOINT_MESSAGE = (
    "Phase 1C-v1/v2 checkpoints are incompatible with PRRAC deterministic evaluation."
)
SUPPORTED_MODES = (
    "full_prrac",
    "executor_residual_off",
    "all_residual_off",
    "oracle_current_target_diagnostic",
)
OUTPUT_FILES = (
    "resolved_evaluation_config.json",
    "evaluation_manifest.json",
    "checkpoint_metadata.json",
    "evaluation_progress.json",
    "episode_evaluation.csv",
    "checkpoint_summary.csv",
    "paired_checkpoint_comparison.csv",
    "failure_funnel.csv",
    "failure_trace.jsonl",
    "failure_trace_index.csv",
    "evaluation_summary.json",
)


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _csv_value(value: Any) -> Any:
    safe = _json_safe(value)
    if isinstance(safe, (dict, list)):
        return json.dumps(safe, sort_keys=True, separators=(",", ":"))
    return safe


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["status"]
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in tuple(row.items()):
            if isinstance(value, str) and value[:1] in {"[", "{"}:
                try:
                    row[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass
            elif value == "True":
                row[key] = True
            elif value == "False":
                row[key] = False
            elif value in {"None", ""}:
                row[key] = None
            elif isinstance(value, str):
                try:
                    row[key] = float(value) if any(mark in value for mark in ".eE") else int(value)
                except ValueError:
                    pass
    return rows


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
    return False


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema": EVALUATION_SCHEMA,
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
            raise ValueError(f"invalid PRRAC evaluation config {key}: {config.get(key)!r}")
    if config.get("explore") is not False:
        raise ValueError("PRRAC deterministic evaluation requires explore=false")
    if config.get("training_update") is not False:
        raise ValueError("PRRAC deterministic evaluation requires training_update=false")
    config["execution_runtime"] = execution_runtime_config(config)
    return config


def _validate_checkpoint_payload(
    payload: Mapping[str, Any], config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    schema = str(payload.get("schema", ""))
    if schema in {"bser.phase1c.training_state.v1", "bser.phase1c.training_state.v2"}:
        raise ValueError(OLD_CHECKPOINT_MESSAGE)
    if schema != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported PRRAC checkpoint schema: {schema!r}")
    metadata = dict(payload.get("metadata", {}))
    expected = {
        "method": METHOD,
        "implementation_version": IMPLEMENTATION_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "observation_dim": 28,
        "action_dim": 3,
        "critic_dim": 124,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"PRRAC checkpoint {key} mismatch: expected {value!r}, "
                f"got {metadata.get(key)!r}"
            )
    state = payload.get("prrac_training_state")
    if not isinstance(state, Mapping):
        raise ValueError("PRRAC checkpoint is missing prrac_training_state")
    architecture = metadata.get("architecture")
    loss = metadata.get("loss")
    if not isinstance(architecture, Mapping) or not isinstance(loss, Mapping):
        raise ValueError("PRRAC checkpoint metadata must contain architecture and loss")
    if dict(state.get("architecture", {})) != dict(architecture):
        raise ValueError("PRRAC checkpoint architecture mismatch")
    if dict(state.get("loss", {})) != dict(loss):
        raise ValueError("PRRAC checkpoint loss mismatch")
    if config is not None and str(metadata.get("execution_runtime_revision", "")) != str(
        config["execution_runtime_revision"]
    ):
        raise ValueError("checkpoint execution_runtime_revision mismatch")
    return dict(payload)


def load_prrac_checkpoint(
    path: Path,
    *,
    device: str | torch.device = "cpu",
    config: Mapping[str, Any] | None = None,
) -> tuple[PRRACMADDPG, dict[str, Any]]:
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    payload = _validate_checkpoint_payload(payload, config)
    metadata = dict(payload["metadata"])
    state = dict(payload["prrac_training_state"])
    learner = PRRACMADDPG(
        architecture=metadata["architecture"],
        loss=metadata["loss"],
        gamma=float(state["gamma"]),
        tau=float(state["tau"]),
    )
    learner.load_training_state_dict(state)
    learner.prep_rollouts(resolved_device)
    return learner, payload


def _checkpoint_info(path: Path, payload: Mapping[str, Any], mode: str) -> dict[str, Any]:
    metadata = dict(payload["metadata"])
    oracle = mode == "oracle_current_target_diagnostic"
    return {
        "checkpoint": str(Path(path).resolve()),
        "checkpoint_episode": int(payload.get("completed_episode", metadata.get("completed_episode", 0))),
        "checkpoint_config_hash": str(metadata.get("config_hash", "")),
        "checkpoint_schema": str(payload["schema"]),
        "evaluation_mode": str(mode),
        "diagnostic_only": oracle,
        "privileged_oracle": oracle,
    }


def _resolve_checkpoints(
    config: Mapping[str, Any],
    explicit: Iterable[Path] | None,
    checkpoint_dir: Path | None,
    checkpoint_pattern: str | None,
) -> list[Path]:
    values = [Path(value) for value in (explicit or ())]
    values.extend(Path(value) for value in config.get("checkpoints", ()))
    for pattern in config.get("checkpoint_globs", ()):
        values.extend(sorted(ROOT.glob(str(pattern))))
    if checkpoint_dir is not None:
        values.extend(
            sorted(Path(checkpoint_dir).glob(checkpoint_pattern or "phase1c_prrac_episode_*.pt"))
        )
    unique: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        path = (value if value.is_absolute() else ROOT / value).resolve()
        if path in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"PRRAC checkpoint not found: {path}")
        unique.append(path)
        seen.add(path)
    if not unique:
        raise ValueError("no PRRAC checkpoints were supplied for evaluation")
    return unique


def _build_evaluation_manifest(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profile = str(config["profile"])
    generated = build_scenario_manifests(
        count=int(config["evaluation_episodes"]),
        generator_seed=int(config["scenario_seed"]),
        split=str(config["split"]),
        profiles=(profile,),
    )[profile]
    scenarios = [copy.deepcopy(dict(value)) for value in generated["scenarios"]]
    body = {
        "schema": "bser.phase1c.prrac.evaluation_manifest.v1",
        "profile": profile,
        "split": str(config["split"]),
        "scenario_seed": int(config["scenario_seed"]),
        "evaluation_episodes": len(scenarios),
        "scenarios": scenarios,
        "source_manifest": generated,
    }
    body["manifest_sha256"] = _hash(body)
    return scenarios, body


def _make_env(config: Mapping[str, Any], reward: Mapping[str, Any]) -> PRRACTrainingEnv:
    env_config = build_ch3_config(str(config["base_candidate"]), str(config["profile"]))
    base = MissionCoreEnv(
        **environment_kwargs_from_config(
            env_config,
            device="cpu",
            max_steps=int(config["max_steps"]),
            return_numpy=False,
        )
    )
    return PRRACTrainingEnv(
        Phase1CV2TrainingEnv(GuidedEnv(base, enabled=True), reward_config=reward)
    )


def _public_context(env: Any, state: Any) -> OnlineMissionContext:
    return OnlineMissionContext.from_public_views(
        env.get_task_state(), env.get_search_execution_state(), state
    )


def _oracle_context(
    context: BSERControlContextV1, target: Iterable[float]
) -> BSERControlContextV1:
    vector = tuple(float(value) for value in target)
    if len(vector) != 3:
        raise ValueError("oracle diagnostic target must be 3D")
    executor_id = int(context.executor_assignment.executor_id)
    assignments = tuple(
        item
        if int(item.agent_id) != executor_id
        else replace(
            item,
            assignment_kind="oracle_current_target_diagnostic",
            assignment_id="PRIVILEGED_ORACLE_CURRENT_TARGET",
            final_waypoint=vector,
            planned_path=(),
            tracking_waypoint=vector,
            hold_state=False,
            reachable=True,
            execution_request=True,
        )
        for item in context.agent_assignments
    )
    executor = ExecutorAssignmentContextV1(
        executor_id=executor_id,
        source="PRIVILEGED_ORACLE_CURRENT_TARGET_DIAGNOSTIC_ONLY",
        target_region=vector,
        planned_path=(),
        tracking_waypoint=vector,
        hold_position=context.executor_assignment.hold_position,
        hold_state=False,
        reachable=True,
        execution_request=True,
    )
    return replace(
        context,
        mission_phase="EXECUTION",
        agent_assignments=assignments,
        executor_assignment=executor,
        decision_reason="PRIVILEGED_ORACLE_CURRENT_TARGET_DIAGNOSTIC_ONLY",
    )


def _install_next_guidance(
    env: Any,
    public_guidance: BSERControlContextV1,
    *,
    mode: str,
    true_target: Iterable[float] | None,
) -> tuple[Any, BSERControlContextV1]:
    """Refresh policy observation from public guidance before any oracle install."""

    env.install_guidance(public_guidance)
    policy_observations = env.refresh_observation_after_guidance()
    installed = public_guidance
    if mode == "oracle_current_target_diagnostic" and true_target is not None:
        installed = _oracle_context(public_guidance, true_target)
        env.install_guidance(installed)
    return policy_observations, installed


def _apply_residual_mode(actions: torch.Tensor, mode: str) -> torch.Tensor:
    result = actions.clone()
    if mode in {"full_prrac", "oracle_current_target_diagnostic"}:
        return result
    if mode == "executor_residual_off":
        result[3].zero_()
        return result
    if mode == "all_residual_off":
        result.zero_()
        return result
    raise ValueError(f"unsupported PRRAC evaluation mode: {mode!r}")


def _policy_outputs(
    actor: PRRACMADDPG, observations: Iterable[Any], device: torch.device
) -> list[Any]:
    """Forward only each agent's public 28D observation; stage is label-only."""

    values = tuple(observations)
    if len(values) != 4:
        raise ValueError("PRRAC evaluation requires four public observations")
    outputs = []
    for agent_i, observation in enumerate(values):
        policy_input = torch.as_tensor(
            observation, dtype=torch.float32, device=device
        ).reshape(1, 28)
        outputs.append(actor.agents[agent_i].actor(policy_input))
    return outputs


def _event_names(result: Any) -> list[str]:
    return [
        str(getattr(event, "value", event)).upper()
        for event in (getattr(result, "events", ()) or ())
    ]


def _trace_step(
    *,
    info: Mapping[str, Any],
    scenario: Mapping[str, Any],
    step: int,
    task: Any,
    metadata: Any,
    result: Any,
    controller: Any,
    public_guidance: BSERControlContextV1,
    installed_guidance: BSERControlContextV1,
    env: Any,
    actor_output: Any,
    residual_action: torch.Tensor,
) -> dict[str, Any]:
    runtime = env.unwrapped
    executor = public_guidance.executor_assignment
    detection = getattr(result, "event_detection", None)
    true_target = env.get_target_state().position
    agent_position = getattr(runtime, "_agent_pos", None)
    executor_position = None if agent_position is None else vector3(agent_position[3])
    target_position = vector3(true_target)
    intercept_position = vector3(executor.tracking_waypoint)

    def distance(left: list[float] | None, right: list[float] | None) -> float | None:
        if left is None or right is None:
            return None
        return float(
            np.linalg.norm(
                np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
            )
        )

    public_target_getter = getattr(env, "get_public_executor_navigation_target", None)
    public_target = public_target_getter() if callable(public_target_getter) else None
    installed_targets = getattr(runtime, "_nav_targets", None)
    prior = getattr(runtime, "_last_prior_acc", None)
    residual = getattr(runtime, "_last_residual_acc", None)
    final = None if prior is None or residual is None else prior[3] + residual[3]
    probabilities = actor_output.router_probabilities.detach().cpu().reshape(-1, 3)[0]
    return failure_trace_row(
        checkpoint_episode=int(info["checkpoint_episode"]),
        scenario_id=str(scenario.get("scenario_id", "")),
        scenario_seed=int(scenario["scenario_seed"]),
        step=int(step),
        mission_phase=str(public_guidance.mission_phase),
        stage_before=int(metadata.stage_before),
        stage_after=int(metadata.stage_after),
        task_found=bool(task.target_found),
        executor_knows_target=bool(task.executor_knows_target),
        mission_complete=bool(task.mission_complete),
        event_names=_event_names(result),
        replanned=bool(getattr(result, "replanned", False)),
        decision_reason=str(getattr(result, "decision_reason", "")),
        executor_invalid_reason=str(getattr(detection, "executor_invalid_reason", "")),
        executor_position=executor_position,
        true_target_position=target_position,
        public_executor_target=vector3(public_target),
        controller_execution_target=vector3(getattr(controller, "execution_target", None)),
        installed_tracking_target=(
            None if installed_targets is None else vector3(installed_targets[3])
        ),
        assignment_kind=str(public_guidance.assignment_for(3).assignment_kind),
        assignment_id=str(public_guidance.assignment_for(3).assignment_id),
        assignment_reachable=bool(executor.reachable),
        execution_request=bool(executor.execution_request),
        executor_target_distance=distance(executor_position, target_position),
        executor_intercept_distance=distance(executor_position, intercept_position),
        navigation_prior_norm=(None if prior is None else float(torch.linalg.vector_norm(prior[3]).item())),
        residual_action_norm=float(torch.linalg.vector_norm(residual_action).item()),
        final_action_norm=(None if final is None else float(torch.linalg.vector_norm(final).item())),
        trust_gate=float(actor_output.trust_gate.detach().cpu().reshape(-1)[0].item()),
        alignment_cosine=float(actor_output.alignment_cosine.detach().cpu().reshape(-1)[0].item()),
        router_probability_search=float(probabilities[0].item()),
        router_probability_intercept=float(probabilities[1].item()),
        router_probability_hold=float(probabilities[2].item()),
        router_prediction=int(probabilities.argmax().item()),
        collision=bool(torch.as_tensor(getattr(runtime, "_collision_flags", False)).any().item()),
        installed_guidance_privileged=bool(
            installed_guidance.decision_reason
            == "PRIVILEGED_ORACLE_CURRENT_TARGET_DIAGNOSTIC_ONLY"
        ),
    )


def _evaluate_episode_job(job: dict[str, Any]) -> dict[str, Any]:
    """Spawn-safe deterministic episode worker with no replay or optimizer calls."""

    if _contains_tensor(job):
        raise TypeError("PRRAC evaluation worker input must not contain torch.Tensor")
    torch.set_num_threads(1)
    scenario = copy.deepcopy(dict(job["scenario"]))
    _seed_all(int(scenario["scenario_seed"]))
    config = copy.deepcopy(dict(job["config"]))
    info = dict(job["checkpoint_info"])
    mode = str(info["evaluation_mode"])
    device = torch.device(str(job["device"]))
    actor = PRRACMADDPG(
        architecture=job["architecture"],
        loss=job["loss"],
        gamma=float(job["gamma"]),
        tau=float(job["tau"]),
    )
    actor.load_policy_snapshot(job["policy_snapshot"])
    actor.prep_rollouts(device)
    env = _make_env(config, job["reward"])
    recorder = FailureTraceRecorder(**dict(job["failure_trace"]))
    recorder.begin_episode()
    try:
        episode_index = int(job["episode_index"])
        env.reset(scenario=scenario, episode_id=episode_index, episode_index=episode_index)
        phase1b_config = load_phase1b2_config()
        runtime_config = execution_runtime_config(config)
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
        controller = OnlineBSERController(phase1b_config)
        initialized = controller.initialize(state, context)
        bridge = RMADDPGGuidanceBridge()
        guidance = bridge.compile_guidance(
            initialized.allocation, state, context, decision_reason="INITIALIZE"
        )
        env.install_guidance(guidance)
        observations = env.refresh_observation_after_guidance()
        diagnostics = PRRACDiagnostics()
        current_stage = PRRACStage.SEARCH
        reward_total = 0.0
        episode_length = 0
        collision_episode = False

        for _ in range(int(config["max_steps"])):
            with torch.no_grad():
                outputs = _policy_outputs(actor, observations, device)
                for output in outputs:
                    diagnostics.observe_actor(output, [int(current_stage)])
                residual_actions = torch.stack(
                    [output.gated_residual_action.squeeze(0) for output in outputs]
                )
                actions = _apply_residual_mode(residual_actions, mode).to(
                    env.unwrapped.device
                )
            _, rewards, dones = env.step(actions)
            metadata = env.last_prrac_transition_metadata
            if metadata is None:
                raise RuntimeError("PRRAC evaluation environment emitted no stage metadata")
            if metadata.stage_before != current_stage:
                raise RuntimeError("PRRAC evaluation stage_before drift")
            current_stage = metadata.stage_after
            reward_total += float(torch.as_tensor(rewards).sum().item())
            task = env.get_task_state()
            episode_length = int(task.step)
            collision_episode = bool(
                collision_episode
                or torch.as_tensor(env.unwrapped._collision_flags).any().item()
            )

            state = provider.snapshot(force=False)
            context = _public_context(env, state)
            result = controller.step(state, context)
            env.observe_controller_result(
                result, controller=controller, state_provider=provider
            )
            public_guidance = bridge.compile_guidance(
                result.allocation,
                state,
                context,
                decision_reason=result.decision_reason,
            )
            use_oracle = bool(
                mode == "oracle_current_target_diagnostic"
                and context.executor_knows_target
                and not context.mission_complete
            )
            target = env.get_target_state().position if use_oracle else None
            next_observations, installed_guidance = _install_next_guidance(
                env, public_guidance, mode=mode, true_target=target
            )
            recorder.record(
                _trace_step(
                    info=info,
                    scenario=scenario,
                    step=episode_length,
                    task=task,
                    metadata=metadata,
                    result=result,
                    controller=controller,
                    public_guidance=public_guidance,
                    installed_guidance=installed_guidance,
                    env=env,
                    actor_output=outputs[3],
                    residual_action=actions[3],
                )
            )
            observations = next_observations
            if all(bool(value) for value in dones):
                break

        task = env.get_task_state()
        row = env.finalize_episode()
        row.update(info)
        row.update(diagnostics.summary())
        row.update(router_class_metrics(row["router_confusion_matrix"]))
        row.update(
            {
                "scenario_id": str(scenario.get("scenario_id", "")),
                "scenario_seed": int(scenario["scenario_seed"]),
                "episode_length": int(episode_length),
                "reward": float(reward_total),
                "found": bool(task.target_found),
                "contact_episode": bool(
                    getattr(env.unwrapped, "capture_contact_step_count", 0)
                ),
                "hold_episode": bool(
                    getattr(env.unwrapped, "capture_full_hold_step_count", 0)
                ),
                "success": bool(task.mission_complete),
                "collision_episode": bool(collision_episode),
                "explore": False,
                "training_update": False,
                "optimizer_update_count": 0,
                "replay_sample_count": 0,
                "parameter_update_count": 0,
            }
        )
        row["failure_stage"] = failure_stage(row)
        traces, trace_index = recorder.finish_episode(
            found=bool(row["found"]),
            success=bool(row["success"]),
            checkpoint=str(info["checkpoint"]),
            checkpoint_episode=int(info["checkpoint_episode"]),
            evaluation_mode=mode,
            scenario_id=str(row["scenario_id"]),
            scenario_seed=int(row["scenario_seed"]),
        )
        payload = {"episode": row, "failure_trace": traces, "trace_index": trace_index}
        if _contains_tensor(payload):
            raise RuntimeError("PRRAC evaluation worker output contains torch.Tensor")
        return payload
    finally:
        env.close()


def _combo_key(info: Mapping[str, Any], manifest_hash: str) -> dict[str, Any]:
    return {
        "checkpoint": str(Path(info["checkpoint"]).resolve()),
        "checkpoint_config_hash": str(info["checkpoint_config_hash"]),
        "checkpoint_episode": int(info["checkpoint_episode"]),
        "evaluation_mode": str(info["evaluation_mode"]),
        "manifest_sha256": str(manifest_hash),
    }


def _failure_funnel(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["checkpoint"]), str(row["evaluation_mode"])), []).append(row)
    output = []
    stages = ("NOT_FOUND", "FOUND_NO_CONTACT", "CONTACT_NO_HOLD", "HOLD_NO_SUCCESS", "SUCCESS")
    for (checkpoint, mode), values in sorted(grouped.items()):
        result = {"checkpoint": checkpoint, "evaluation_mode": mode, "evaluation_episodes": len(values)}
        for stage in stages:
            result[stage.lower()] = sum(row.get("failure_stage") == stage for row in values)
        output.append(result)
    return output


def _paired_rows(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in episode_rows:
        grouped.setdefault((str(row["checkpoint"]), str(row["evaluation_mode"])), []).append(row)
    output = []
    modes = sorted({mode for _, mode in grouped})
    for mode in modes:
        checkpoints = sorted(checkpoint for checkpoint, item_mode in grouped if item_mode == mode)
        for base, candidate in itertools.combinations(checkpoints, 2):
            output.append(
                paired_checkpoint_comparison(
                    grouped[(base, mode)],
                    grouped[(candidate, mode)],
                    base_checkpoint=base,
                    candidate_checkpoint=candidate,
                    evaluation_mode=mode,
                )
            )
    return output


def _plot(summary_rows: list[dict[str, Any]], output: Path) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    labels = [f"{row.get('checkpoint_episode')}:{row.get('evaluation_mode')}" for row in summary_rows]

    def line(name: str, title: str, fields: tuple[str, ...]) -> None:
        figure, axis = plt.subplots(figsize=(max(8, len(labels) * 0.7), 4.8))
        for field in fields:
            values = [np.nan if row.get(field) in {None, ""} else float(row[field]) for row in summary_rows]
            axis.plot(range(len(labels)), values, marker="o", label=field)
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_title(title)
        if len(fields) > 1:
            axis.legend()
        figure.tight_layout()
        figure.savefig(plots / name, dpi=180)
        plt.close(figure)

    line("found_success_curve.png", "Found / success", ("found_rate", "success_rate"))
    line("success_if_found_curve.png", "Success if found", ("success_if_found_rate",))
    line("found_to_contact_curve.png", "Found to contact", ("contact_if_found_rate",))
    line("collision_rate_curve.png", "Collision rate", ("collision_episode_rate",))
    line("executor_invalid_curve.png", "Executor invalid", ("mean_executor_invalid_count_if_found",))
    line("assignment_unreachable_curve.png", "Assignment unreachable", ("mean_assignment_unreachable_if_found",))
    line("executor_distance_curve.png", "Executor distance", ("mean_executor_min_distance_if_found", "mean_executor_final_distance_if_found"))
    line("router_class_recall_curve.png", "Router recall", tuple(f"router_recall_{name}" for name in ("search", "intercept", "hold")))
    line("router_balanced_accuracy_curve.png", "Router balanced accuracy", ("router_balanced_accuracy",))
    line("trust_gate_curve.png", "Trust gate", ("gate_mean", "gate_p10", "gate_p90"))
    line("residual_ratio_curve.png", "Post-found residual ratio", ("mean_post_found_residual_ratio_if_found",))


def _evaluation_summary(
    *,
    checkpoint_paths: list[Path],
    scenarios: list[dict[str, Any]],
    modes: tuple[str, ...],
    summary_rows: list[dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    return {
        "schema": SUMMARY_SCHEMA,
        "method": METHOD,
        "implementation_version": IMPLEMENTATION_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_count": len(checkpoint_paths),
        "scenario_count": len(scenarios),
        "evaluation_modes": list(modes),
        "same_scenarios_for_all_checkpoints": True,
        "explore": False,
        "training_update": False,
        "optimizer_update_count": 0,
        "replay_sample_count": 0,
        "parameter_update_count": 0,
        "output_dir": str(output.resolve()),
        **recommend_checkpoint(summary_rows),
    }


def _write_outputs(
    output: Path,
    *,
    checkpoint_paths: list[Path],
    scenarios: list[dict[str, Any]],
    modes: tuple[str, ...],
    episode_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    trace_index: list[dict[str, Any]],
    progress: dict[str, Any],
) -> None:
    _write_csv(output / "episode_evaluation.csv", episode_rows)
    _write_csv(output / "checkpoint_summary.csv", summary_rows)
    _write_csv(output / "paired_checkpoint_comparison.csv", _paired_rows(episode_rows))
    _write_csv(output / "failure_funnel.csv", _failure_funnel(episode_rows))
    _write_csv(output / "failure_trace_index.csv", trace_index)
    _atomic_text(
        output / "failure_trace.jsonl",
        "".join(_canonical_json(row) + "\n" for row in trace_rows),
    )
    _write_json(output / "evaluation_progress.json", progress)
    _write_json(
        output / "evaluation_summary.json",
        _evaluation_summary(
            checkpoint_paths=checkpoint_paths,
            scenarios=scenarios,
            modes=modes,
            summary_rows=summary_rows,
            output=output,
        ),
    )
    _plot(summary_rows, output)


def run_evaluation(
    *,
    config_path: Path = DEFAULT_CONFIG,
    checkpoints: Iterable[Path] | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_pattern: str | None = None,
    output_dir: Path | None = None,
    episodes_override: int | None = None,
    scenario_seed_override: int | None = None,
    workers_override: int | None = None,
    device_override: str | None = None,
    modes_override: Iterable[str] | None = None,
    resume_evaluation: bool = False,
    disable_failure_trace: bool = False,
) -> dict[str, Any]:
    assert_registered_ch3_method(METHOD)
    config = copy.deepcopy(_load_config(config_path))
    for key, value in (
        ("evaluation_episodes", episodes_override),
        ("scenario_seed", scenario_seed_override),
        ("workers", workers_override),
    ):
        if value is not None:
            config[key] = int(value)
    if device_override is not None:
        config["device"] = str(device_override)
    modes = tuple(modes_override or config.get("modes", ("full_prrac",)))
    if not modes or any(mode not in SUPPORTED_MODES for mode in modes):
        raise ValueError(f"unsupported PRRAC evaluation modes: {modes}")
    config["modes"] = list(modes)
    if disable_failure_trace:
        config["failure_trace"]["enabled"] = False
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    workers = max(1, int(config["workers"]))
    if device.type == "cuda" and workers > 1:
        raise ValueError("PRRAC CUDA evaluation requires workers=1")
    config["workers"] = workers
    output = Path(output_dir) if output_dir is not None else Path(config["output_dir"])
    if not output.is_absolute():
        output = ROOT / output
    existing = [output / name for name in OUTPUT_FILES if (output / name).exists()]
    if existing and not resume_evaluation:
        raise FileExistsError(f"PRRAC evaluation output exists: {existing[0]}")
    output.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = _resolve_checkpoints(
        config, checkpoints, checkpoint_dir, checkpoint_pattern
    )
    scenarios, manifest = _build_evaluation_manifest(config)
    manifest_hash = str(manifest["manifest_sha256"])
    if resume_evaluation:
        saved_manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
        if saved_manifest.get("manifest_sha256") != manifest_hash:
            raise ValueError("resume evaluation manifest hash mismatch")
    else:
        _write_json(output / "resolved_evaluation_config.json", config)
        _write_json(output / "evaluation_manifest.json", manifest)

    episode_rows = _read_csv(output / "episode_evaluation.csv") if resume_evaluation else []
    summary_rows = _read_csv(output / "checkpoint_summary.csv") if resume_evaluation else []
    trace_index = _read_csv(output / "failure_trace_index.csv") if resume_evaluation else []
    trace_rows = []
    if resume_evaluation and (output / "failure_trace.jsonl").is_file():
        trace_rows = [
            json.loads(line)
            for line in (output / "failure_trace.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    progress = (
        json.loads((output / "evaluation_progress.json").read_text(encoding="utf-8"))
        if resume_evaluation
        else {
            "schema": "bser.phase1c.prrac.evaluation_progress.v1",
            "manifest_sha256": manifest_hash,
            "completed": [],
        }
    )
    if progress.get("manifest_sha256") != manifest_hash:
        raise ValueError("resume evaluation progress manifest hash mismatch")
    completed = {_canonical_json(value) for value in progress.get("completed", ())}
    checkpoint_metadata = []
    context = mp.get_context("spawn")

    for checkpoint in checkpoint_paths:
        learner, payload = load_prrac_checkpoint(checkpoint, device="cpu", config=config)
        metadata = dict(payload["metadata"])
        state = dict(payload["prrac_training_state"])
        checkpoint_metadata.append(
            {
                "checkpoint": str(checkpoint),
                "schema": payload["schema"],
                "completed_episode": int(payload.get("completed_episode", 0)),
                "metadata": metadata,
            }
        )
        snapshot = learner.policy_snapshot()
        if _contains_tensor(snapshot):
            raise RuntimeError("PRRAC evaluation snapshot contains torch.Tensor")
        for mode in modes:
            info = _checkpoint_info(checkpoint, payload, mode)
            combo = _combo_key(info, manifest_hash)
            if _canonical_json(combo) in completed:
                continue
            trace_config = {
                "enabled": bool(config["failure_trace"]["enabled"]),
                "only_found_failures": bool(config["failure_trace"]["only_found_failures"]),
                "max_traces": int(config["failure_trace"]["max_traces_per_checkpoint_mode"]),
            }
            jobs = [
                {
                    "episode_index": index,
                    "scenario": scenario,
                    "config": config,
                    "checkpoint_info": info,
                    "architecture": metadata["architecture"],
                    "loss": metadata["loss"],
                    "gamma": state["gamma"],
                    "tau": state["tau"],
                    "reward": metadata["reward"],
                    "policy_snapshot": snapshot,
                    "failure_trace": trace_config,
                    "device": str(device),
                }
                for index, scenario in enumerate(scenarios)
            ]
            if any(_contains_tensor(job) for job in jobs):
                raise RuntimeError("PRRAC evaluation worker job contains torch.Tensor")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
                results = list(executor.map(_evaluate_episode_job, jobs))
            rows = [dict(result["episode"]) for result in results]
            episode_rows.extend(rows)
            summary_rows.append(aggregate_checkpoint(rows, info))
            accepted_trace_count = 0
            trace_limit = int(
                config["failure_trace"]["max_traces_per_checkpoint_mode"]
            )
            for result in results:
                if result["trace_index"] is not None and accepted_trace_count < trace_limit:
                    trace_rows.extend(dict(row) for row in result["failure_trace"])
                    trace_index.append(dict(result["trace_index"]))
                    accepted_trace_count += 1
            progress["completed"].append(combo)
            completed.add(_canonical_json(combo))
            _write_json(output / "checkpoint_metadata.json", checkpoint_metadata)
            _write_outputs(
                output,
                checkpoint_paths=checkpoint_paths,
                scenarios=scenarios,
                modes=modes,
                episode_rows=episode_rows,
                summary_rows=summary_rows,
                trace_rows=trace_rows,
                trace_index=trace_index,
                progress=progress,
            )

    _write_json(output / "checkpoint_metadata.json", checkpoint_metadata)
    _write_outputs(
        output,
        checkpoint_paths=checkpoint_paths,
        scenarios=scenarios,
        modes=modes,
        episode_rows=episode_rows,
        summary_rows=summary_rows,
        trace_rows=trace_rows,
        trace_index=trace_index,
        progress=progress,
    )
    return _evaluation_summary(
        checkpoint_paths=checkpoint_paths,
        scenarios=scenarios,
        modes=modes,
        summary_rows=summary_rows,
        output=output,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-pattern")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--scenario-seed", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--modes", nargs="+")
    parser.add_argument("--resume-evaluation", action="store_true")
    parser.add_argument("--disable-failure-trace", action="store_true")
    args = parser.parse_args(argv)
    summary = run_evaluation(
        config_path=args.config,
        checkpoints=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_pattern=args.checkpoint_pattern,
        output_dir=args.output_dir,
        episodes_override=args.episodes,
        scenario_seed_override=args.scenario_seed,
        workers_override=args.workers,
        device_override=args.device,
        modes_override=args.modes,
        resume_evaluation=args.resume_evaluation,
        disable_failure_trace=args.disable_failure_trace,
    )
    print(json.dumps(_json_safe(summary), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
