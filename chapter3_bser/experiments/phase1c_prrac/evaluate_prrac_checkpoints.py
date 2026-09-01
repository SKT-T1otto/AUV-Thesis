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
    EvaluationTransitionDiagnostics,
    aggregate_checkpoint,
    failure_stage,
    paired_checkpoint_comparison,
    recommend_checkpoint,
    router_class_metrics,
)
from chapter3_bser.experiments.phase1c_prrac.evaluation_provenance import (
    EVALUATION_REPORT_SCHEMA,
    EVALUATION_SUMMARY_SCHEMA,
    EXECUTION_SUMMARY_SCHEMA,
    PROGRESS_SCHEMA,
    SEARCH_SUMMARY_SCHEMA,
    derive_unique_provenance,
    validate_evaluation_provenance,
    validate_resume_config,
    validate_summary_provenance,
)
from chapter3_bser.experiments.phase1c_prrac.evaluation_trace import (
    FailureTraceRecorder,
    failure_trace_row,
    vector3,
)
from chapter3_bser.experiments.phase1c_prrac.execution_continuity import (
    CHECKPOINT_RUNTIME_REVISION,
    EXECUTION_ABLATION_SCHEMA,
    OVERLAY_RUNTIME_REVISION,
    VARIANT_ORDER,
    ExecutionContinuityActionAdapter,
    ExecutionContinuityController,
    ExecutionContinuityDiagnostics,
    ExecutionVariant,
    aggregate_execution_variant,
    overlay_config,
    overlay_enabled,
    paired_execution_variant_comparisons,
    parse_execution_variant,
)
from chapter3_bser.experiments.phase1c_prrac.runtime_factory import (
    CONTROLLER_FACTORY_VERSION,
    NATIVE_B1_RUNTIME_REVISION,
    build_prrac_online_controller,
)
from chapter3_bser.experiments.phase1c_prrac.search_continuity import (
    SEARCH_CONTINUITY_SCHEMA,
    SearchContinuityDiagnostics,
    aggregate_search_continuity,
    apply_residual_mode,
    paired_searcher_residual_comparisons,
    search_continuity_config,
    search_continuity_config_hash,
    search_failure_funnel,
)
from chapter3_bser.experiments.phase1c_prrac.search_collision_recovery import (
    SEARCH_COLLISION_RECOVERY_SCHEMA,
    SEARCH_COLLISION_RECOVERY_SCHEMA_V2,
    SEARCH_RECOVERY_VARIANT_ORDER,
    ACTIVATION_DIAGNOSTICS_SCHEMA,
    SearchRecoveryVariant,
    SearchRecoveryVariantV2,
    aggregate_search_collision_recovery,
    apply_search_recovery_guidance,
    baseline_recovery_summary,
    build_search_recovery_controller,
    paired_search_collision_recovery_baseline_strata,
    paired_search_collision_recovery_comparisons,
    parse_search_recovery_variant,
    search_collision_recovery_config,
    search_collision_recovery_config_hash,
    search_collision_recovery_failure_funnel,
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
from chapter3_bser.online.mission_context import OnlineMissionContext
from core.config.ch3_config import build_ch3_config
from core.env.mission_env import MissionCoreEnv, environment_kwargs_from_config
from core.registry.experiment_registry import assert_registered_ch3_method
from core.scenarios.ch3_generator_impl import build_scenario_manifests


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "chapter3" / "bser_phase1c_prrac_eval.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "chapter3" / "phase1c_prrac" / "evaluation_v1"
EVALUATION_SCHEMA = EVALUATION_REPORT_SCHEMA
LEGACY_EVALUATION_SCHEMA = "bser.phase1c.prrac.evaluation.v1"
SUMMARY_SCHEMA = EVALUATION_SUMMARY_SCHEMA
EXECUTION_VARIANT_SUMMARY_SCHEMA = EXECUTION_SUMMARY_SCHEMA
OLD_CHECKPOINT_MESSAGE = (
    "Phase 1C-v1/v2 checkpoints are incompatible with PRRAC deterministic evaluation."
)
SUPPORTED_MODES = (
    "full_prrac",
    "searcher_residual_off",
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
    "execution_variant_episode.csv",
    "execution_variant_summary.csv",
    "paired_execution_variant_comparison.csv",
    "execution_variant_failure_funnel.csv",
    "execution_variant_summary.json",
    "search_continuity_episode.csv",
    "search_continuity_summary.csv",
    "paired_searcher_residual_comparison.csv",
    "search_failure_funnel.csv",
    "search_continuity_summary.json",
    "search_collision_recovery_episode.csv",
    "search_collision_recovery_summary.csv",
    "paired_search_collision_recovery_comparison.csv",
    "paired_search_collision_recovery_baseline_strata.csv",
    "search_collision_recovery_failure_funnel.csv",
    "search_collision_recovery_planning_failures.csv",
    "search_collision_recovery_activation_summary.csv",
    "search_collision_recovery_summary.json",
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
        "method": METHOD,
        "implementation_version": IMPLEMENTATION_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "observation_dim": 28,
        "action_dim": 3,
        "critic_dim": 124,
    }
    if config.get("schema") not in {
        EVALUATION_SCHEMA, LEGACY_EVALUATION_SCHEMA, EXECUTION_ABLATION_SCHEMA
    }:
        raise ValueError(f"invalid PRRAC evaluation config schema: {config.get('schema')!r}")
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"invalid PRRAC evaluation config {key}: {config.get(key)!r}")
    if config.get("explore") is not False:
        raise ValueError("PRRAC deterministic evaluation requires explore=false")
    if config.get("training_update") is not False:
        raise ValueError("PRRAC deterministic evaluation requires training_update=false")
    checkpoint_revision = str(
        config.get(
            "checkpoint_runtime_revision",
            config.get("execution_runtime_revision", ""),
        )
    )
    if checkpoint_revision not in {
        CHECKPOINT_RUNTIME_REVISION,
        NATIVE_B1_RUNTIME_REVISION,
    }:
        raise ValueError(
            f"unsupported checkpoint runtime revision: {checkpoint_revision!r}"
        )
    configured_variants = tuple(
        parse_execution_variant(value)
        for value in config.get(
            "execution_variants", (ExecutionVariant.B0_LEGACY_V2_1.value,)
        )
    )
    if checkpoint_revision == NATIVE_B1_RUNTIME_REVISION:
        if configured_variants != (ExecutionVariant.B1_ATOMIC_LAST_VALID,):
            raise ValueError("native checkpoint evaluation requires exactly B1_ATOMIC_LAST_VALID")
        expected_integration = "native"
        expected_evaluation_revision = NATIVE_B1_RUNTIME_REVISION
    else:
        expected_integration = (
            "overlay" if any(overlay_enabled(value) for value in configured_variants) else "legacy"
        )
        expected_evaluation_revision = (
            OVERLAY_RUNTIME_REVISION
            if any(overlay_enabled(value) for value in configured_variants)
            else CHECKPOINT_RUNTIME_REVISION
        )
    integration = str(config.get("runtime_integration_mode", expected_integration))
    if integration != expected_integration:
        raise ValueError(
            f"checkpoint runtime integration mismatch: {integration!r}"
        )
    evaluation_revision = str(
        config.get("evaluation_runtime_revision", expected_evaluation_revision)
    )
    if evaluation_revision != expected_evaluation_revision:
        raise ValueError(
            f"unregistered evaluation runtime overlay mismatch: {evaluation_revision!r}"
        )
    config["checkpoint_runtime_revision"] = checkpoint_revision
    config["evaluation_runtime_revision"] = expected_evaluation_revision
    config["runtime_integration_mode"] = integration
    if str(config.get("controller_factory_version", CONTROLLER_FACTORY_VERSION)) != CONTROLLER_FACTORY_VERSION:
        raise ValueError("unsupported controller factory version")
    config["controller_factory_version"] = CONTROLLER_FACTORY_VERSION
    config["schema"] = EVALUATION_SCHEMA
    config["execution_runtime"] = execution_runtime_config(config)
    config["execution_continuity"] = overlay_config(config)
    config["search_continuity_diagnostics"] = search_continuity_config(config)
    config["search_continuity_diagnostics_hash"] = search_continuity_config_hash(config)
    config["search_collision_recovery"] = search_collision_recovery_config(config)
    config["search_collision_recovery_config_hash"] = (
        search_collision_recovery_config_hash(config["search_collision_recovery"])
    )
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
    if config is not None:
        expected_checkpoint_revision = str(
            config.get("checkpoint_runtime_revision", config["execution_runtime_revision"])
        )
        actual_revision = str(metadata.get("execution_runtime_revision", ""))
        if actual_revision != expected_checkpoint_revision:
            raise ValueError("checkpoint execution runtime revision mismatch")
        if expected_checkpoint_revision == NATIVE_B1_RUNTIME_REVISION:
            if str(metadata.get("execution_variant", "")) != ExecutionVariant.B1_ATOMIC_LAST_VALID.value:
                raise ValueError("native checkpoint execution variant mismatch")
            if str(metadata.get("runtime_integration_mode", "")) != "native":
                raise ValueError("native checkpoint runtime integration mismatch")
            if str(metadata.get("controller_factory_version", "")) != CONTROLLER_FACTORY_VERSION:
                raise ValueError("native checkpoint controller factory mismatch")
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


def _checkpoint_info(
    path: Path,
    payload: Mapping[str, Any],
    mode: str,
    execution_variant: str | ExecutionVariant = ExecutionVariant.B0_LEGACY_V2_1,
    *,
    manifest_sha256: str = "",
    execution_overlay_config_hash: str = "",
    evaluation_runtime_revision: str | None = None,
    runtime_integration_mode: str | None = None,
    search_diagnostics_hash: str = "",
    search_recovery_variant: str | SearchRecoveryVariant = SearchRecoveryVariant.S2A_C0_BASELINE,
    search_recovery_config_hash: str = "",
    search_recovery_schema: str = SEARCH_COLLISION_RECOVERY_SCHEMA,
) -> dict[str, Any]:
    metadata = dict(payload["metadata"])
    oracle = mode == "oracle_current_target_diagnostic"
    diagnostic_only = oracle or mode == "searcher_residual_off"
    variant = parse_execution_variant(execution_variant)
    enabled = overlay_enabled(variant)
    checkpoint_revision = str(
        metadata.get("execution_runtime_revision", CHECKPOINT_RUNTIME_REVISION)
    )
    integration = str(
        runtime_integration_mode
        or ("legacy" if not enabled else "native" if checkpoint_revision == NATIVE_B1_RUNTIME_REVISION else "overlay")
    )
    return {
        "checkpoint": str(Path(path).resolve()),
        "checkpoint_episode": int(payload.get("completed_episode", metadata.get("completed_episode", 0))),
        "checkpoint_config_hash": str(metadata.get("config_hash", "")),
        "checkpoint_schema": str(payload["schema"]),
        "evaluation_mode": str(mode),
        "diagnostic_only": diagnostic_only,
        "privileged_oracle": oracle,
        "execution_variant": variant.value,
        "checkpoint_runtime_revision": checkpoint_revision,
        "evaluation_runtime_revision": str(
            evaluation_runtime_revision
            or (OVERLAY_RUNTIME_REVISION if enabled else CHECKPOINT_RUNTIME_REVISION)
        ),
        "runtime_integration_mode": integration,
        "runtime_overlay_enabled": integration == "overlay",
        "manifest_sha256": str(manifest_sha256),
        "execution_overlay_config_hash": str(execution_overlay_config_hash),
        "search_continuity_diagnostics_schema": SEARCH_CONTINUITY_SCHEMA,
        "search_continuity_diagnostics_hash": str(search_diagnostics_hash),
        "search_recovery_variant": parse_search_recovery_variant(search_recovery_variant).value,
        "search_collision_recovery_schema": str(search_recovery_schema),
        "activation_diagnostics_schema": (
            ACTIVATION_DIAGNOSTICS_SCHEMA
            if str(search_recovery_schema) == SEARCH_COLLISION_RECOVERY_SCHEMA_V2
            else ""
        ),
        "search_collision_recovery_config_hash": str(search_recovery_config_hash),
        "report_schema": EVALUATION_SCHEMA,
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
    selected_ids = tuple(str(value) for value in config.get("scenario_ids", ()))
    if selected_ids:
        by_id = {str(value.get("scenario_id", "")): value for value in scenarios}
        missing = [value for value in selected_ids if value not in by_id]
        if missing:
            raise ValueError(f"scenario-id-file contains IDs outside the generated manifest: {missing[:3]}")
        scenarios = [copy.deepcopy(by_id[value]) for value in selected_ids]
    body = {
        "schema": "bser.phase1c.prrac.evaluation_manifest.v1",
        "profile": profile,
        "split": str(config["split"]),
        "scenario_seed": int(config["scenario_seed"]),
        "evaluation_episodes": len(scenarios),
        "scenarios": scenarios,
        "source_manifest": generated,
        "diagnostic_only": bool(selected_ids),
        "scenario_selection_mode": (
            "baseline_collision_targeted_smoke" if selected_ids else "generated_manifest_order"
        ),
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


def _build_episode_controller(
    phase1b_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
    *,
    execution_variant: str | ExecutionVariant,
    runtime_integration_mode: str,
    checkpoint_runtime_revision: str,
):
    """Thin evaluation entry point into the shared PRRAC controller factory."""

    return build_prrac_online_controller(
        phase1b_config,
        experiment_config,
        execution_variant=execution_variant,
        runtime_integration_mode=runtime_integration_mode,
        checkpoint_runtime_revision=checkpoint_runtime_revision,
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


def _apply_residual_mode(
    actions: torch.Tensor,
    mode: str,
    stage_before: int | PRRACStage = PRRACStage.SEARCH,
) -> torch.Tensor:
    return apply_residual_mode(actions, mode, stage_before)


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
    actor_outputs: Any,
    raw_actions: torch.Tensor,
    applied_actions: torch.Tensor,
    recovery_snapshot: Any | None = None,
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
    actor_output = actor_outputs[3]
    probabilities = actor_output.router_probabilities.detach().cpu().reshape(-1, 3)[0]
    empty = {agent_id: None for agent_id in range(3)}
    modes = empty if recovery_snapshot is None else recovery_snapshot.mode_by_agent
    return failure_trace_row(
        checkpoint_episode=int(info["checkpoint_episode"]),
        execution_variant=str(info.get("execution_variant", "")),
        search_recovery_variant=str(info.get("search_recovery_variant", "")),
        checkpoint_runtime_revision=str(info.get("checkpoint_runtime_revision", "")),
        evaluation_runtime_revision=str(info.get("evaluation_runtime_revision", "")),
        runtime_integration_mode=str(info.get("runtime_integration_mode", "")),
        search_collision_recovery_schema=str(info.get("search_collision_recovery_schema", "")),
        search_collision_recovery_config_hash=str(info.get("search_collision_recovery_config_hash", "")),
        scenario_id=str(scenario.get("scenario_id", "")),
        scenario_seed=int(scenario["scenario_seed"]),
        step=int(step),
        mission_phase=str(public_guidance.mission_phase),
        stage_before=int(metadata.stage_before),
        stage_after=int(metadata.stage_after),
        task_found=bool(task.target_found),
        executor_knows_target=bool(task.executor_knows_target),
        mission_complete=bool(task.mission_complete),
        event_names=list(
            dict.fromkeys(
                (
                    *_event_names(result),
                    *tuple(
                        getattr(
                            getattr(controller, "last_detection", None),
                            "events",
                            (),
                        )
                    ),
                )
            )
        ),
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
        residual_action_norm=float(torch.linalg.vector_norm(applied_actions[3]).item()),
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
        recovery_active_agent_ids=[] if recovery_snapshot is None else recovery_snapshot.active_agent_ids,
        recovery_mode_by_agent=modes,
        recovery_attempt_id_by_agent=empty if recovery_snapshot is None else recovery_snapshot.attempt_id_by_agent,
        collision_streak_by_agent=empty if recovery_snapshot is None else recovery_snapshot.collision_streak_by_agent,
        semantic_search_candidate_id_by_agent=empty if recovery_snapshot is None else recovery_snapshot.semantic_candidate_id_by_agent,
        semantic_search_waypoint_by_agent=empty if recovery_snapshot is None else recovery_snapshot.semantic_waypoint_by_agent,
        recovery_navigation_endpoint_by_agent=empty if recovery_snapshot is None else recovery_snapshot.navigation_endpoint_by_agent,
        recovery_endpoint_cell_index_by_agent=empty if recovery_snapshot is None else recovery_snapshot.endpoint_cell_index_by_agent,
        recovery_trigger_reason_by_agent=empty if recovery_snapshot is None else recovery_snapshot.trigger_reason_by_agent,
        recovery_failure_reason_by_agent=empty if recovery_snapshot is None else recovery_snapshot.failure_reason_by_agent,
        recovery_route_refresh_attempted_by_agent=empty if recovery_snapshot is None else recovery_snapshot.route_refresh_attempted_by_agent,
        recovery_egress_attempted_by_agent=empty if recovery_snapshot is None else recovery_snapshot.egress_attempted_by_agent,
        raw_searcher_action_by_agent={agent_id: raw_actions[agent_id] for agent_id in range(3)},
        applied_searcher_action_by_agent={agent_id: applied_actions[agent_id] for agent_id in range(3)},
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
    execution_variant = parse_execution_variant(
        info.get("execution_variant", ExecutionVariant.B0_LEGACY_V2_1.value)
    )
    search_recovery_variant = parse_search_recovery_variant(
        info.get("search_recovery_variant", SearchRecoveryVariant.S2A_C0_BASELINE.value)
    )
    checkpoint_revision = str(
        info.get("checkpoint_runtime_revision", CHECKPOINT_RUNTIME_REVISION)
    )
    effective_integration = (
        "native"
        if checkpoint_revision == NATIVE_B1_RUNTIME_REVISION
        else "overlay"
        if overlay_enabled(execution_variant)
        else "legacy"
    )
    info["runtime_integration_mode"] = effective_integration
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
        controller = _build_episode_controller(
            phase1b_config,
            config,
            execution_variant=execution_variant,
            runtime_integration_mode=effective_integration,
            checkpoint_runtime_revision=checkpoint_revision,
        )
        initialized = controller.initialize(state, context)
        bridge = RMADDPGGuidanceBridge()
        guidance = bridge.compile_guidance(
            initialized.allocation, state, context, decision_reason="INITIALIZE"
        )
        env.install_guidance(guidance)
        installed_guidance = guidance
        observations = env.refresh_observation_after_guidance()
        diagnostics = PRRACDiagnostics()
        search_diagnostics = SearchContinuityDiagnostics()
        search_diagnostics.begin_episode(state)
        transition_diagnostics = EvaluationTransitionDiagnostics()
        recovery_controller = build_search_recovery_controller(search_recovery_variant)
        current_stage = PRRACStage.SEARCH
        reward_total = 0.0
        episode_length = 0
        collision_episode = False
        continuity_diagnostics = (
            controller.diagnostics
            if isinstance(controller, ExecutionContinuityController)
            else ExecutionContinuityDiagnostics(execution_variant)
        )
        continuity_action_adapter = ExecutionContinuityActionAdapter()

        for _ in range(int(config["max_steps"])):
            state_before = state
            transition_guidance = installed_guidance
            with torch.no_grad():
                outputs = _policy_outputs(actor, observations, device)
                for output in outputs:
                    diagnostics.observe_actor(output, [int(current_stage)])
                residual_actions = torch.stack(
                    [output.gated_residual_action.squeeze(0) for output in outputs]
                )
                mode_actions = _apply_residual_mode(
                    residual_actions, mode, current_stage
                )
                actions, suppression = continuity_action_adapter.apply(
                    mode_actions,
                    plan=getattr(controller, "current_plan", None),
                    variant=execution_variant,
                    mission_phase=str(guidance.mission_phase),
                    executor_id=3,
                )
                actions = actions.to(env.unwrapped.device)
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
            transition_diagnostics.observe(
                step=episode_length,
                stage_before=int(metadata.stage_before),
                stage_after=int(metadata.stage_after),
                contact_step_count=int(getattr(env.unwrapped, "capture_contact_step_count", 0)),
                full_hold_step_count=int(getattr(env.unwrapped, "capture_full_hold_step_count", 0)),
                mission_complete=bool(task.mission_complete),
            )
            collision_episode = bool(
                collision_episode
                or torch.as_tensor(env.unwrapped._collision_flags).any().item()
            )

            state = provider.snapshot(force=False)
            if recovery_controller is not None:
                recovery_controller.observe_transition(
                    stage_before=metadata.stage_before,
                    planning_state_after=state,
                    collision_flags=env.unwrapped.collision_flags,
                    planning_state_before=state_before,
                    installed_guidance_before=transition_guidance,
                )
                if bool(getattr(recovery_controller, "force_refresh_requested", False)):
                    state = provider.snapshot(force=True)
                for agent_id in recovery_controller.rejoined_agent_ids:
                    bridge.path_tracker.reset(agent_id)
            search_diagnostics.observe_transition(
                stage_before=metadata.stage_before,
                stage_after=metadata.stage_after,
                installed_guidance=transition_guidance,
                planning_state_before=state_before,
                planning_state_after=state,
                collision_flags=env.unwrapped.collision_flags,
                raw_actions=residual_actions,
                applied_actions=actions,
                actor_outputs=outputs,
                residual_contribution_ratios=(
                    env.unwrapped.last_residual_contribution_ratio_search
                ),
            )
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
            if recovery_controller is not None:
                recovery_controller.prepare_next_guidance(state, public_guidance)
            recovery_snapshot = (
                None if recovery_controller is None else recovery_controller.snapshot()
            )
            next_public_guidance = apply_search_recovery_guidance(
                public_guidance, state, recovery_controller
            )
            if recovery_controller is not None and hasattr(recovery_controller, "observe_activation"):
                recovery_controller.observe_activation(public_guidance, next_public_guidance)
            detection = getattr(controller, "last_detection", None)
            legacy_detection = getattr(result, "event_detection", None)
            continuity_diagnostics.observe_step(
                post_found=bool(
                    state_before.target_found
                    and int(metadata.stage_before) != int(PRRACStage.SEARCH)
                ),
                plan=getattr(controller, "current_plan", None),
                detection=detection,
                suppression=suppression,
                legacy_route_active=bool(public_guidance.executor_assignment.reachable),
                legacy_invalid_reason=str(
                    getattr(legacy_detection, "executor_invalid_reason", "")
                ),
                executor_collision=bool(
                    len(env.unwrapped.collision_flags) > 3
                    and env.unwrapped.collision_flags[3]
                ),
                transition_step=episode_length,
            )
            use_oracle = bool(
                mode == "oracle_current_target_diagnostic"
                and context.executor_knows_target
                and not context.mission_complete
            )
            target = env.get_target_state().position if use_oracle else None
            next_observations, installed_guidance = _install_next_guidance(
                env, next_public_guidance, mode=mode, true_target=target
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
                    actor_outputs=outputs,
                    raw_actions=residual_actions,
                    applied_actions=actions,
                    recovery_snapshot=recovery_snapshot,
                )
            )
            guidance = public_guidance
            observations = next_observations
            if all(bool(value) for value in dones):
                break

        task = env.get_task_state()
        row = env.finalize_episode()
        row.update(info)
        row.update(diagnostics.summary())
        row.update(continuity_diagnostics.summary())
        row.update(transition_diagnostics.summary())
        row.update(
            baseline_recovery_summary()
            if recovery_controller is None
            else recovery_controller.summary()
        )
        row.update(
            search_diagnostics.summary(
                found=bool(task.target_found),
                max_steps=int(config["max_steps"]),
                searcher_residual_off_enabled=(mode == "searcher_residual_off"),
            )
        )
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
                "max_steps": int(config["max_steps"]),
                "explore": False,
                "training_update": False,
                "optimizer_update_count": 0,
                "replay_sample_count": 0,
                "parameter_update_count": 0,
                "execution_variant": execution_variant.value,
                "search_recovery_variant": search_recovery_variant.value,
                "search_collision_recovery_schema": str(info.get("search_collision_recovery_schema", SEARCH_COLLISION_RECOVERY_SCHEMA)),
                "search_collision_recovery_config_hash": str(
                    info.get("search_collision_recovery_config_hash", "")
                ),
                "search_recovery_enabled": bool(recovery_controller is not None),
                "checkpoint_runtime_revision": str(
                    info["checkpoint_runtime_revision"]
                ),
                "evaluation_runtime_revision": str(
                    info["evaluation_runtime_revision"]
                ),
                "runtime_overlay_enabled": bool(info["runtime_overlay_enabled"]),
                "runtime_integration_mode": str(info["runtime_integration_mode"]),
                "manifest_sha256": str(info.get("manifest_sha256", "")),
                "execution_overlay_config_hash": str(
                    info.get("execution_overlay_config_hash", "")
                ),
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
            execution_variant=execution_variant.value,
            search_recovery_variant=search_recovery_variant.value,
            checkpoint_runtime_revision=str(info["checkpoint_runtime_revision"]),
            evaluation_runtime_revision=str(info["evaluation_runtime_revision"]),
            runtime_integration_mode=str(info["runtime_integration_mode"]),
            search_collision_recovery_schema=str(info.get("search_collision_recovery_schema", SEARCH_COLLISION_RECOVERY_SCHEMA)),
            search_collision_recovery_config_hash=str(info.get("search_collision_recovery_config_hash", "")),
            recovery_triggered=int(row.get("search_recovery_entry_count") or 0) > 0,
            pre_found_collision=bool(row.get("searcher_collision_episode_pre_found")),
        )
        planning_failures = [] if recovery_controller is None or not hasattr(recovery_controller, "planning_failure_rows") else recovery_controller.planning_failure_rows()
        for planning_row in planning_failures:
            planning_row.update({
                "checkpoint": str(info["checkpoint"]), "scenario_id": str(row["scenario_id"]),
                "scenario_seed": int(row["scenario_seed"]), "variant": search_recovery_variant.value,
                "manifest_sha256": str(info.get("manifest_sha256", "")),
                "search_collision_recovery_schema": str(info.get("search_collision_recovery_schema", "")),
                "search_collision_recovery_config_hash": str(info.get("search_collision_recovery_config_hash", "")),
                "report_schema": str(info.get("report_schema", EVALUATION_SCHEMA)),
            })
            planning_row["step"] = int(planning_row.get("planning_state_step", 0))
            planning_row["forced_refresh"] = bool(planning_row.get("forced_public_refresh", False))
        payload = _json_safe(
            {"episode": row, "failure_trace": traces, "trace_index": trace_index,
             "recovery_planning_failures": planning_failures}
        )
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
        "checkpoint_runtime_revision": str(info["checkpoint_runtime_revision"]),
        "evaluation_runtime_revision": str(info.get("evaluation_runtime_revision", "")),
        "runtime_integration_mode": str(info["runtime_integration_mode"]),
        "evaluation_mode": str(info["evaluation_mode"]),
        "execution_variant": str(
            info.get("execution_variant", ExecutionVariant.B0_LEGACY_V2_1.value)
        ),
        "manifest_sha256": str(manifest_hash),
        "execution_overlay_config_hash": str(
            info.get("execution_overlay_config_hash", "")
        ),
        "search_continuity_diagnostics_schema": str(
            info.get("search_continuity_diagnostics_schema", "")
        ),
        "search_continuity_diagnostics_hash": str(
            info.get("search_continuity_diagnostics_hash", "")
        ),
        "search_recovery_variant": str(info.get("search_recovery_variant", "")),
        "search_collision_recovery_schema": str(info.get("search_collision_recovery_schema", "")),
        "search_collision_recovery_config_hash": str(info.get("search_collision_recovery_config_hash", "")),
        "activation_diagnostics_schema": str(info.get("activation_diagnostics_schema", "")),
        "report_schema": str(info.get("report_schema", EVALUATION_SCHEMA)),
    }


def _validate_resume_search_diagnostics(
    saved_config: Mapping[str, Any], expected_hash: str
) -> None:
    saved_search = dict(saved_config.get("search_continuity_diagnostics", {}))
    if (
        saved_search.get("schema") != SEARCH_CONTINUITY_SCHEMA
        or str(saved_config.get("search_continuity_diagnostics_hash", ""))
        != str(expected_hash)
    ):
        raise ValueError("resume search diagnostics schema/hash mismatch")


def _failure_funnel(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row["checkpoint"]),
                str(row["evaluation_mode"]),
                str(row.get("execution_variant", ExecutionVariant.B0_LEGACY_V2_1.value)),
            ),
            [],
        ).append(row)
    output = []
    stages = ("NOT_FOUND", "FOUND_NO_CONTACT", "CONTACT_NO_HOLD", "HOLD_NO_SUCCESS", "SUCCESS")
    for (checkpoint, mode, variant), values in sorted(grouped.items()):
        result = {
            "checkpoint": checkpoint,
            "evaluation_mode": mode,
            "execution_variant": variant,
            "evaluation_episodes": len(values),
        }
        for stage in stages:
            result[stage.lower()] = sum(row.get("failure_stage") == stage for row in values)
        output.append(result)
    return output


def _paired_rows(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in episode_rows:
        grouped.setdefault(
            (
                str(row["checkpoint"]),
                str(row["evaluation_mode"]),
                str(row.get("execution_variant", ExecutionVariant.B0_LEGACY_V2_1.value)),
            ),
            [],
        ).append(row)
    output = []
    mode_variants = sorted({(mode, variant) for _, mode, variant in grouped})
    for mode, variant in mode_variants:
        checkpoints = sorted(
            checkpoint
            for checkpoint, item_mode, item_variant in grouped
            if item_mode == mode and item_variant == variant
        )
        for base, candidate in itertools.combinations(checkpoints, 2):
            row = paired_checkpoint_comparison(
                    grouped[(base, mode, variant)],
                    grouped[(candidate, mode, variant)],
                    base_checkpoint=base,
                    candidate_checkpoint=candidate,
                    evaluation_mode=mode,
                )
            row["execution_variant"] = variant
            output.append(row)
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


def _plot_execution_variants(
    summary_rows: list[dict[str, Any]], output: Path
) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    order = [item.value for item in VARIANT_ORDER]
    short = [f"B{index}" for index in range(4)]
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in order}
    for row in summary_rows:
        name = str(row.get("execution_variant", ""))
        if name in grouped:
            grouped[name].append(row)

    def value(name: str, field: str) -> float:
        numbers = [
            float(row[field])
            for row in grouped[name]
            if row.get(field) not in {None, ""}
        ]
        return float(statistics.fmean(numbers)) if numbers else float("nan")

    def bars(name: str, title: str, fields: tuple[str, ...]) -> None:
        figure, axis = plt.subplots(figsize=(8.0, 4.8))
        x = np.arange(4, dtype=np.float64)
        width = 0.8 / max(1, len(fields))
        for index, field in enumerate(fields):
            offset = (index - (len(fields) - 1) / 2.0) * width
            axis.bar(
                x + offset,
                [value(variant, field) for variant in order],
                width=width,
                label=field,
            )
        axis.set_xticks(x, short)
        axis.set_title(title)
        if len(fields) > 1:
            axis.legend()
        figure.tight_layout()
        figure.savefig(plots / name, dpi=180)
        plt.close(figure)

    bars("execution_variant_success.png", "Execution variant success", ("success_rate",))
    bars("execution_variant_found_contact.png", "Found and contact", ("found_rate", "contact_rate"))
    bars("execution_variant_success_if_found.png", "Success if found", ("success_if_found_rate",))
    bars("execution_variant_route_active_rate.png", "Post-found active route", ("executor_route_active_rate_post_found",))
    bars("execution_variant_invalid_rate.png", "Post-found executor invalid", ("executor_invalid_rate_post_found",))
    bars("execution_variant_assignment_unreachable_rate.png", "Post-found assignment unreachable", ("assignment_unreachable_rate_post_found",))
    bars("execution_variant_executor_distance.png", "Executor distance if found", ("mean_executor_min_distance_if_found", "mean_executor_final_distance_if_found"))
    bars("execution_variant_collision_rate.png", "Collision rate", ("collision_episode_rate",))
    bars("execution_variant_proxy_last_valid_hold.png", "Proxy / last-valid / SAFE_HOLD", ("mean_proxy_plan_count_if_found", "mean_last_valid_plan_count_if_found", "mean_safe_hold_active_rate_if_found"))
    bars("execution_variant_residual_suppression.png", "Executor residual suppression", ("mean_executor_residual_suppressed_steps_if_found",))


def _plot_search_recovery(
    episode_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    strata_rows: list[dict[str, Any]],
    output: Path,
) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    present = {str(row.get("search_recovery_variant", "")) for row in summary_rows}
    order = [item.value for item in SEARCH_RECOVERY_VARIANT_ORDER if item.value in present]
    labels = [
        "C0" if "C0_" in value else "C1" if "C1_" in value else "C2"
        for value in order
    ]

    def summary_value(variant: str, field: str) -> float:
        values = [float(row[field]) for row in summary_rows if row.get("search_recovery_variant") == variant and row.get(field) not in {None, ""}]
        return float(statistics.fmean(values)) if values else float("nan")

    def bars(filename: str, title: str, fields: tuple[str, ...]) -> None:
        figure, axis = plt.subplots(figsize=(8, 4.8)); x = np.arange(len(order)); width = .8/max(1,len(fields))
        for index, field in enumerate(fields):
            axis.bar(x+(index-(len(fields)-1)/2)*width, [summary_value(variant, field) for variant in order], width=width, label=field)
        axis.set_xticks(x, labels); axis.set_title(title)
        if len(fields)>1: axis.legend()
        figure.tight_layout(); figure.savefig(plots/filename,dpi=180); plt.close(figure)

    bars("s2a_found_rate.png", "Found rate by recovery variant", ("found_rate",))
    bars("s2a_success_rate.png", "Success rate by recovery variant", ("success_rate",))
    bars("s2a_pre_found_collision_episode_rate.png", "Pre-found collision episode rate", ("pre_found_collision_episode_rate",))
    for filename, title, field in (
        ("s2a_collision_count_distribution.png", "Total collision count distribution", "searcher_collision_count_pre_found_total"),
        ("s2a_max_collision_streak_distribution.png", "Max collision streak distribution", "searcher_collision_max_streak_pre_found"),
        ("s2a_found_step_distribution.png", "Found-step distribution", "found_step"),
    ):
        figure, axis = plt.subplots(figsize=(8,4.8))
        for variant,label in zip(order,labels):
            values=[float(row[field]) for row in episode_rows if row.get("search_recovery_variant")==variant and row.get(field) is not None]
            if values: axis.hist(values,bins=min(20,max(1,len(set(values)))),alpha=.4,label=label)
        axis.set_title(title)
        if axis.get_legend_handles_labels()[0]: axis.legend()
        figure.tight_layout(); figure.savefig(plots/filename,dpi=180); plt.close(figure)
    bars("s2a_recovery_attempts_successes.png", "Recovery attempts and successes", ("route_refresh_attempt_count","route_refresh_success_count","egress_attempt_count","egress_success_count"))

    def strata_plot(filename: str, stratum: str, title: str) -> None:
        figure,axis=plt.subplots(figsize=(8,4.8)); selected=[row for row in strata_rows if row.get("stratum")==stratum]
        names=[]; base=[]; candidate=[]
        for row in selected:
            names.append("C1" if "C1_" in str(row.get("candidate_search_recovery_variant", "")) else "C2")
            base.append(float(row.get("baseline_found_rate") or 0)); candidate.append(float(row.get("candidate_found_rate") or 0))
        x=np.arange(len(names)); axis.bar(x-.2,base,.4,label="C0"); axis.bar(x+.2,candidate,.4,label="candidate"); axis.set_xticks(x,names); axis.set_title(title); axis.legend(); figure.tight_layout(); figure.savefig(plots/filename,dpi=180); plt.close(figure)
    strata_plot("s2a_baseline_collision_stratum.png","BASELINE_COLLISION","C0 baseline-collision stratum outcomes")
    strata_plot("s2a_baseline_no_collision_preservation.png","BASELINE_NO_COLLISION","C0 baseline-no-collision preservation")


def _execution_variant_summary(
    *,
    scenarios: list[dict[str, Any]],
    variants: tuple[ExecutionVariant, ...],
    summary_rows: list[dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    provenance = derive_unique_provenance(summary_rows)
    return {
        "schema": EXECUTION_VARIANT_SUMMARY_SCHEMA,
        **provenance,
        "execution_variants": provenance["execution_variant_values"],
        "scenario_count": len(scenarios),
        "same_scenarios_for_all_execution_variants": True,
        "explore": False,
        "training_update": False,
        "performance_passed": None,
        "output_dir": str(output.resolve()),
        "variant_summaries": summary_rows,
    }


def _evaluation_summary(
    *,
    checkpoint_paths: list[Path],
    scenarios: list[dict[str, Any]],
    modes: tuple[str, ...],
    execution_variants: tuple[ExecutionVariant, ...],
    search_recovery_variants: tuple[SearchRecoveryVariant, ...] = (
        SearchRecoveryVariant.S2A_C0_BASELINE,
    ),
    summary_rows: list[dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    provenance = derive_unique_provenance(summary_rows)
    return {
        "schema": SUMMARY_SCHEMA,
        **provenance,
        "method": METHOD,
        "implementation_version": IMPLEMENTATION_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_count": len(checkpoint_paths),
        "scenario_count": len(scenarios),
        "evaluation_modes": provenance["evaluation_mode_values"],
        "execution_variants": provenance["execution_variant_values"],
        "search_recovery_variants": provenance["search_recovery_variant_values"],
        "same_scenarios_for_all_checkpoints": True,
        "explore": False,
        "training_update": False,
        "optimizer_update_count": 0,
        "replay_sample_count": 0,
        "parameter_update_count": 0,
        "output_dir": str(output.resolve()),
        **recommend_checkpoint(
            [
                row
                for row in summary_rows
                if row.get("execution_variant", ExecutionVariant.B0_LEGACY_V2_1.value)
                == ExecutionVariant.B0_LEGACY_V2_1.value
            ]
        ),
    }


def _write_outputs(
    output: Path,
    *,
    checkpoint_paths: list[Path],
    scenarios: list[dict[str, Any]],
    modes: tuple[str, ...],
    execution_variants: tuple[ExecutionVariant, ...],
    search_recovery_variants: tuple[SearchRecoveryVariant, ...] = (
        SearchRecoveryVariant.S2A_C0_BASELINE,
    ),
    episode_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    trace_index: list[dict[str, Any]],
    progress: dict[str, Any],
    recovery_planning_failures: list[dict[str, Any]] | None = None,
) -> None:
    _write_csv(output / "episode_evaluation.csv", episode_rows)
    _write_csv(output / "checkpoint_summary.csv", summary_rows)
    _write_csv(output / "paired_checkpoint_comparison.csv", _paired_rows(episode_rows))
    _write_csv(output / "failure_funnel.csv", _failure_funnel(episode_rows))
    execution_summary_rows = [
        aggregate_execution_variant(
            values,
            {
                key: values[0].get(key)
                for key in (
                    "checkpoint",
                    "checkpoint_config_hash",
                    "checkpoint_episode",
                    "evaluation_mode",
                    "execution_variant",
                    "search_recovery_variant",
                    "checkpoint_runtime_revision",
                    "evaluation_runtime_revision",
                    "runtime_integration_mode",
                    "runtime_overlay_enabled",
                    "manifest_sha256",
                    "execution_overlay_config_hash",
                    "search_collision_recovery_config_hash",
                    "search_collision_recovery_schema",
                    "search_continuity_diagnostics_hash",
                )
            },
        )
        for _, values in sorted(
            {
                (
                    str(row["checkpoint"]),
                    str(row["evaluation_mode"]),
                    str(row["execution_variant"]),
                    str(row.get("search_recovery_variant", "")),
                    str(row["manifest_sha256"]),
                ): [
                    item
                    for item in episode_rows
                    if (
                        str(item["checkpoint"]),
                        str(item["evaluation_mode"]),
                        str(item["execution_variant"]),
                        str(item.get("search_recovery_variant", "")),
                        str(item["manifest_sha256"]),
                    )
                    == (
                        str(row["checkpoint"]),
                        str(row["evaluation_mode"]),
                        str(row["execution_variant"]),
                        str(row.get("search_recovery_variant", "")),
                        str(row["manifest_sha256"]),
                    )
                ]
                for row in episode_rows
            }.items()
        )
    ]
    validate_summary_provenance(episode_rows, execution_summary_rows, scenarios)
    _write_csv(output / "execution_variant_episode.csv", episode_rows)
    _write_csv(output / "execution_variant_summary.csv", execution_summary_rows)
    _write_csv(
        output / "paired_execution_variant_comparison.csv",
        paired_execution_variant_comparisons(episode_rows),
    )
    _write_csv(
        output / "execution_variant_failure_funnel.csv", _failure_funnel(episode_rows)
    )
    search_provenance = derive_unique_provenance(episode_rows)
    _write_json(
        output / "execution_variant_summary.json",
        _execution_variant_summary(
            scenarios=scenarios,
            variants=execution_variants,
            summary_rows=execution_summary_rows,
            output=output,
        ),
    )
    search_group_keys = (
        "checkpoint",
        "checkpoint_config_hash",
        "checkpoint_episode",
        "checkpoint_runtime_revision",
        "evaluation_runtime_revision",
        "runtime_integration_mode",
        "execution_variant",
        "evaluation_mode",
        "manifest_sha256",
        "search_continuity_diagnostics_schema",
        "search_continuity_diagnostics_hash",
        "search_recovery_variant",
        "search_collision_recovery_config_hash",
    )
    search_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in episode_rows:
        search_groups.setdefault(
            tuple(row.get(key) for key in search_group_keys), []
        ).append(row)
    search_summary_rows = [
        aggregate_search_continuity(values, dict(zip(search_group_keys, key)))
        for key, values in sorted(search_groups.items(), key=lambda item: str(item[0]))
    ]
    validate_summary_provenance(episode_rows, search_summary_rows, scenarios)
    paired_search_rows = paired_searcher_residual_comparisons(episode_rows)
    search_funnel_rows = search_failure_funnel(episode_rows)
    _write_csv(output / "search_continuity_episode.csv", episode_rows)
    _write_csv(output / "search_continuity_summary.csv", search_summary_rows)
    _write_csv(output / "paired_searcher_residual_comparison.csv", paired_search_rows)
    _write_csv(output / "search_failure_funnel.csv", search_funnel_rows)
    _write_json(
        output / "search_continuity_summary.json",
        {
            "schema": SEARCH_SUMMARY_SCHEMA,
            **search_provenance,
            "search_continuity_diagnostics_hashes": sorted(
                {
                    str(row.get("search_continuity_diagnostics_hash", ""))
                    for row in episode_rows
                }
            ),
            "scenario_count": len(scenarios),
            "evaluation_modes": search_provenance["evaluation_mode_values"],
            "execution_variants": search_provenance["execution_variant_values"],
            "search_recovery_variants": search_provenance["search_recovery_variant_values"],
            "summary": search_summary_rows,
            "paired_searcher_residual_comparison": paired_search_rows,
            "failure_funnel": search_funnel_rows,
        },
    )
    recovery_episode_rows = [
        {
            **row,
            "search_recovery_variant": row.get("search_recovery_variant")
            or SearchRecoveryVariant.S2A_C0_BASELINE.value,
            "search_collision_recovery_schema": row.get(
                "search_collision_recovery_schema", SEARCH_COLLISION_RECOVERY_SCHEMA
            ),
            "search_collision_recovery_config_hash": row.get(
                "search_collision_recovery_config_hash", ""
            ),
        }
        for row in episode_rows
    ]
    recovery_group_keys = (
        "checkpoint", "checkpoint_config_hash", "checkpoint_episode",
        "checkpoint_runtime_revision", "evaluation_runtime_revision",
        "runtime_integration_mode", "execution_variant", "evaluation_mode",
        "search_recovery_variant", "manifest_sha256",
        "search_continuity_diagnostics_hash", "search_collision_recovery_schema",
        "search_collision_recovery_config_hash", "activation_diagnostics_schema", "report_schema",
    )
    recovery_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in recovery_episode_rows:
        recovery_groups.setdefault(tuple(row.get(key) for key in recovery_group_keys), []).append(row)
    recovery_summary_rows = [
        aggregate_search_collision_recovery(values, dict(zip(recovery_group_keys, key)))
        for key, values in sorted(recovery_groups.items(), key=lambda item: str(item[0]))
    ]
    validate_summary_provenance(recovery_episode_rows, recovery_summary_rows, scenarios)
    paired_recovery_rows = paired_search_collision_recovery_comparisons(recovery_episode_rows)
    strata_rows = paired_search_collision_recovery_baseline_strata(recovery_episode_rows)
    recovery_funnel_rows = search_collision_recovery_failure_funnel(recovery_episode_rows)
    _write_csv(output / "search_collision_recovery_episode.csv", recovery_episode_rows)
    _write_csv(output / "search_collision_recovery_summary.csv", recovery_summary_rows)
    _write_csv(output / "paired_search_collision_recovery_comparison.csv", paired_recovery_rows)
    _write_csv(output / "paired_search_collision_recovery_baseline_strata.csv", strata_rows)
    _write_csv(output / "search_collision_recovery_failure_funnel.csv", recovery_funnel_rows)
    _write_csv(
        output / "search_collision_recovery_planning_failures.csv",
        list(recovery_planning_failures or ()),
    )
    activation_summary_rows = [
        {
            key: row.get(key)
            for key in (
                "checkpoint", "execution_variant", "evaluation_mode", "search_recovery_variant",
                "search_collision_recovery_schema", "search_collision_recovery_config_hash",
                "activation_diagnostics_schema", "report_schema",
                "manifest_sha256", "evaluation_episodes", "search_recovery_entry_count",
                "forced_public_refresh_count", "recovery_plan_active_step_count",
                "recovery_guidance_changed_step_count", "recovery_effective_intervention_count",
                "local_connector_attempt_count", "local_connector_plan_count",
                "tracking_waypoint_delta_norm_sum", "tracking_waypoint_delta_norm_mean",
                "tracking_waypoint_delta_norm_max", "path_changed_step_count",
            )
        }
        for row in recovery_summary_rows
    ]
    _write_csv(output / "search_collision_recovery_activation_summary.csv", activation_summary_rows)
    recovery_provenance = derive_unique_provenance(recovery_episode_rows)
    _write_json(
        output / "search_collision_recovery_summary.json",
        {
            "schema": SEARCH_SUMMARY_SCHEMA,
            **recovery_provenance,
            "scenario_count": len(scenarios),
            "search_recovery_variants": recovery_provenance["search_recovery_variant_values"],
            "summary": recovery_summary_rows,
            "paired_comparison": paired_recovery_rows,
            "baseline_strata": strata_rows,
            "failure_funnel": recovery_funnel_rows,
            "activation_summary": activation_summary_rows,
        },
    )
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
            execution_variants=execution_variants,
            search_recovery_variants=search_recovery_variants,
            summary_rows=summary_rows,
            output=output,
        ),
    )
    _plot(summary_rows, output)
    _plot_execution_variants(execution_summary_rows, output)
    _plot_search_recovery(recovery_episode_rows, recovery_summary_rows, strata_rows, output)


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
    execution_variants_override: Iterable[str] | None = None,
    search_recovery_variants_override: Iterable[str] | None = None,
    resume_evaluation: bool = False,
    disable_failure_trace: bool = False,
    scenario_id_file: Path | None = None,
    formal: bool = False,
) -> dict[str, Any]:
    assert_registered_ch3_method(METHOD)
    requested_checkpoints = tuple(checkpoints or ())
    config = copy.deepcopy(_load_config(config_path))
    if formal and scenario_id_file is not None:
        raise ValueError("--formal cannot be combined with --scenario-id-file")
    if formal and episodes_override not in {None, 100}:
        raise ValueError("--formal requires exactly 100 episodes")
    if formal:
        config["evaluation_episodes"] = 100
    if scenario_id_file is not None:
        scenario_path = Path(scenario_id_file)
        raw = scenario_path.read_text(encoding="utf-8")
        try:
            parsed = json.loads(raw)
            scenario_ids = parsed.get("scenario_ids", ()) if isinstance(parsed, Mapping) else parsed
        except json.JSONDecodeError:
            scenario_ids = [line.strip() for line in raw.splitlines() if line.strip()]
        scenario_ids = [str(value) for value in scenario_ids]
        if not scenario_ids or len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("scenario-id-file must contain a non-empty unique scenario ID list")
        config["scenario_ids"] = scenario_ids
        config["diagnostic_only"] = True
        config["scenario_selection_mode"] = "baseline_collision_targeted_smoke"
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
    execution_variants = tuple(
        parse_execution_variant(value)
        for value in (
            execution_variants_override
            or config.get(
                "execution_variants", (ExecutionVariant.B0_LEGACY_V2_1.value,)
            )
        )
    )
    if not execution_variants or len(set(execution_variants)) != len(execution_variants):
        raise ValueError("execution variants must be a non-empty unique registered list")
    checkpoint_runtime_revision = str(config["checkpoint_runtime_revision"])
    if checkpoint_runtime_revision == NATIVE_B1_RUNTIME_REVISION and execution_variants != (
        ExecutionVariant.B1_ATOMIC_LAST_VALID,
    ):
        raise ValueError("native checkpoint evaluation permits only B1_ATOMIC_LAST_VALID")
    config["execution_variants"] = [item.value for item in execution_variants]
    resolved_evaluation_runtime_revisions = sorted(
        {
            NATIVE_B1_RUNTIME_REVISION
            if checkpoint_runtime_revision == NATIVE_B1_RUNTIME_REVISION
            else OVERLAY_RUNTIME_REVISION
            if overlay_enabled(item)
            else CHECKPOINT_RUNTIME_REVISION
            for item in execution_variants
        }
    )
    resolved_runtime_integration_modes = sorted(
        {
            "native"
            if checkpoint_runtime_revision == NATIVE_B1_RUNTIME_REVISION
            else "overlay"
            if overlay_enabled(item)
            else "legacy"
            for item in execution_variants
        }
    )
    search_recovery_variants = tuple(
        parse_search_recovery_variant(value)
        for value in (
            search_recovery_variants_override
            or config.get(
                "search_recovery_variants",
                config["search_collision_recovery"].get(
                    "variants", (SearchRecoveryVariant.S2A_C0_BASELINE.value,)
                ),
            )
        )
    )
    if not search_recovery_variants or len(set(search_recovery_variants)) != len(search_recovery_variants):
        raise ValueError("search recovery variants must be a non-empty unique registered list")
    config["search_recovery_variants"] = [item.value for item in search_recovery_variants]
    config["search_collision_recovery"]["variants"] = list(config["search_recovery_variants"])
    baseline_variants = {SearchRecoveryVariant.S2A_C0_BASELINE, SearchRecoveryVariantV2.S2A1_C0_BASELINE}
    if any(item not in baseline_variants for item in search_recovery_variants):
        config["search_collision_recovery"]["enabled"] = True
    config["search_collision_recovery_config_hash"] = search_collision_recovery_config_hash(
        config["search_collision_recovery"]
    )
    execution_overlay_config_hash = _hash(config["execution_continuity"])
    search_diagnostics_hash = str(config["search_continuity_diagnostics_hash"])
    search_recovery_hash = str(config["search_collision_recovery_config_hash"])
    search_recovery_schema = str(config["search_collision_recovery"]["schema"])
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
        config, requested_checkpoints, checkpoint_dir, checkpoint_pattern
    )
    scenarios, manifest = _build_evaluation_manifest(config)
    manifest_hash = str(manifest["manifest_sha256"])
    config.update(
        {
            "requested_config_path": str(Path(config_path)),
            "resolved_config_path": str(Path(config_path).resolve()),
            "resolved_config_output_path": str((output / "resolved_evaluation_config.json").resolve()),
            "requested_checkpoint_arguments": [str(Path(value)) for value in requested_checkpoints],
            "resolved_checkpoint_paths": [str(path.resolve()) for path in checkpoint_paths],
            "resolved_output_dir": str(output.resolve()),
            "resolved_evaluation_modes": list(modes),
            "resolved_execution_variants": [item.value for item in execution_variants],
            "resolved_evaluation_runtime_revisions": resolved_evaluation_runtime_revisions,
            "resolved_runtime_integration_modes": resolved_runtime_integration_modes,
            "resolved_search_recovery_variants": [item.value for item in search_recovery_variants],
            "resolved_workers": workers,
            "resolved_evaluation_episodes": int(config["evaluation_episodes"]),
            "resolved_scenario_seed": int(config["scenario_seed"]),
            "resolved_device": str(device),
            "manifest_sha256": manifest_hash,
            "resolved_scenario_ids": [str(value["scenario_id"]) for value in scenarios],
            "checkpoints": [str(path.resolve()) for path in checkpoint_paths],
            "output_dir": str(output.resolve()),
            "execution_overlay_config_hash": execution_overlay_config_hash,
            "search_continuity_diagnostics_schema": SEARCH_CONTINUITY_SCHEMA,
            "search_collision_recovery_schema": search_recovery_schema,
        }
    )
    config["resolved_config_hash"] = _hash(
        {key: value for key, value in config.items() if key != "resolved_config_hash"}
    )
    if resume_evaluation:
        saved_config = json.loads(
            (output / "resolved_evaluation_config.json").read_text(encoding="utf-8")
        )
        _validate_resume_search_diagnostics(saved_config, search_diagnostics_hash)
        validate_resume_config(saved_config, config)
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
    recovery_planning_failures = (
        _read_csv(output / "search_collision_recovery_planning_failures.csv")
        if resume_evaluation else []
    )
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
            "schema": PROGRESS_SCHEMA,
            "manifest_sha256": manifest_hash,
            "resolved_config_hash": config["resolved_config_hash"],
            "search_collision_recovery_config_hash": search_recovery_hash,
            "report_schema": EVALUATION_SCHEMA,
            "completed": [],
        }
    )
    if progress.get("manifest_sha256") != manifest_hash:
        raise ValueError("resume evaluation progress manifest hash mismatch")
    if progress.get("schema") != PROGRESS_SCHEMA:
        raise ValueError("resume evaluation progress schema mismatch")
    if progress.get("resolved_config_hash") != config["resolved_config_hash"]:
        raise ValueError("resume evaluation progress resolved config hash mismatch")
    if progress.get("search_collision_recovery_config_hash") != search_recovery_hash:
        raise ValueError("resume evaluation progress search recovery hash mismatch")
    if progress.get("report_schema") != EVALUATION_SCHEMA:
        raise ValueError("resume evaluation progress report schema mismatch")
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
            for execution_variant in execution_variants:
                for search_recovery_variant in search_recovery_variants:
                    info = _checkpoint_info(
                        checkpoint,
                        payload,
                        mode,
                        execution_variant,
                        manifest_sha256=manifest_hash,
                        execution_overlay_config_hash=execution_overlay_config_hash,
                        evaluation_runtime_revision=(
                            NATIVE_B1_RUNTIME_REVISION
                            if checkpoint_runtime_revision == NATIVE_B1_RUNTIME_REVISION
                            else OVERLAY_RUNTIME_REVISION
                            if overlay_enabled(execution_variant)
                            else CHECKPOINT_RUNTIME_REVISION
                        ),
                        runtime_integration_mode=(
                            "native"
                            if checkpoint_runtime_revision == NATIVE_B1_RUNTIME_REVISION
                            else "overlay"
                            if overlay_enabled(execution_variant)
                            else "legacy"
                        ),
                        search_diagnostics_hash=search_diagnostics_hash,
                        search_recovery_variant=search_recovery_variant,
                        search_recovery_config_hash=search_recovery_hash,
                        search_recovery_schema=search_recovery_schema,
                    )
                    if bool(config.get("diagnostic_only", False)):
                        info["diagnostic_only"] = True
                    combo = _combo_key(info, manifest_hash)
                    if _canonical_json(combo) in completed:
                        continue
                    trace_config = {
                        "enabled": bool(config["failure_trace"]["enabled"]),
                        "only_found_failures": bool(config["failure_trace"]["only_found_failures"]),
                        "max_traces": int(config["failure_trace"]["max_traces_per_checkpoint_mode"]),
                        "selector": str(config["failure_trace"].get("selector", "found_failures")),
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
                    for result in results:
                        recovery_planning_failures.extend(
                            dict(value) for value in result.get("recovery_planning_failures", ())
                        )
                    episode_rows.extend(rows)
                    summary_rows.append(aggregate_checkpoint(rows, info))
                    accepted_trace_count = 0
                    trace_limit = int(config["failure_trace"]["max_traces_per_checkpoint_mode"])
                    for result in results:
                        if result["trace_index"] is not None and accepted_trace_count < trace_limit:
                            trace_rows.extend(dict(row) for row in result["failure_trace"])
                            trace_index.append(dict(result["trace_index"]))
                            accepted_trace_count += 1
                    progress["completed"].append(combo)
                    completed.add(_canonical_json(combo))
                    _write_json(output / "checkpoint_metadata.json", checkpoint_metadata)
                    _write_csv(output / "episode_evaluation.csv", episode_rows)
                    _write_csv(output / "failure_trace_index.csv", trace_index)
                    _write_csv(output / "search_collision_recovery_planning_failures.csv", recovery_planning_failures)
                    _atomic_text(output / "failure_trace.jsonl", "".join(_canonical_json(row) + "\n" for row in trace_rows))
                    _write_json(output / "evaluation_progress.json", progress)

    _write_json(output / "checkpoint_metadata.json", checkpoint_metadata)
    validate_evaluation_provenance(
        rows=episode_rows,
        resolved_config=config,
        progress=progress,
        checkpoint_metadata=checkpoint_metadata,
        summary_groups=summary_rows,
        expected_scenarios=scenarios,
    )
    _write_outputs(
        output,
        checkpoint_paths=checkpoint_paths,
        scenarios=scenarios,
        modes=modes,
        execution_variants=execution_variants,
        search_recovery_variants=search_recovery_variants,
        episode_rows=episode_rows,
        summary_rows=summary_rows,
        trace_rows=trace_rows,
        trace_index=trace_index,
        progress=progress,
        recovery_planning_failures=recovery_planning_failures,
    )
    return _evaluation_summary(
        checkpoint_paths=checkpoint_paths,
        scenarios=scenarios,
        modes=modes,
        execution_variants=execution_variants,
        search_recovery_variants=search_recovery_variants,
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
    parser.add_argument(
        "--execution-variants",
        nargs="+",
        choices=[item.value for item in VARIANT_ORDER],
    )
    parser.add_argument(
        "--search-recovery-variants",
        nargs="+",
        choices=[item.value for item in SEARCH_RECOVERY_VARIANT_ORDER],
    )
    parser.add_argument("--resume-evaluation", action="store_true")
    parser.add_argument("--disable-failure-trace", action="store_true")
    parser.add_argument("--scenario-id-file", type=Path)
    parser.add_argument("--formal", action="store_true")
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
        execution_variants_override=args.execution_variants,
        search_recovery_variants_override=args.search_recovery_variants,
        resume_evaluation=args.resume_evaluation,
        disable_failure_trace=args.disable_failure_trace,
        scenario_id_file=args.scenario_id_file,
        formal=args.formal,
    )
    print(json.dumps(_json_safe(summary), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
