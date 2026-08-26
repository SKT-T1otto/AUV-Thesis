"""Read-only deterministic execution diagnostics for Phase 1C checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import replace
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from chapter3_bser.controllers.state_provider import OnlinePlanningStateProvider
from chapter3_bser.experiments.phase1c_common import ExecutionEpisodeDiagnostics
from chapter3_bser.experiments.phase1c_bser_rmaddpg import METHOD
from chapter3_bser.integration.control_context import (
    AgentAssignmentContextV1,
    BSERControlContextV1,
    ExecutorAssignmentContextV1,
)
from chapter3_bser.integration.guided_env import GuidedEnv
from chapter3_bser.integration.rmaddpg_bridge import RMADDPGGuidanceBridge
from chapter3_bser.online.config import load_phase1b2_config
from chapter3_bser.online.controller import OnlineBSERController
from chapter3_bser.online.mission_context import OnlineMissionContext
from core.algorithms.maddpg import MADDPG
from core.config.ch3_config import build_ch3_config
from core.env.mission_env import MissionCoreEnv, environment_kwargs_from_config
from core.registry.experiment_registry import assert_registered_ch3_method
from core.scenarios.ch3_generator_impl import build_scenario_manifests


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "chapter3" / "bser_phase1c_diagnostic_eval.json"
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "chapter3" / "phase1c_bser_rmaddpg" / "diagnostics_v1"
)
SUPPORTED_SCHEMAS = {
    "bser.phase1c.training_state.v1",
    "bser.phase1c.training_state.v2",
}
SUPPORTED_MODES = (
    "full_residual",
    "executor_residual_off",
    "all_residual_off",
    "oracle_current_target_diagnostic",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
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


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema") != "bser.phase1c.diagnostic_eval.v1":
        raise ValueError("invalid Phase 1C diagnostic configuration schema")
    if config.get("method") != METHOD:
        raise ValueError("Phase 1C diagnostic method mismatch")
    if bool(config.get("explore", False)):
        raise ValueError("diagnostic evaluator requires explore=false")
    if bool(config.get("training_update", False)):
        raise ValueError("diagnostic evaluator requires training_update=false")
    for key, expected in (
        ("observation_dim", 28),
        ("action_dim", 3),
        ("critic_dim", 124),
    ):
        if int(config.get(key, expected)) != expected:
            raise ValueError(f"diagnostic {key} must remain {expected}")
    return config


def _validate_checkpoint_payload(payload: dict[str, Any]) -> dict[str, Any]:
    schema = str(payload.get("schema", ""))
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"unsupported Phase 1C checkpoint schema: {schema!r}")
    metadata = dict(payload.get("metadata", {}))
    method = metadata.get("method", METHOD)
    if method != METHOD:
        raise ValueError("checkpoint method mismatch")
    for key, expected in (
        ("observation_dim", 28),
        ("action_dim", 3),
        ("critic_dim", 124),
    ):
        actual = int(metadata.get(key, expected))
        if actual != expected:
            raise ValueError(
                f"checkpoint {key} mismatch: expected {expected}, got {actual}"
            )
    return payload


def _build_scenarios(config: dict[str, Any]):
    count = int(
        config.get(
            "evaluation_episodes",
            config.get("scenario_count", 200),
        )
    )
    seed = int(
        config.get(
            "scenario_seed",
            config.get("evaluation_seed", 1729),
        )
    )
    split = str(config.get("split", "validation"))
    profile = str(config["profile"])
    manifest = build_scenario_manifests(
        count=count,
        generator_seed=seed,
        split=split,
        profiles=(profile,),
    )[profile]
    return list(manifest["scenarios"]), manifest


def _env_config(config: dict[str, Any]) -> dict[str, Any]:
    return build_ch3_config(
        str(config.get("base_candidate", "ch3_v3_full_reference")),
        str(config["profile"]),
    )


def _make_env(config: dict[str, Any]) -> GuidedEnv:
    base_env = MissionCoreEnv(
        **environment_kwargs_from_config(
            _env_config(config),
            device="cpu",
            max_steps=int(config.get("max_steps", 400)),
            return_numpy=False,
        )
    )
    return GuidedEnv(base_env, enabled=True)


def _load_actor(path: Path, config: dict[str, Any]):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    _validate_checkpoint_payload(payload)
    env = _make_env(config)
    try:
        rl = config.get("rl", {})
        actor = MADDPG.init_from_env(
            env.unwrapped,
            gamma=float(rl.get("gamma", 0.95)),
            tau=float(rl.get("tau", 0.005)),
            lr_actor=float(rl.get("lr_actor", 0.001)),
            lr_critic=float(rl.get("lr_critic", 0.0005)),
            hidden_dim=int(rl.get("hidden_dim", 128)),
            residual_action_reg=float(rl.get("residual_action_reg", 0.01)),
        )
    finally:
        env.close()
    actor.load_training_state_dict(payload["maddpg_training_state"])
    actor.prep_rollouts(device="cpu")
    return actor, payload


def _public_context(env: GuidedEnv, state) -> OnlineMissionContext:
    return OnlineMissionContext.from_public_views(
        env.get_task_state(), env.get_search_execution_state(), state
    )


def _oracle_context(
    context: BSERControlContextV1,
    target: Iterable[float],
) -> BSERControlContextV1:
    """Privileged/oracle diagnostic only; never used in training."""

    vector = tuple(float(value) for value in target)
    if len(vector) != 3:
        raise ValueError("oracle diagnostic target must be 3D")
    executor_id = int(context.executor_assignment.executor_id)
    assignments: list[AgentAssignmentContextV1] = []
    for item in context.agent_assignments:
        if int(item.agent_id) != executor_id:
            assignments.append(item)
            continue
        assignments.append(
            replace(
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
        agent_assignments=tuple(assignments),
        executor_assignment=executor,
        decision_reason="PRIVILEGED_ORACLE_CURRENT_TARGET_DIAGNOSTIC_ONLY",
    )


def _apply_residual_mode(actions: torch.Tensor, mode: str) -> torch.Tensor:
    result = actions.clone()
    if mode == "full_residual" or mode == "oracle_current_target_diagnostic":
        return result
    if mode == "executor_residual_off":
        result[3] = 0.0
        return result
    if mode == "all_residual_off":
        result.zero_()
        return result
    raise ValueError(f"unsupported evaluation mode={mode!r}")


def _evaluate_episode(
    actor: MADDPG,
    scenario: dict[str, Any],
    *,
    episode_index: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    mode = str(config.get("_evaluation_mode", "full_residual"))
    env = _make_env(config)
    collector = ExecutionEpisodeDiagnostics()
    try:
        env.reset(scenario=scenario)
        provider = OnlinePlanningStateProvider(
            env,
            refresh_interval=int(
                load_phase1b2_config()["online"]["state_refresh_interval"]
            ),
        )
        state = provider.initialize()
        mission_context = _public_context(env, state)
        controller = OnlineBSERController(load_phase1b2_config())
        initial = controller.initialize(state, mission_context)
        bridge = RMADDPGGuidanceBridge()
        guidance = bridge.compile_guidance(
            initial.allocation,
            state,
            mission_context,
            decision_reason="INITIALIZE",
        )
        env.install_guidance(guidance)
        observations = env.refresh_observation_after_guidance()
        collector.reset(
            env,
            episode_id=episode_index,
            episode_index=episode_index,
            scenario_id=str(scenario.get("scenario_id", "")),
            scenario_seed=int(scenario["scenario_seed"]),
            max_steps=int(config.get("max_steps", 400)),
        )
        actor.prep_rollouts(device="cpu")
        reward_total = 0.0
        episode_length = 0

        for _ in range(int(config.get("max_steps", 400))):
            task_before = env.get_task_state()
            with torch.no_grad():
                parts = actor.step(observations, explore=False)
                actions = torch.stack(
                    [item.squeeze(0) for item in parts]
                ).to(env.unwrapped.device)
            actions = _apply_residual_mode(actions, mode)
            _, rewards, dones = env.step(actions)
            task_after = env.get_task_state()
            collector.observe_step(
                env,
                rewards,
                task_before=task_before,
                task_after=task_after,
            )
            reward_total += float(torch.as_tensor(rewards).sum().item())
            episode_length = int(getattr(task_after, "step", episode_length + 1))

            state = provider.snapshot(force=False)
            mission_context = _public_context(env, state)
            result = controller.step(state, mission_context)
            next_guidance = bridge.compile_guidance(
                result.allocation,
                state,
                mission_context,
                decision_reason=result.decision_reason,
            )
            if (
                mode == "oracle_current_target_diagnostic"
                and bool(mission_context.executor_knows_target)
                and not bool(mission_context.mission_complete)
            ):
                target = env.get_target_state().position
                next_guidance = _oracle_context(next_guidance, target)
            env.install_guidance(next_guidance)
            observations = env.refresh_observation_after_guidance()

            # Add controller-event counts without influencing policy state.
            names = [
                str(getattr(event, "value", event)).upper()
                for event in (getattr(result, "events", ()) or ())
            ]
            collector.executor_invalid_count += sum(
                "EXECUTOR_INVALID" in item for item in names
            )
            if bool(getattr(result, "replanned", False)):
                details = getattr(result, "diagnostics", None)
                affected = tuple(getattr(details, "affected_agent_ids", ()) or ())
                scope = str(getattr(details, "allocation_scope", ""))
                if 3 in affected or "executor" in scope:
                    collector.executor_replan_count += 1
            if all(bool(value) for value in dones):
                break

        row = collector.finalize(env)
        row["episode_length"] = int(episode_length)
        row["reward"] = float(reward_total)
        row["evaluation_mode"] = mode
        row["diagnostic_only"] = bool(mode == "oracle_current_target_diagnostic")
        row["privileged_oracle"] = bool(mode == "oracle_current_target_diagnostic")
        return row
    finally:
        env.close()


def _mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            values.append(numeric)
    return None if not values else float(statistics.fmean(values))


def _aggregate_checkpoint(
    rows: list[dict[str, Any]],
    checkpoint_info: dict[str, Any],
) -> dict[str, Any]:
    count = len(rows)
    found_count = sum(bool(row.get("found")) for row in rows)
    success_count = sum(bool(row.get("success")) for row in rows)
    result = dict(checkpoint_info)
    result["evaluation_episodes"] = count
    result["found_rate"] = 0.0 if count == 0 else found_count / count
    result["success_rate"] = 0.0 if count == 0 else success_count / count
    result["success_if_found_rate"] = (
        0.0 if found_count == 0 else success_count / found_count
    )
    result["success_if_found"] = result["success_if_found_rate"]
    result["mean_episode_length"] = _mean_numeric(rows, "episode_length")
    result["mean_found_step"] = _mean_numeric(rows, "found_step")
    result["mean_success_step"] = _mean_numeric(rows, "success_step")
    result["mean_executor_min_distance_to_target"] = _mean_numeric(
        rows, "executor_min_distance_to_target"
    )
    result["mean_capture_hold_counter_max"] = _mean_numeric(
        rows, "capture_hold_counter_max"
    )
    result["collision_episode_rate"] = (
        0.0
        if count == 0
        else sum(int(row.get("post_found_collision_count") or 0) > 0 for row in rows)
        / count
    )
    result["mean_reward"] = _mean_numeric(rows, "reward")
    return result


def _checkpoint_info(path: Path, payload: dict[str, Any], mode: str) -> dict[str, Any]:
    metadata = dict(payload.get("metadata", {}))
    schema = str(payload.get("schema"))
    implementation = metadata.get("implementation_version")
    if implementation is None:
        implementation = "v1" if schema.endswith("v1") else "v2"
    return {
        "checkpoint": str(Path(path).resolve()),
        "checkpoint_episode": int(payload.get("completed_episode", 0)),
        "completed_episode": int(payload.get("completed_episode", 0)),
        "schema": schema,
        "implementation_version": str(implementation),
        "evaluation_mode": mode,
        "diagnostic_only": bool(mode == "oracle_current_target_diagnostic"),
        "privileged_oracle": bool(mode == "oracle_current_target_diagnostic"),
    }


def _resolve_checkpoints(
    config: dict[str, Any],
    explicit: Iterable[Path] | None,
    checkpoint_dir: Path | None,
    checkpoint_pattern: str | None,
) -> list[Path]:
    values: list[Path] = []
    if explicit:
        values.extend(Path(item) for item in explicit)
    for item in config.get("checkpoints", ()):
        values.append(Path(item))
    for pattern in config.get("checkpoint_globs", ()):
        values.extend(sorted(ROOT.glob(str(pattern))))
    if checkpoint_dir is not None:
        pattern = checkpoint_pattern or "phase1c_episode_*.pt"
        values.extend(sorted(Path(checkpoint_dir).glob(pattern)))
    unique: list[Path] = []
    seen = set()
    for path in values:
        resolved = path if path.is_absolute() else ROOT / path
        resolved = resolved.resolve()
        if resolved in seen:
            continue
        if not resolved.is_file():
            raise FileNotFoundError(f"diagnostic checkpoint not found: {resolved}")
        unique.append(resolved)
        seen.add(resolved)
    if not unique:
        raise ValueError("no Phase 1C checkpoints were supplied for diagnostics")
    return unique


def _plot_outputs(summary_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]], output: Path) -> None:
    if not summary_rows:
        return
    labels = [
        f"{row.get('checkpoint_episode', 0)}:{row.get('evaluation_mode', '')}"
        for row in summary_rows
    ]
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 0.7), 4.8))
    axis.plot(range(len(labels)), [float(row["success_if_found_rate"]) for row in summary_rows], marker="o")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set(ylabel="Success if found", ylim=(0, 1), title="Phase 1C execution success conditional on discovery")
    figure.tight_layout(); figure.savefig(output / "success_if_found_curve.png", dpi=180); plt.close(figure)

    means = [row.get("mean_executor_min_distance_to_target") for row in summary_rows]
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 0.7), 4.8))
    axis.plot(range(len(labels)), [np.nan if value is None else float(value) for value in means], marker="o")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set(ylabel="Distance", title="Executor minimum distance to current target")
    figure.tight_layout(); figure.savefig(output / "executor_distance_curve.png", dpi=180); plt.close(figure)

    contact = [_mean_numeric([row for row in episode_rows if row.get("checkpoint") == summary.get("checkpoint") and row.get("evaluation_mode") == summary.get("evaluation_mode")], "capture_contact_step_count") for summary in summary_rows]
    hold = [_mean_numeric([row for row in episode_rows if row.get("checkpoint") == summary.get("checkpoint") and row.get("evaluation_mode") == summary.get("evaluation_mode")], "capture_full_hold_step_count") for summary in summary_rows]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 0.7), 4.8))
    axis.plot(x, [0.0 if value is None else value for value in contact], marker="o", label="contact steps")
    axis.plot(x, [0.0 if value is None else value for value in hold], marker="o", label="full-hold steps")
    axis.set_xticks(x, labels, rotation=45, ha="right")
    axis.set(ylabel="Mean steps", title="Capture diagnostics")
    axis.legend(); figure.tight_layout(); figure.savefig(output / "capture_diagnostics.png", dpi=180); plt.close(figure)


def run_evaluation(
    *,
    config_path: Path = DEFAULT_CONFIG,
    checkpoints: Iterable[Path] | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_pattern: str | None = None,
    output_dir: Path | None = None,
    episodes_override: int | None = None,
    workers_override: int | None = None,
    device_override: str | None = None,
    modes_override: Iterable[str] | None = None,
) -> dict[str, Any]:
    del workers_override  # deterministic evaluator is intentionally single-process
    assert_registered_ch3_method(METHOD)
    config = _load_config(config_path)
    config = dict(config)
    if episodes_override is not None:
        config["evaluation_episodes"] = int(episodes_override)
        config["scenario_count"] = int(episodes_override)
    if device_override is not None:
        config["device"] = str(device_override)
    modes = tuple(modes_override or config.get("modes", ("full_residual",)))
    if not modes:
        modes = ("full_residual",)
    invalid = [mode for mode in modes if mode not in SUPPORTED_MODES]
    if invalid:
        raise ValueError(f"unsupported diagnostic modes: {invalid}")

    output_value = config.get("output_dir")
    if output_dir is not None:
        output = Path(output_dir)
    elif output_value:
        output = Path(output_value)
        if not output.is_absolute():
            output = ROOT / output
    else:
        output = DEFAULT_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    for name in ("episode_execution_diagnostics.csv", "checkpoint_execution_summary.csv", "diagnostic_eval_summary.json"):
        if (output / name).exists():
            raise FileExistsError(
                f"diagnostic output already exists; choose a new output directory: {output / name}"
            )

    checkpoint_paths = _resolve_checkpoints(
        config, checkpoints, checkpoint_dir, checkpoint_pattern
    )
    scenarios, manifest = _build_scenarios(config)
    _write_json(output / "resolved_diagnostic_eval_config.json", config)
    _write_json(output / "evaluation_manifest.json", manifest)

    episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for checkpoint in checkpoint_paths:
        actor, payload = _load_actor(checkpoint, config)
        _validate_checkpoint_payload(payload)
        for mode in modes:
            mode_config = dict(config)
            mode_config["_evaluation_mode"] = mode
            rows = []
            for episode_index, scenario in enumerate(scenarios):
                row = _evaluate_episode(
                    actor,
                    scenario,
                    episode_index=episode_index,
                    config=mode_config,
                )
                row["checkpoint"] = str(checkpoint.resolve())
                row["checkpoint_episode"] = int(payload.get("completed_episode", 0))
                rows.append(row)
            episode_rows.extend(rows)
            summary_rows.append(
                _aggregate_checkpoint(rows, _checkpoint_info(checkpoint, payload, mode))
            )

    _write_csv(output / "episode_execution_diagnostics.csv", episode_rows)
    _write_csv(output / "checkpoint_execution_summary.csv", summary_rows)
    # Compatibility with the thesis-task naming requested for checkpoint tables.
    _write_csv(output / "checkpoint_summary.csv", summary_rows)
    comparison = {
        "schema": "bser.phase1c.diagnostic_eval.comparison.v1",
        "method": METHOD,
        "training_update": False,
        "explore": False,
        "same_scenarios_for_all_checkpoints": True,
        "oracle_mode_is_privileged_diagnostic_only": True,
        "checkpoint_count": len(checkpoint_paths),
        "scenario_count": len(scenarios),
        "modes": list(modes),
        "summary": summary_rows,
    }
    _write_json(output / "checkpoint_comparison.json", comparison)
    _plot_outputs(summary_rows, episode_rows, output)
    final = {
        "schema": "bser.phase1c.diagnostic_eval.summary.v1",
        "method": METHOD,
        "passed": bool(summary_rows),
        "training_update": False,
        "explore": False,
        "checkpoint_count": len(checkpoint_paths),
        "evaluation_episode_count": len(episode_rows),
        "output_dir": str(output.resolve()),
        "oracle_mode_is_privileged_diagnostic_only": True,
    }
    _write_json(output / "diagnostic_eval_summary.json", final)
    return final


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-pattern")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--modes", nargs="+")
    args = parser.parse_args(argv)
    summary = run_evaluation(
        config_path=args.config,
        checkpoints=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_pattern=args.checkpoint_pattern,
        output_dir=args.output_dir,
        episodes_override=args.episodes,
        workers_override=args.workers,
        device_override=args.device,
        modes_override=args.modes,
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
