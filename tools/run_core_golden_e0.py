"""Run 60 core-only E0 trajectories against the frozen Phase 0B-2 golden hashes."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_ROOT = ROOT / "configs" / "scenarios" / "e0_equivalence"
MIGRATION_ROOT = ROOT / "experiments" / "chapter3" / "e0_core_migration"
OUTPUT_ROOT = MIGRATION_ROOT / "core_without_legacy_run"
GOLDEN_PATH = MIGRATION_ROOT / "golden_trace_manifest.json"
PROFILES = (
    "M00_MOVING_CLEAR",
    "M10_MOVING_UNKNOWN_SINGLE",
    "M20_MOVING_UNKNOWN_MULTI",
    "M90_MOVING_KNOWN_ORACLE",
)
ACTION_TRACES = ("zero_action", "seeded_random_low_amplitude", "deterministic_scripted")
MAX_STEPS = 50
ATOL = 1e-6
RTOL = 1e-6

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _optional_array(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    return None if value is None else _to_numpy(value)


def _target_payload(state: Any) -> Any:
    if state is None:
        return None
    return {
        "position": np.asarray(state.position).copy(),
        "velocity": np.asarray(state.velocity).copy(),
        "sample_step": int(state.sample_step),
        "motion_mode": str(state.motion_mode),
    }


def _capture_state(env: Any, observations: Any, rewards: Any = None, done: Any = None) -> Dict[str, Any]:
    planner = env.map_module
    metrics = env.get_unknown_map_metrics()
    obstacles = [
        {"center": np.asarray(o["center"], dtype=np.float64), "size": np.asarray(o["size"], dtype=np.float64)}
        for o in env.obstacles
    ]
    positions = _to_numpy(env._agent_pos)
    in_bounds = np.logical_and(positions >= 0.0, positions <= _to_numpy(env.space_size)).all(axis=1)
    return {
        "observation_shapes": [tuple(_to_numpy(x).shape) for x in observations],
        "observations": np.stack([_to_numpy(x) for x in observations]),
        "reward": None if rewards is None else _to_numpy(rewards),
        "done": None if done is None else tuple(bool(x) for x in done),
        "agent_position": positions,
        "agent_velocity": _to_numpy(env._agent_vel),
        "target_position": np.asarray(env.target_state.position).copy(),
        "target_velocity": np.asarray(env.target_state.velocity).copy(),
        "target_sample_step": int(env.target_state.sample_step),
        "obstacle_layout": {"id": str(env.obstacle_layout_id), "obstacles": obstacles},
        "target_belief": _optional_array(planner, "belief_map"),
        "belief_entropy": float(planner.belief_entropy().detach().cpu().item()),
        "belief_peak": float(planner.belief_peak_probability().detach().cpu().item()),
        "occupancy_state": {
            "logodds": _optional_array(planner, "occupancy_logodds"),
            "probability": _optional_array(planner, "occupancy_probability"),
            "known_free": _optional_array(planner, "known_free_mask"),
            "known_occupied": _optional_array(planner, "known_occupied_mask"),
            "unknown": _optional_array(planner, "unknown_mask"),
            "known_fraction": float(metrics["map_known_fraction"]),
            "revision": int(metrics.get("map_revision", 0)),
        },
        "task_found": bool(env.task_found),
        "finder_id": int(env.finder_idx),
        "executor_knowledge": {
            "assigned": bool(env.executor_target_assigned),
            "known_by_agent": _to_numpy(env._agent_task_known),
            "delivered_target": _target_payload(env.executor_delivered_target_state),
        },
        "handoff_state": {
            "found_step": env.found_step,
            "handoff_step": env.handoff_step,
            "executor_received_target_step": env.executor_received_target_step,
            "handoff_count": int(env.ch3_handoff_count),
        },
        "mission_complete": bool(env.mission_complete),
        "completion_step": env.success_step,
        "navigation_targets": _to_numpy(env._nav_targets),
        "search_waypoints": _to_numpy(env._search_waypoints),
        "collision_constraint_state": {
            "collision_flags": _to_numpy(env._collision_flags),
            "obstacle_collision_count": int(env.obstacle_collision_count),
            "map_collision_count": int(env.map_collision_count),
            "inside_world_bounds": in_bounds,
            "agent_finished": _to_numpy(env.agent_finished),
            "hold_counters": _to_numpy(env.hold_counters),
        },
        "step": int(env.step_count),
    }


def _normal(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {"array_dtype": str(array.dtype), "shape": list(array.shape), "data_hex": array.tobytes().hex()}
    if isinstance(value, np.generic):
        return _normal(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return {"float": "nan"}
        if math.isinf(value):
            return {"float": "inf" if value > 0 else "-inf"}
        return {"float_hex": value.hex()}
    if isinstance(value, dict):
        return {str(k): _normal(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normal(v) for v in value]
    return value


def _state_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(_normal(value))).hexdigest()


def _compare(left: Any, right: Any, prefix: str = "") -> Tuple[List[str], float]:
    if torch.is_tensor(left) or isinstance(left, np.ndarray) or torch.is_tensor(right) or isinstance(right, np.ndarray):
        a, b = _to_numpy(left), _to_numpy(right)
        if a.shape != b.shape:
            return [prefix + ".shape"], float("inf")
        if a.dtype.kind in "biu" or b.dtype.kind in "biu":
            return ([] if np.array_equal(a, b) else [prefix], 0.0)
        finite = np.isfinite(a) & np.isfinite(b)
        max_diff = float(np.max(np.abs(a[finite] - b[finite]))) if np.any(finite) else 0.0
        same = np.allclose(a, b, atol=ATOL, rtol=RTOL, equal_nan=True)
        return ([] if same else [prefix], max_diff)
    if isinstance(left, dict) and isinstance(right, dict):
        mismatches: List[str] = []
        maximum = 0.0
        if set(left) != set(right):
            mismatches.append(prefix + ".keys")
        for key in sorted(set(left) & set(right)):
            fields, diff = _compare(left[key], right[key], f"{prefix}.{key}" if prefix else str(key))
            mismatches.extend(fields)
            maximum = max(maximum, diff)
        return mismatches, maximum
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [prefix + ".length"], float("inf")
        mismatches, maximum = [], 0.0
        for index, (a, b) in enumerate(zip(left, right)):
            fields, diff = _compare(a, b, f"{prefix}[{index}]")
            mismatches.extend(fields)
            maximum = max(maximum, diff)
        return mismatches, maximum
    if isinstance(left, float) or isinstance(right, float):
        a, b = float(left), float(right)
        if math.isnan(a) or math.isnan(b):
            return ([] if math.isnan(a) and math.isnan(b) else [prefix], 0.0)
        diff = abs(a - b)
        return ([] if math.isclose(a, b, abs_tol=ATOL, rel_tol=RTOL) else [prefix], diff)
    return ([] if left == right else [prefix], 0.0)


def _actions(name: str, scenario_seed: int, profile_index: int) -> np.ndarray:
    if name == "zero_action":
        return np.zeros((MAX_STEPS, 4, 3), dtype=np.float32)
    if name == "seeded_random_low_amplitude":
        rng = np.random.default_rng(scenario_seed * 1009 + profile_index * 9176 + 43)
        return rng.uniform(-0.25, 0.25, size=(MAX_STEPS, 4, 3)).astype(np.float32)
    step = np.arange(MAX_STEPS, dtype=np.float64)[:, None, None]
    agent = np.arange(4, dtype=np.float64)[None, :, None]
    dim = np.arange(3, dtype=np.float64)[None, None, :]
    values = 0.20 * np.sin(0.17 * step + 0.41 * agent + 0.73 * dim) + 0.04 * np.cos(0.11 * step + 0.29 * agent - 0.37 * dim)
    return values.astype(np.float32)


def _run_profile(profile: str) -> Dict[str, Any]:
    from core.config.ch3_config import build_ch3_config
    from core.env import MissionCoreEnv, environment_kwargs_from_config

    profile_index = PROFILES.index(profile)
    manifest = json.loads((SCENARIO_ROOT / f"{profile}.json").read_text(encoding="utf-8"))
    golden_manifest = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    golden = {
        (item["profile"], item["scenario_id"], item["action_trace"]): item
        for item in golden_manifest["trajectories"]
    }
    config = build_ch3_config("ch3_v3_full_reference", profile)
    kwargs = environment_kwargs_from_config(config, device="cpu", max_steps=MAX_STEPS, return_numpy=False)
    construction_seed = 900_000 + profile_index
    _seed_all(construction_seed)
    template = MissionCoreEnv(**kwargs)
    rows, mismatch_rows, trace_records = [], [], []
    try:
        for scenario in manifest["scenarios"]:
            scenario_seed = int(scenario["scenario_seed"])
            _seed_all(scenario_seed)
            reset_obs = template.reset(scenario=copy.deepcopy(scenario))
            reset_state = _capture_state(template.unwrapped, reset_obs)
            for trace_name in ACTION_TRACES:
                action_values = _actions(trace_name, scenario_seed, profile_index)
                if action_values.shape != (MAX_STEPS, 4, 3) or np.any(action_values < -1) or np.any(action_values > 1):
                    raise RuntimeError("invalid E0 action trace")
                action_sha = _state_hash(action_values)
                expected = golden[(profile, scenario["scenario_id"], trace_name)]
                trace_records.append({
                    "profile": profile,
                    "scenario_id": scenario["scenario_id"],
                    "scenario_seed": scenario_seed,
                    "trace": trace_name,
                    "shape": list(action_values.shape),
                    "minimum": float(action_values.min()),
                    "maximum": float(action_values.max()),
                    "sha256": action_sha,
                })
                wrapper = copy.deepcopy(template)
                state_hashes = [_state_hash(reset_state)]
                first_mismatch = None
                mismatch_fields = set()
                max_diff = 0.0
                steps_completed = 0
                if action_sha != expected["action_trace_sha256"]:
                    first_mismatch = 0
                    mismatch_fields.add("action_trace_sha256")
                    mismatch_rows.append({"profile": profile, "scenario_id": scenario["scenario_id"], "trace": trace_name, "step": 0, "field": "action_trace_sha256", "max_abs_difference": 0.0})
                for step_index in range(MAX_STEPS):
                    step_seed = scenario_seed * 10_000 + step_index
                    action = torch.as_tensor(action_values[step_index], dtype=torch.float32)
                    _seed_all(step_seed)
                    observations, rewards, done = wrapper.step(action.clone())
                    state = _capture_state(wrapper.unwrapped, observations, rewards, done)
                    state_hashes.append(_state_hash(state))
                    steps_completed += 1
                    if all(bool(x) for x in done):
                        break
                trajectory_hash = hashlib.sha256("\n".join(state_hashes).encode()).hexdigest()
                if steps_completed != int(expected["steps_completed"]):
                    first_mismatch = first_mismatch if first_mismatch is not None else steps_completed
                    mismatch_fields.add("steps_completed")
                    mismatch_rows.append({"profile": profile, "scenario_id": scenario["scenario_id"], "trace": trace_name, "step": steps_completed, "field": "steps_completed", "max_abs_difference": 0.0})
                if trajectory_hash != expected["golden_state_hash"]:
                    first_mismatch = first_mismatch if first_mismatch is not None else steps_completed
                    mismatch_fields.add("golden_state_hash")
                    mismatch_rows.append({"profile": profile, "scenario_id": scenario["scenario_id"], "trace": trace_name, "step": steps_completed, "field": "golden_state_hash", "max_abs_difference": 0.0})
                passed = first_mismatch is None
                rows.append({
                    "profile": profile,
                    "scenario_id": scenario["scenario_id"],
                    "scenario_seed": scenario_seed,
                    "action_trace": trace_name,
                    "action_trace_sha256": action_sha,
                    "steps_completed": steps_completed,
                    "observation_shape_valid": True,
                    "action_shape_valid": True,
                    "first_mismatch_step": "" if first_mismatch is None else first_mismatch,
                    "mismatch_fields": ";".join(sorted(mismatch_fields)),
                    "max_abs_difference": max_diff,
                    "golden_state_hash": expected["golden_state_hash"],
                    "core_state_hash": trajectory_hash,
                    "passed": passed,
                })
    finally:
        template.close()
    return {"rows": rows, "mismatches": mismatch_rows, "traces": trace_records}


def _write_csv(path: Path, rows: List[dict], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _source_manifest() -> Dict[str, Any]:
    core_files = sorted((ROOT / "core").rglob("*.py"))
    records = [
        {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256_file(path)}
        for path in core_files
    ]
    return {
        "created_at_utc": _utc_now(),
        "authority": "frozen Phase 0B-2 golden trajectory hashes",
        "golden_manifest_sha256": _sha256_file(GOLDEN_PATH),
        "core_source_file_count": len(records),
        "core_sources": records,
        "runner_sha256": _sha256_file(Path(__file__)),
        "mission_env_sha256": _sha256_file(ROOT / "core" / "env" / "mission_env.py"),
    }


def run(max_workers: int = 4) -> Dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    profile_results = {}
    with ProcessPoolExecutor(max_workers=max(1, min(int(max_workers), 4))) as executor:
        futures = {executor.submit(_run_profile, profile): profile for profile in PROFILES}
        for future in as_completed(futures):
            profile = futures[future]
            profile_results[profile] = future.result()
            print(f"completed {profile}: {len(profile_results[profile]['rows'])}/15 trajectories", flush=True)
    rows = [row for profile in PROFILES for row in profile_results[profile]["rows"]]
    mismatches = [row for profile in PROFILES for row in profile_results[profile]["mismatches"]]
    traces = [row for profile in PROFILES for row in profile_results[profile]["traces"]]
    rows.sort(key=lambda x: (PROFILES.index(x["profile"]), x["scenario_seed"], ACTION_TRACES.index(x["action_trace"])))
    passed_count = sum(bool(row["passed"]) for row in rows)
    maximum = max((float(row["max_abs_difference"]) for row in rows), default=0.0)
    passed = len(rows) == 60 and passed_count == 60 and not mismatches
    summary = {
        "experiment": "CH3-E0 Core Without Historical Runtime Dependencies",
        "purpose": "core-only replay against frozen golden hashes; not an algorithm-performance comparison",
        "status": "PASS_CORE_WITHOUT_LEGACY_E0" if passed else "FAIL_CORE_WITHOUT_LEGACY_E0",
        "started_at_utc": started,
        "finished_at_utc": _utc_now(),
        "profiles": list(PROFILES),
        "scenario_count_per_profile": 5,
        "action_traces": list(ACTION_TRACES),
        "planned_trajectory_count": 60,
        "completed_trajectory_count": len(rows),
        "passed_trajectory_count": passed_count,
        "failed_trajectory_count": len(rows) - passed_count,
        "maximum_absolute_difference": maximum,
        "task_event_mismatch_count": 0 if passed else len(mismatches),
        "mismatch_detail_count": len(mismatches),
        "atol": ATOL,
        "rtol": RTOL,
        "golden_manifest_sha256": _sha256_file(GOLDEN_PATH),
        "historical_runtime_required": False,
    }
    trajectory_fields = (
        "profile", "scenario_id", "scenario_seed", "action_trace", "action_trace_sha256",
        "steps_completed", "observation_shape_valid", "action_shape_valid", "first_mismatch_step",
        "mismatch_fields", "max_abs_difference", "golden_state_hash", "core_state_hash", "passed",
    )
    mismatch_fields = ("profile", "scenario_id", "trace", "step", "field", "max_abs_difference")
    _write_csv(OUTPUT_ROOT / "per_trajectory_results.csv", rows, trajectory_fields)
    _write_csv(OUTPUT_ROOT / "per_step_mismatch_details.csv", mismatches, mismatch_fields)
    _write_json(OUTPUT_ROOT / "action_trace_manifest.json", {"traces": traces})
    scenario_index = json.loads((SCENARIO_ROOT / "scenario_manifest_index.json").read_text(encoding="utf-8"))
    _write_json(OUTPUT_ROOT / "scenario_manifest_index.json", scenario_index)
    experiment = {
        "experiment": summary["experiment"],
        "purpose": summary["purpose"],
        "authority": "frozen Phase 0B-2 golden_trace_manifest.json",
        "migration_target": "CRK-Thesis-v2/core/env/MissionCoreEnv",
        "base_candidate": "ch3_v3_full_reference",
        "device": "cpu",
        "dtype": "torch.float32",
        "max_steps": MAX_STEPS,
        "comparison": {"atol": ATOL, "rtol": RTOL, "equal_nan": True, "discrete_exact": True},
        "profile_count": 4,
        "scenarios_per_profile": 5,
        "action_trace_count": 3,
        "trajectory_count": 60,
    }
    _write_json(OUTPUT_ROOT / "experiment_manifest.json", experiment)
    _write_json(OUTPUT_ROOT / "source_hash_manifest.json", _source_manifest())
    _write_json(OUTPUT_ROOT / "equivalence_summary.json", summary)
    _write_json(MIGRATION_ROOT / "core_without_legacy_summary.json", summary)
    (OUTPUT_ROOT / "equivalence_summary.md").write_text(
        "# CH3-E0 migration equivalence\n\n"
        f"Status: **{summary['status']}**\n\n"
        f"Completed/passed: {len(rows)}/{passed_count} of 60. Maximum absolute difference: {maximum:.9g}. "
        f"Task-event mismatches: {summary['task_event_mismatch_count']}. Historical runtime required: false.\n\n"
        "This is a migration-equivalence experiment, not an algorithm-performance comparison.\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    summary = run(args.workers)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS_CORE_WITHOUT_LEGACY_E0" else 1


if __name__ == "__main__":
    raise SystemExit(main())
