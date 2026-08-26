"""Run the 80-condition-episode execution-consistent BSER pilot."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from chapter3_bser.experiments.phase1b1_pilot.run_pilot import (
    MemoizedAllocator,
    _episode,
    _mean,
    _write_csv,
    _write_json,
)
from chapter3_bser.online.config import load_phase1b1_config, load_phase1b2_config
from core.scenarios.ch3_generator_impl import build_scenario_manifests


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT / "docs2" / "phase1b2"
METHODS = (
    "No-BSER-static",
    "Periodic-BSER",
    "Event-BSER-phase1b1",
    "Event-BSER-phase1b2_corrected",
)
_WORKER_ALLOCATOR = None


def _worker(job):
    global _WORKER_ALLOCATOR
    torch.set_num_threads(1)
    if _WORKER_ALLOCATOR is None:
        _WORKER_ALLOCATOR = MemoizedAllocator()
    metric, _ = _episode(*job, execution_consistent=True, allocator=_WORKER_ALLOCATOR)
    return metric


def _summary(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        group = [row for row in metrics if row["method"] == method]
        successes = [float(row["completion_step"]) for row in group if row["completion_step"] != ""]
        obstacle_count = sum(int(row["obstacle_event_count"]) for row in group)
        impacted_count = sum(int(row["obstacle_route_impacted_count"]) for row in group)
        partial_attempts = sum(int(row["partial_reallocation_attempt_count"]) for row in group)
        partial_successes = sum(int(row["partial_reallocation_success_count"]) for row in group)
        rows.append(
            {
                "method": method,
                "episode_count": len(group),
                "success_rate": sum(bool(row["success"]) for row in group) / len(group),
                "found_rate": sum(row["found_step"] != "" for row in group) / len(group),
                "mean_completion_step_success_only": _mean(successes),
                "mean_completion_step_truncated": _mean(
                    float(row["completion_time_truncated"]) for row in group
                ),
                "executor_invalid_count": sum(int(row["executor_invalid_count"]) for row in group),
                "waypoint_stale_count": sum(int(row["waypoint_stale_count"]) for row in group),
                "collision_count": sum(int(row["collision_count"]) for row in group),
                "accepted_replan_count": sum(int(row["accepted_replan_count"]) for row in group),
                "rejected_replan_count": sum(int(row["rejected_replan_count"]) for row in group),
                "optimizer_invocation_count": sum(int(row["optimizer_invocation_count"]) for row in group),
                "mean_replans": _mean(float(row["replan_count"]) for row in group),
                "waypoint_switch_count": sum(int(row["waypoint_switch_count"]) for row in group),
                "total_switch_distance": sum(float(row["total_switch_distance"]) for row in group),
                "path_tracking_error": _mean(float(row["path_tracking_error"]) for row in group),
                "route_impact_ratio": impacted_count / obstacle_count if obstacle_count else 0.0,
                "partial_reallocation_attempt_count": partial_attempts,
                "partial_reallocation_success_count": partial_successes,
                "partial_reallocation_success_rate": (
                    partial_successes / partial_attempts if partial_attempts else 0.0
                ),
            }
        )
    return rows


def _checkpoint_path(output: Path, seed: int, episode_index: int, method: str) -> Path:
    return output / "_checkpoints" / f"s{seed}_e{episode_index}_m{METHODS.index(method)}.json"


def _valid_checkpoint(value: dict[str, Any], *, seed: int, episode_index: int, method: str, max_steps: int) -> bool:
    return bool(
        value.get("schema") == "bser.phase1b2.pilot.job.v1"
        and int(value.get("scenario_seed", -1)) == seed
        and int(value.get("episode_index", -1)) == episode_index
        and value.get("method") == method
        and int(value.get("max_steps", -1)) == max_steps
    )


def run(output_dir: Path = DEFAULT_OUTPUT, *, smoke: bool = False, workers: int = 4) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = load_phase1b2_config()
    seeds = tuple(int(value) for value in config["experiment"]["scenario_seeds"])
    indices = (0,) if smoke else tuple(int(value) for value in config["experiment"]["episode_indices"])
    run_seeds = seeds[:1] if smoke else seeds
    max_steps = 5 if smoke else int(config["experiment"]["max_steps"])
    manifest = build_scenario_manifests(
        count=5,
        generator_seed=seeds[0],
        split="validation",
        profiles=("M20_MOVING_UNKNOWN_MULTI",),
    )["M20_MOVING_UNKNOWN_MULTI"]
    scenarios = {int(row["scenario_seed"]): row for row in manifest["scenarios"]}
    if tuple(sorted(scenarios)) != seeds:
        raise RuntimeError("Phase 1B.2 scenario generator did not produce seeds 2729-2733")

    metrics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    pending = []
    for seed in run_seeds:
        for episode_index in indices:
            for method in METHODS:
                path = _checkpoint_path(output, seed, episode_index, method)
                checkpoint = None
                if path.is_file():
                    candidate = json.loads(path.read_text(encoding="utf-8"))
                    if _valid_checkpoint(
                        candidate,
                        seed=seed,
                        episode_index=episode_index,
                        method=method,
                        max_steps=max_steps,
                    ):
                        checkpoint = candidate
                if checkpoint is None:
                    pending.append((method, copy.deepcopy(scenarios[seed]), episode_index, max_steps))
                elif checkpoint.get("failure") is not None:
                    failures.append(checkpoint["failure"])
                else:
                    metrics.append(checkpoint["metric"])

    if pending:
        with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 4))) as executor:
            future_map = {executor.submit(_worker, job): job for job in pending}
            for future in as_completed(future_map):
                method, scenario, episode_index, _ = future_map[future]
                seed = int(scenario["scenario_seed"])
                checkpoint = {
                    "schema": "bser.phase1b2.pilot.job.v1",
                    "scenario_seed": seed,
                    "episode_index": episode_index,
                    "method": method,
                    "max_steps": max_steps,
                    "metric": None,
                    "failure": None,
                }
                try:
                    checkpoint["metric"] = future.result()
                    metrics.append(checkpoint["metric"])
                except Exception as exc:
                    checkpoint["failure"] = {
                        "method": method,
                        "scenario_seed": seed,
                        "episode_index": episode_index,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                    failures.append(checkpoint["failure"])
                _write_json(
                    _checkpoint_path(output, seed, episode_index, method),
                    checkpoint,
                )

    order = {method: index for index, method in enumerate(METHODS)}
    metrics.sort(key=lambda row: (order[row["method"]], int(row["scenario_seed"]), int(row["episode_index"])))
    failures.sort(key=lambda row: (order[row["method"]], int(row["scenario_seed"]), int(row["episode_index"])))
    planned = len(METHODS) * len(run_seeds) * len(indices)
    complete = not failures and len(metrics) == planned
    method_summary = _summary(metrics) if complete else []
    by_method = {row["method"]: row for row in method_summary}
    gates = {}
    experiment_passed = False
    if complete:
        baseline = by_method["Event-BSER-phase1b1"]
        corrected = by_method["Event-BSER-phase1b2_corrected"]
        gates = {
            "success_rate_at_least_0_15": corrected["success_rate"] >= 0.15,
            "executor_invalid_count_decreased": corrected["executor_invalid_count"] < baseline["executor_invalid_count"],
            "waypoint_stale_count_decreased": corrected["waypoint_stale_count"] < baseline["waypoint_stale_count"],
            "accepted_replans_not_increased": corrected["accepted_replan_count"] <= baseline["accepted_replan_count"],
            "rejected_replans_not_increased": corrected["rejected_replan_count"] <= baseline["rejected_replan_count"],
            "mean_replans_at_most_20": corrected["mean_replans"] <= 20.0,
        }
        experiment_passed = all(gates.values())
    status = (
        "PASS_BSER_PHASE1B2_PILOT_PENDING_FINAL_TESTS"
        if experiment_passed
        else "FAIL_BSER_PHASE1B2_PILOT"
    )
    snapshot = {
        "phase1b1": load_phase1b1_config(),
        "phase1b2_corrected": config,
    }
    snapshot_hash = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    protocol = {
        "schema": "bser.phase1b2.pilot.experiment.v1",
        "profile": "M20_MOVING_UNKNOWN_MULTI",
        "scenario_seeds": list(run_seeds),
        "episode_indices": list(indices),
        "methods": list(METHODS),
        "planned_condition_episode_count": planned,
        "max_steps": max_steps,
        "execution_consistent_path_tracking": True,
        "formal_training": False,
        "oracle_access": False,
        "smoke": smoke,
    }
    result = {
        "schema": "bser.phase1b2.pilot.summary.v1",
        "status": status,
        "planned_condition_episode_count": planned,
        "completed_condition_episode_count": len(metrics),
        "failure_count": len(failures),
        "experiment_gates": gates,
        "experiment_passed": experiment_passed,
        "method_summary": method_summary,
        "optimizer_invocation_overhead": (
            {
                "phase1b1": by_method["Event-BSER-phase1b1"]["optimizer_invocation_count"],
                "phase1b2": by_method["Event-BSER-phase1b2_corrected"]["optimizer_invocation_count"],
                "relative_change": (
                    by_method["Event-BSER-phase1b2_corrected"]["optimizer_invocation_count"]
                    / by_method["Event-BSER-phase1b1"]["optimizer_invocation_count"]
                    - 1.0
                ),
                "residual_risk": "Phase 1B.2 invokes the optimizer more often despite fewer accepted and rejected replans.",
            }
            if complete
            else None
        ),
        "config_snapshot_sha256": snapshot_hash,
        "formal_training_run": False,
        "oracle_access": False,
    }
    _write_json(output / "config_snapshot.json", snapshot)
    _write_json(output / "experiment_manifest.json", protocol)
    _write_json(output / "delivery_validation.json", result)
    _write_json(
        output / "test_report.json",
        {"schema": "bser.phase1b2.test_report.v1", "status": "PENDING_FINAL_TESTS"},
    )
    _write_csv(output / "episode_metrics.csv", metrics)
    _write_csv(output / "method_summary.csv", method_summary)
    _write_csv(
        output / "failure_cases.csv",
        failures,
        fields=["method", "scenario_seed", "episode_index", "error_type", "message"],
    )
    lines = [
        "# Phase 1B.2 summary",
        "",
        f"Status: **{status}**.",
        "",
        f"Completed {len(metrics)}/{planned} condition-episodes; failures: {len(failures)}.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    table = [
        "# Phase 1B.2 pilot results",
        "",
        "| Method | Success | Found | Completion | Invalid | Stale | Replans | Tracking error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_summary:
        table.append(
            f"| {row['method']} | {row['success_rate']:.3f} | {row['found_rate']:.3f} | "
            f"{row['mean_completion_step_truncated']:.3f} | {row['executor_invalid_count']} | "
            f"{row['waypoint_stale_count']} | {row['mean_replans']:.3f} | "
            f"{row['path_tracking_error']:.3f} |"
        )
    (output / "pilot_results.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    changed = output / "changed_files.txt"
    if not changed.exists():
        changed.write_text("", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = run(args.output_dir, smoke=args.smoke, workers=args.workers)
    raise SystemExit(0 if result["failure_count"] == 0 else 1)


if __name__ == "__main__":
    main()
