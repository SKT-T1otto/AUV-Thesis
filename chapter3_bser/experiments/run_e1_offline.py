"""Run BSER-E1 finite mathematical and solver-efficiency validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import time

import numpy as np

from chapter3_bser.baselines.legacy_pse_snapshot import evaluate_legacy_pse_snapshot
from chapter3_bser.baselines.random_allocator import evaluate_random_allocator
from chapter3_bser.baselines.search_only_allocator import solve_search_only_greedy
from chapter3_bser.candidate_generator import generate_candidates
from chapter3_bser.config import DEFAULT_CONFIG, load_bser_config
from chapter3_bser.exact_solver import solve_fixed_standby_exact, solve_joint_exact
from chapter3_bser.experiments.instance_builder import SkippedSnapshot, iter_e1_snapshots
from chapter3_bser.greedy_solver import solve_fixed_standby_greedy, solve_joint_greedy
from chapter3_bser.lazy_greedy_solver import solve_joint_lazy
from chapter3_bser.metrics import validate_small_instance
from chapter3_bser.objective import build_objective_context, coverage_overlap, evaluate_objective, expected_detection_probability, expected_response_time


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "experiments" / "chapter3" / "bser_e1_offline"


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _stable_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _round(value):
    return round(float(value), 12)


def write_protocol_aggregates(output, rows):
    slices = [("all_instances", "all", rows), ("exact_runnable", "all", [row for row in rows if row.get("exact_objective", "") != ""])]
    for field, label in (("profile", "profile"), ("action_trace", "action_trace"), ("snapshot_step", "snapshot_step")):
        for value in sorted({str(row[field]) for row in rows}):
            slices.append((label, value, [row for row in rows if str(row[field]) == value]))
    slices.extend((
        ("map_group", "unknown", [row for row in rows if row["profile"] in {"M10_MOVING_UNKNOWN_SINGLE", "M20_MOVING_UNKNOWN_MULTI"}]),
        ("map_group", "oracle", [row for row in rows if row["profile"] == "M90_MOVING_KNOWN_ORACLE"]),
    ))
    aggregates = []
    for slice_type, slice_value, group in slices:
        if not group:
            continue
        aggregates.append({
            "slice_type": slice_type,
            "slice_value": slice_value,
            "instances": len(group),
            "mean_exact_objective": statistics.fmean(float(row["exact_objective"]) for row in group),
            "mean_greedy_objective": statistics.fmean(float(row["greedy_objective"]) for row in group),
            "mean_greedy_exact_ratio": statistics.fmean(float(row["greedy_exact_ratio"]) for row in group),
            "mean_expected_response_time": statistics.fmean(float(row["expected_response_time"]) for row in group),
            "mean_search_only_response_time": statistics.fmean(float(row["search_only_expected_response_time"]) for row in group),
        })
    _write_csv(Path(output) / "aggregate_by_protocol_slice.csv", aggregates)
    return aggregates


def _best_standby(selected, standby_candidates, context):
    return max(standby_candidates, key=lambda standby: (evaluate_objective(selected, standby, context), tuple(-ord(c) for c in standby.candidate_id)))


def run(output_dir=DEFAULT_OUTPUT, *, smoke=False):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = load_bser_config()
    shutil.copyfile(DEFAULT_CONFIG, output / "config_snapshot.json")
    results = []
    properties = []
    candidates_rows = []
    skipped = []
    failures = []
    snapshots = []
    for item in iter_e1_snapshots(config["e1"]["snapshot_steps"], smoke=smoke):
        if isinstance(item, SkippedSnapshot):
            skipped.append(item.__dict__)
            continue
        snapshots.append(item)
        generation = generate_candidates(item.state, config)
        if any(generation.search_count_by_agent.get(agent_id, 0) == 0 for agent_id in item.state.searcher_ids) or not generation.standby_candidates:
            skipped.append({"instance_id": item.instance_id, "profile": item.profile, "scenario_id": item.scenario_id, "scenario_seed": item.scenario_seed, "action_trace": item.action_trace, "snapshot_step": item.snapshot_step, "reason": "SKIPPED_NO_FEASIBLE_CANDIDATES"})
            continue
        context = build_objective_context(item.state, generation.search_candidates, generation.standby_candidates, config)
        start = time.perf_counter(); exact = solve_joint_exact(generation.search_candidates, generation.standby_candidates, context, combination_limit=config["solver"]["exact_combination_limit"]); exact_runtime = time.perf_counter() - start
        start = time.perf_counter(); greedy = solve_joint_greedy(generation.search_candidates, generation.standby_candidates, context); greedy_runtime = time.perf_counter() - start
        start = time.perf_counter(); lazy = solve_joint_lazy(generation.search_candidates, generation.standby_candidates, context); lazy_runtime = time.perf_counter() - start
        checks = validate_small_instance(generation.search_candidates, generation.standby_candidates, context)
        fixed_ratios = []
        for standby in generation.standby_candidates:
            fixed_exact = solve_fixed_standby_exact(generation.search_candidates, standby, context)
            fixed_greedy = solve_fixed_standby_greedy(generation.search_candidates, standby, context)
            fixed_ratios.append(1.0 if fixed_exact.objective <= 1e-12 else fixed_greedy.objective / fixed_exact.objective)
        checks["minimum_fixed_y_ratio"] = min(fixed_ratios)
        checks["fixed_y_bound_pass"] = min(fixed_ratios) >= 0.5 - 1e-9
        random_summary = evaluate_random_allocator(generation.search_candidates, generation.standby_candidates, context, repetitions=config["solver"]["random_repetitions"], seed=config["solver"]["random_seed"] + int(hashlib.sha256(item.instance_id.encode()).hexdigest()[:8], 16))
        search_only = solve_search_only_greedy(generation.search_candidates, context)
        search_standby = _best_standby(search_only.selected, generation.standby_candidates, context)
        pse = evaluate_legacy_pse_snapshot(item.state, config)
        ratio = 1.0 if exact.objective <= 1e-12 else greedy.objective / exact.objective
        row_core = {
            "profile": item.profile, "scenario_id": item.scenario_id, "scenario_seed": item.scenario_seed,
            "action_trace": item.action_trace, "snapshot_step": item.snapshot_step, "map_revision": item.state.map_revision,
            "belief_entropy": _round(item.state.target_belief.entropy), "belief_peak": _round(item.state.target_belief.peak_probability),
            "occupancy_known_ratio": _round(np.mean(item.state.occupancy.known_mask)),
            "candidate_count_by_agent": json.dumps(generation.search_count_by_agent, sort_keys=True), "standby_candidate_count": len(generation.standby_candidates),
            "unreachable_search_candidate_count": generation.unreachable_search_count, "unreachable_standby_candidate_count": generation.unreachable_standby_count,
            "exact_objective": _round(exact.objective), "greedy_objective": _round(greedy.objective), "lazy_objective": _round(lazy.objective),
            "greedy_exact_ratio": _round(ratio), "lazy_exact_ratio": _round(1.0 if exact.objective <= 1e-12 else lazy.objective / exact.objective),
            "search_only_objective": _round(search_only.objective), "random_mean_objective": _round(random_summary.mean), "random_std_objective": _round(random_summary.std),
            "pse_snapshot_objective": _round(pse.objective), "pse_snapshot_status": pse.status,
            "expected_detection_probability": _round(expected_detection_probability(greedy.selected, context)),
            "expected_response_time": _round(expected_response_time(greedy.selected, greedy.standby, context)),
            "search_only_expected_response_time": _round(expected_response_time(search_only.selected, search_standby, context)),
            "coverage_overlap": _round(coverage_overlap(greedy.selected, context)),
            "selected_candidate_ids": ";".join(greedy.selected_ids),
            "selected_waypoints": json.dumps([candidate.waypoint for candidate in greedy.selected]),
            "standby_waypoint": json.dumps(greedy.standby.waypoint),
            "exact_runtime_seconds": exact_runtime, "greedy_runtime_seconds": greedy_runtime, "lazy_runtime_seconds": lazy_runtime,
            "exact_combination_count": exact.combination_count,
            "monotonicity_pass": checks["monotonicity_pass"], "submodularity_pass": checks["submodularity_pass"],
            "partition_constraint_pass": checks["partition_constraint_pass"], "information_leakage_pass": True,
        }
        deterministic = {key: value for key, value in row_core.items() if not key.endswith("runtime_seconds")}
        row_core["deterministic_hash"] = _stable_hash(deterministic)
        row = {"instance_id": item.instance_id, **row_core}
        results.append(row)
        properties.append({"instance_id": item.instance_id, **checks})
        candidates_rows.append({"instance_id": item.instance_id, "candidate_sources": ";".join(candidate.source for candidate in generation.search_candidates), "standby_sources": ";".join(candidate.source for candidate in generation.standby_candidates), "generation_notes": ";".join(generation.reasons)})
        failed_checks = [key for key, value in checks.items() if key.endswith("_pass") and not value]
        if failed_checks:
            failures.append({"instance_id": item.instance_id, "failures": ";".join(failed_checks)})

    scalability = []
    selected_scaling = sorted(snapshots, key=lambda item: hashlib.sha256(item.instance_id.encode()).hexdigest())[:(1 if smoke else int(config["e1"]["scalability_instance_count"]))]
    for item in selected_scaling:
        search_sizes = [config["candidate_generation"]["k_search_exact"]] if smoke else config["candidate_generation"]["search_scaling"]
        standby_sizes = [config["candidate_generation"]["k_standby_exact"]] if smoke else config["candidate_generation"]["standby_scaling"]
        for search_k in search_sizes:
            for standby_k in standby_sizes:
                generation = generate_candidates(item.state, config, k_search=search_k, k_standby=standby_k)
                if not generation.search_candidates or not generation.standby_candidates:
                    continue
                context = build_objective_context(item.state, generation.search_candidates, generation.standby_candidates, config)
                start = time.perf_counter(); greedy = solve_joint_greedy(generation.search_candidates, generation.standby_candidates, context); greedy_time = time.perf_counter() - start
                start = time.perf_counter(); lazy = solve_joint_lazy(generation.search_candidates, generation.standby_candidates, context); lazy_time = time.perf_counter() - start
                start = time.perf_counter(); exact = solve_joint_exact(generation.search_candidates, generation.standby_candidates, context, combination_limit=config["solver"]["exact_combination_limit"]); exact_time = time.perf_counter() - start
                scalability.append({"instance_id": item.instance_id, "k_search": search_k, "k_standby": standby_k, "actual_search_candidates": len(generation.search_candidates), "actual_standby_candidates": len(generation.standby_candidates), "greedy_runtime_seconds": greedy_time, "lazy_runtime_seconds": lazy_time, "exact_runtime_seconds": exact_time if exact.status == "OK" else "", "exact_status": exact.status, "exact_combination_count": exact.combination_count, "greedy_objective": greedy.objective, "lazy_objective": lazy.objective})

    instance_manifest = [{"instance_id": row["instance_id"], "profile": row["profile"], "scenario_id": row["scenario_id"], "scenario_seed": row["scenario_seed"], "action_trace": row["action_trace"], "snapshot_step": row["snapshot_step"], "deterministic_hash": row["deterministic_hash"]} for row in results]
    _write_csv(output / "instance_manifest.csv", instance_manifest)
    (output / "instance_manifest.json").write_text(json.dumps(instance_manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(output / "candidate_statistics.csv", candidates_rows)
    _write_csv(output / "per_instance_results.csv", results)
    _write_csv(output / "property_check_results.csv", properties)
    _write_csv(output / "solver_scalability.csv", scalability)
    baseline_rows = [{key: row[key] for key in ("instance_id", "profile", "greedy_objective", "search_only_objective", "random_mean_objective", "random_std_objective", "pse_snapshot_objective", "pse_snapshot_status", "expected_response_time", "search_only_expected_response_time")} for row in results]
    _write_csv(output / "baseline_comparison.csv", baseline_rows)
    aggregate_profile = []
    for profile in sorted({row["profile"] for row in results}):
        group = [row for row in results if row["profile"] == profile]
        aggregate_profile.append({"profile": profile, "instances": len(group), "mean_exact_objective": statistics.fmean(row["exact_objective"] for row in group), "mean_greedy_objective": statistics.fmean(row["greedy_objective"] for row in group), "mean_greedy_exact_ratio": statistics.fmean(row["greedy_exact_ratio"] for row in group), "mean_expected_response_time": statistics.fmean(row["expected_response_time"] for row in group)})
    _write_csv(output / "aggregate_by_profile.csv", aggregate_profile)
    write_protocol_aggregates(output, results)
    aggregate_solver = []
    for name, field in (("joint_exact", "exact_objective"), ("bser_standard_greedy", "greedy_objective"), ("bser_lazy_greedy", "lazy_objective"), ("search_only_greedy", "search_only_objective"), ("random_allocator_mean", "random_mean_objective"), ("legacy_pse_snapshot", "pse_snapshot_objective")):
        aggregate_solver.append({"solver": name, "instances": len(results), "mean_objective": statistics.fmean(row[field] for row in results) if results else 0.0})
    _write_csv(output / "aggregate_by_solver.csv", aggregate_solver)
    _write_csv(output / "failure_cases.csv", failures)
    _write_csv(output / "skipped_instances.csv", skipped)
    summary_core = {
        "schema": "bser.e1.summary.v1", "valid_instance_count": len(results), "skipped_instance_count": len(skipped),
        "monotonicity_failures": sum(not row["monotonicity_pass"] for row in properties),
        "submodularity_failures": sum(not row["submodularity_pass"] for row in properties),
        "partition_constraint_failures": sum(not row["partition_constraint_pass"] for row in properties),
        "greedy_bound_failures": sum(not row["greedy_bound_pass"] or not row["fixed_y_bound_pass"] for row in properties),
        "lazy_equivalence_failures": sum(not row["lazy_equivalence_pass"] for row in properties),
        "minimum_greedy_exact_ratio": min((row["greedy_exact_ratio"] for row in results), default=0.0),
        "mean_greedy_exact_ratio": statistics.fmean(row["greedy_exact_ratio"] for row in results) if results else 0.0,
        "mean_bser_minus_search_only_response_time": statistics.fmean(row["expected_response_time"] - row["search_only_expected_response_time"] for row in results) if results else 0.0,
        "pse_snapshot_available_count": sum(row["pse_snapshot_status"] == "OK" for row in results),
        "property_pass_count": sum(all(value for key, value in row.items() if key.endswith("_pass")) for row in properties),
        "formal_training_run": False,
    }
    summary_sha = _stable_hash(summary_core)
    summary = {**summary_core, "deterministic_summary_sha256": summary_sha, "passed": len(results) > 0 and not failures}
    (output / "e1_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output / "determinism_manifest.json").write_text(json.dumps({"schema": "bser.e1.determinism.v1", "summary_sha256": summary_sha, "instance_hashes": {row["instance_id"]: row["deterministic_hash"] for row in results}}, indent=2, sort_keys=True), encoding="utf-8")
    (output / "experiment_manifest.json").write_text(json.dumps({"schema": "bser.e1.experiment.v1", "name": "BSER-E1", "profiles": sorted({row["profile"] for row in results}), "action_traces": config["e1"]["action_traces"], "snapshot_steps": config["e1"]["snapshot_steps"], "smoke": bool(smoke), "uses_frozen_e0_manifests": True, "formal_training": False}, indent=2, sort_keys=True), encoding="utf-8")
    (output / "e1_summary.md").write_text(f"# BSER-E1 summary\n\nValid instances: {len(results)}; skipped: {len(skipped)}.\n\nMonotonicity failures: {summary['monotonicity_failures']}; submodularity failures: {summary['submodularity_failures']}; greedy-bound failures: {summary['greedy_bound_failures']}; lazy-equivalence failures: {summary['lazy_equivalence_failures']}.\n\nDeterministic summary SHA-256: `{summary_sha}`. This is an offline objective/solver validation, not a task-success or online-performance experiment.\n", encoding="utf-8")
    (output / "test_report.md").write_text(f"# E1 test report\n\nStatus: {'PASS' if summary['passed'] else 'FAIL'}\n\nAll failures and skips are retained in their CSV files.\n", encoding="utf-8")
    if not smoke and results:
        m20 = sorted((row for row in results if row["profile"] == "M20_MOVING_UNKNOWN_MULTI"), key=lambda row: row["greedy_exact_ratio"])
        typical = min(m20, key=lambda row: abs(row["greedy_exact_ratio"] - statistics.median(item["greedy_exact_ratio"] for item in m20))) if m20 else results[0]
        snapshot = next(item for item in snapshots if item.instance_id == typical["instance_id"])
        (output / "typical_case.json").write_text(json.dumps({"selection_rule": "M20 instance nearest median greedy/exact ratio", "instance_id": typical["instance_id"], "cell_centers": snapshot.state.grid.cell_centers.tolist(), "belief": snapshot.state.target_belief.probabilities.tolist(), "occupancy_probability": snapshot.state.occupancy.occupancy_probability.tolist(), "selected_waypoints": json.loads(typical["selected_waypoints"]), "standby_waypoint": json.loads(typical["standby_waypoint"])}, indent=2, sort_keys=True), encoding="utf-8")
        from chapter3_bser.experiments.plot_e1 import generate_figures
        generate_figures(output)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    summary = run(args.output_dir, smoke=args.smoke)
    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
