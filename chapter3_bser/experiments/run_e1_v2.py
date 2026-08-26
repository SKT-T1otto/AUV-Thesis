"""Run the deduplicated BSER E1-v2 offline validation protocol."""

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
from chapter3_bser.baselines.search_only_allocator import solve_search_only_greedy
from chapter3_bser.candidate_generator import generate_candidates
from chapter3_bser.config import PHASE1A1_CONFIG, load_bser_phase1a1_config
from chapter3_bser.exact_solver import partition_groups, solve_fixed_standby_exact, solve_joint_exact
from chapter3_bser.experiments.instance_builder import SkippedSnapshot, iter_e1_snapshots
from chapter3_bser.greedy_solver import solve_fixed_standby_greedy, solve_joint_greedy
from chapter3_bser.lazy_greedy_solver import solve_joint_lazy
from chapter3_bser.metrics import validate_small_instance
from chapter3_bser.objective import (
    build_objective_context, coverage_overlap, evaluate_objective,
    expected_detection_probability, response_diagnostics,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "experiments" / "chapter3" / "bser_e1_offline_v2"


def _write_csv(path, rows, fields=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fields or (sorted({key for row in rows for key in row}) if rows else ["status"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _finite(value):
    return "" if not np.isfinite(float(value)) else float(value)


def _metadata_row(item):
    metadata = item.metadata
    return {
        "protocol_snapshot_id": item.instance_id,
        "profile": metadata.profile,
        "scenario_id": metadata.scenario_id,
        "scenario_seed": metadata.scenario_seed,
        "action_trace": metadata.action_trace,
        "requested_snapshot_step": metadata.requested_snapshot_step,
        "realized_step": metadata.realized_step,
        "obstacle_layout_id": metadata.obstacle_layout_id,
        "termination_step": "" if metadata.termination_step is None else metadata.termination_step,
        "termination_reason": metadata.termination_reason or "",
        "target_found_step": "" if metadata.target_found_step is None else metadata.target_found_step,
        "mission_complete_step": "" if metadata.mission_complete_step is None else metadata.mission_complete_step,
    }


def _best_standby(selected, standby_candidates, context):
    return max(standby_candidates, key=lambda standby: (evaluate_objective(selected, standby, context), tuple(-ord(c) for c in standby.candidate_id)))


def _diagnostic_columns(diag):
    return {
        "total_detected_mass": diag.total_detected_mass,
        "reachable_detected_mass": diag.reachable_detected_mass,
        "unreachable_detected_mass": diag.unreachable_detected_mass,
        "unreachable_detected_mass_ratio": diag.unreachable_detected_mass_ratio,
        "conditional_reachable_response_time": _finite(diag.conditional_reachable_response_time),
        "maximum_reachable_response_time": _finite(diag.maximum_reachable_response_time),
        "response_defined": diag.response_defined,
        "all_detected_mass_reachable": diag.all_detected_mass_reachable,
    }


def _method_metrics(method, result, context, *, own_objective=""):
    if result is None or result.standby is None:
        return {
            "method": method, "joint_objective": "", "detection_probability": "",
            "response_defined": False, "status": "UNAVAILABLE", "own_scale_objective": own_objective,
        }
    diag = response_diagnostics(result.selected, result.standby, context)
    return {
        "method": method,
        "joint_objective": evaluate_objective(result.selected, result.standby, context),
        "detection_probability": expected_detection_probability(result.selected, context),
        "own_scale_objective": own_objective,
        "selected_candidate_ids": ";".join(result.selected_ids),
        "standby_candidate_id": result.standby.candidate_id,
        "status": result.status,
        **_diagnostic_columns(diag),
    }


def _random_metrics(candidates, standby_candidates, context, repetitions, seed):
    rng = np.random.default_rng(int(seed)); groups = partition_groups(candidates); rows = []
    for _ in range(int(repetitions)):
        selected = []
        for group in groups:
            choice = int(rng.integers(0, len(group) + 1))
            if choice: selected.append(group[choice - 1])
        standby = standby_candidates[int(rng.integers(0, len(standby_candidates)))]
        diag = response_diagnostics(selected, standby, context)
        rows.append((evaluate_objective(selected, standby, context), expected_detection_probability(selected, context), diag))
    def mean(name): return statistics.fmean(float(getattr(row[2], name)) for row in rows)
    return {
        "method": "random_under_joint", "status": "OK", "own_scale_objective": "",
        "joint_objective": statistics.fmean(row[0] for row in rows),
        "detection_probability": statistics.fmean(row[1] for row in rows),
        "total_detected_mass": mean("total_detected_mass"),
        "reachable_detected_mass": mean("reachable_detected_mass"),
        "unreachable_detected_mass": mean("unreachable_detected_mass"),
        "unreachable_detected_mass_ratio": mean("unreachable_detected_mass_ratio"),
        "conditional_reachable_response_time": statistics.fmean(
            row[2].conditional_reachable_response_time for row in rows if row[2].response_defined
        ) if any(row[2].response_defined for row in rows) else "",
        "maximum_reachable_response_time": statistics.fmean(
            row[2].maximum_reachable_response_time for row in rows if row[2].response_defined
        ) if any(row[2].response_defined for row in rows) else "",
        "response_defined": any(row[2].response_defined for row in rows),
        "all_detected_mass_reachable": all(row[2].all_detected_mass_reachable for row in rows),
        "selected_candidate_ids": "", "standby_candidate_id": "",
    }


def _allocation_jaccard(left, right):
    a = set(left.selected_ids); b = set(right.selected_ids)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def _aggregate(rows, keys):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in keys); groups.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        output.append({
            **dict(zip(keys, key)), "protocol_snapshot_count": len(group),
            "mean_joint_objective": statistics.fmean(float(row["greedy_joint_objective"]) for row in group),
            "mean_detection_probability": statistics.fmean(float(row["greedy_detection_probability"]) for row in group),
            "mean_conditional_reachable_response_time": statistics.fmean(
                float(row["greedy_conditional_response_time"]) for row in group if row["greedy_conditional_response_time"] != ""
            ) if any(row["greedy_conditional_response_time"] != "" for row in group) else "",
            "mean_unreachable_detected_mass": statistics.fmean(float(row["greedy_unreachable_detected_mass"]) for row in group),
        })
    return output


def run(output_dir=DEFAULT_OUTPUT, *, smoke=False, run_sensitivity=True):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    config = load_bser_phase1a1_config(); shutil.copyfile(PHASE1A1_CONFIG, output / "config_snapshot.json")
    protocol = []; skipped = []; unique = {}; mapping = []
    for item in iter_e1_snapshots(config["e1_v2"]["snapshot_steps"], smoke=smoke):
        row = _metadata_row(item)
        if isinstance(item, SkippedSnapshot):
            row.update(status="SKIPPED", skip_reason=item.reason, unique_state_sha256="")
            skipped.append(dict(row)); protocol.append(row); continue
        row.update(status="VALID", skip_reason="", unique_state_sha256=item.unique_state_sha256)
        protocol.append(row); mapping.append({"protocol_snapshot_id": item.instance_id, "unique_state_sha256": item.unique_state_sha256})
        unique.setdefault(item.unique_state_sha256, item)
    expected = 4 if smoke else int(config["e1_v2"]["protocol_request_count"])
    if len(protocol) != expected:
        raise RuntimeError(f"protocol request count {len(protocol)} != {expected}")

    unique_manifest = []
    per_unique = []; properties = []; candidate_rows = []; method_rows = []; failures = []; equivalence_refs = []
    results_by_hash = {}; no_feasible_hashes = set()
    for state_hash, item in sorted(unique.items()):
        state = item.state; generation = generate_candidates(state, config)
        unique_manifest.append({
            "unique_state_sha256": state_hash, "representative_protocol_snapshot_id": item.instance_id,
            "step": state.step, "map_revision": state.map_revision, "knowledge_mode": state.knowledge_mode,
            "planning_graph_sha256": state.planning_graph.graph_sha256,
        })
        candidate_rows.append({
            "unique_state_sha256": state_hash, "search_candidate_count": len(generation.search_candidates),
            "standby_candidate_count": len(generation.standby_candidates),
            "candidate_count_by_agent": json.dumps(generation.search_count_by_agent, sort_keys=True),
            "unreachable_search_count": generation.unreachable_search_count,
            "unreachable_standby_count": generation.unreachable_standby_count,
            "generation_notes": ";".join(generation.reasons),
        })
        if any(generation.search_count_by_agent.get(agent_id, 0) == 0 for agent_id in state.searcher_ids) or not generation.standby_candidates:
            no_feasible_hashes.add(state_hash); continue
        context = build_objective_context(state, generation.search_candidates, generation.standby_candidates, config)
        started = time.perf_counter(); exact = solve_joint_exact(generation.search_candidates, generation.standby_candidates, context, combination_limit=config["solver"]["exact_combination_limit"]); exact_time = time.perf_counter() - started
        started = time.perf_counter(); greedy = solve_joint_greedy(generation.search_candidates, generation.standby_candidates, context); greedy_time = time.perf_counter() - started
        started = time.perf_counter(); lazy = solve_joint_lazy(generation.search_candidates, generation.standby_candidates, context); lazy_time = time.perf_counter() - started
        checks = validate_small_instance(generation.search_candidates, generation.standby_candidates, context)
        fixed_ratios = []
        for standby in generation.standby_candidates:
            fixed_exact = solve_fixed_standby_exact(generation.search_candidates, standby, context)
            fixed_greedy = solve_fixed_standby_greedy(generation.search_candidates, standby, context)
            fixed_ratios.append(1.0 if fixed_exact.objective <= 1e-12 else fixed_greedy.objective / fixed_exact.objective)
        checks.update(minimum_fixed_y_ratio=min(fixed_ratios), fixed_y_bound_pass=min(fixed_ratios) >= 0.5 - 1e-9)
        properties.append({"unique_state_sha256": state_hash, **checks})
        search_only = solve_search_only_greedy(generation.search_candidates, context)
        search_standby = _best_standby(search_only.selected, generation.standby_candidates, context)
        search_joint_result = type(search_only)(
            search_only.solver, search_only.selected, search_standby,
            evaluate_objective(search_only.selected, search_standby, context),
            search_only.combination_count, search_only.status,
        )
        pse = evaluate_legacy_pse_snapshot(state, config)
        pse_context = None
        if pse.status == "OK" and pse.standby is not None:
            pse_context = build_objective_context(state, pse.selected, (pse.standby,), config)
        methods = [
            _method_metrics("exact", exact, context),
            _method_metrics("bser_standard_greedy", greedy, context),
            _method_metrics("bser_lazy_greedy", lazy, context),
            _method_metrics("search_only_under_joint", search_joint_result, context, own_objective=search_only.objective),
            _random_metrics(
                generation.search_candidates, generation.standby_candidates, context,
                config["solver"]["random_repetitions"],
                config["solver"]["random_seed"] + int(state_hash[:8], 16),
            ),
            _method_metrics("pse_under_joint", pse if pse_context is not None else None, pse_context or context),
        ]
        for method in methods: method_rows.append({"unique_state_sha256": state_hash, **method})
        by_method = {row["method"]: row for row in methods}
        greedy_diag = by_method["bser_standard_greedy"]
        search_diag = by_method["search_only_under_joint"]
        exact_ratio = 1.0 if exact.objective <= 1e-12 else greedy.objective / exact.objective
        result = {
            "unique_state_sha256": state_hash, "representative_protocol_snapshot_id": item.instance_id,
            "profile": item.profile, "scenario_id": item.scenario_id, "scenario_seed": item.scenario_seed,
            "step": state.step, "knowledge_mode": state.knowledge_mode,
            "exact_joint_objective": exact.objective, "greedy_joint_objective": greedy.objective,
            "lazy_joint_objective": lazy.objective, "greedy_exact_ratio": exact_ratio,
            "search_only_own_objective": search_only.objective,
            "search_only_joint_objective": search_diag["joint_objective"],
            "greedy_detection_probability": greedy_diag["detection_probability"],
            "search_only_detection_probability": search_diag["detection_probability"],
            "detection_probability_difference": greedy_diag["detection_probability"] - search_diag["detection_probability"],
            "detection_probability_retention": greedy_diag["detection_probability"] / max(search_diag["detection_probability"], 1e-12),
            "greedy_conditional_response_time": greedy_diag["conditional_reachable_response_time"],
            "search_only_conditional_response_time": search_diag["conditional_reachable_response_time"],
            "response_time_difference": "" if greedy_diag["conditional_reachable_response_time"] == "" or search_diag["conditional_reachable_response_time"] == "" else greedy_diag["conditional_reachable_response_time"] - search_diag["conditional_reachable_response_time"],
            "greedy_unreachable_detected_mass": greedy_diag["unreachable_detected_mass"],
            "search_only_unreachable_detected_mass": search_diag["unreachable_detected_mass"],
            "unreachable_detected_mass_difference": greedy_diag["unreachable_detected_mass"] - search_diag["unreachable_detected_mass"],
            "joint_objective_difference": greedy.objective - search_diag["joint_objective"],
            "allocation_candidate_id_jaccard": _allocation_jaccard(greedy, search_only),
            "allocation_waypoint_overlap_count": len(set(candidate.waypoint for candidate in greedy.selected) & set(candidate.waypoint for candidate in search_only.selected)),
            "selected_candidate_ids": ";".join(greedy.selected_ids),
            "selected_waypoints": json.dumps([candidate.waypoint for candidate in greedy.selected]),
            "standby_candidate_id": greedy.standby.candidate_id,
            "standby_waypoint": json.dumps(greedy.standby.waypoint),
            "coverage_overlap": coverage_overlap(greedy.selected, context),
            "exact_runtime_seconds": exact_time, "greedy_runtime_seconds": greedy_time,
            "lazy_runtime_seconds": lazy_time, "exact_combination_count": exact.combination_count,
        }
        per_unique.append(result); results_by_hash[state_hash] = result
        equivalence_refs.append({
            "unique_state_sha256": state_hash, "planning_graph_sha256": state.planning_graph.graph_sha256,
            "executor_path_tree_sha256": context.response_time_by_id and hashlib.sha256(
                np.asarray(next(iter(context.response_time_by_id.values()))).tobytes(order="C")
            ).hexdigest(),
        })
        failed = [key for key, value in checks.items() if key.endswith("_pass") and not value]
        if failed: failures.append({"unique_state_sha256": state_hash, "failure": ";".join(failed)})

    for row in protocol:
        if row.get("unique_state_sha256") in no_feasible_hashes:
            row["status"]="SKIPPED"; row["skip_reason"]="SKIPPED_NO_FEASIBLE_CANDIDATES"; skipped.append(dict(row))
    protocol_results = []
    metadata_by_id = {row["protocol_snapshot_id"]: row for row in protocol}
    for item in mapping:
        result = results_by_hash.get(item["unique_state_sha256"])
        if result is not None:
            protocol_results.append({**metadata_by_id[item["protocol_snapshot_id"]], **result})

    scalability = []
    selected_scaling = sorted(unique.items())[:(1 if smoke else int(config["e1_v2"]["scalability_unique_state_count"]))]
    for state_hash, item in selected_scaling:
        for search_k in ([4] if smoke else config["candidate_generation"]["search_scaling"]):
            for standby_k in ([4] if smoke else config["candidate_generation"]["standby_scaling"]):
                generation = generate_candidates(item.state, config, k_search=search_k, k_standby=standby_k)
                if not generation.search_candidates or not generation.standby_candidates: continue
                context = build_objective_context(item.state, generation.search_candidates, generation.standby_candidates, config)
                started=time.perf_counter(); greedy=solve_joint_greedy(generation.search_candidates,generation.standby_candidates,context); gt=time.perf_counter()-started
                started=time.perf_counter(); lazy=solve_joint_lazy(generation.search_candidates,generation.standby_candidates,context); lt=time.perf_counter()-started
                started=time.perf_counter(); exact=solve_joint_exact(generation.search_candidates,generation.standby_candidates,context,combination_limit=config["solver"]["exact_combination_limit"]); et=time.perf_counter()-started
                scalability.append({"unique_state_sha256":state_hash,"k_search":search_k,"k_standby":standby_k,"actual_search_candidates":len(generation.search_candidates),"actual_standby_candidates":len(generation.standby_candidates),"greedy_runtime_seconds":gt,"lazy_runtime_seconds":lt,"exact_runtime_seconds":et if exact.status=="OK" else "","exact_status":exact.status,"exact_combination_count":exact.combination_count,"greedy_objective":greedy.objective,"lazy_objective":lazy.objective})

    _write_csv(output / "protocol_snapshot_manifest.csv", protocol); _write_json(output / "protocol_snapshot_manifest.json", protocol)
    _write_csv(output / "unique_planning_state_manifest.csv", unique_manifest); _write_json(output / "unique_planning_state_manifest.json", unique_manifest)
    _write_csv(output / "protocol_to_unique_state_map.csv", mapping)
    _write_csv(output / "candidate_statistics.csv", candidate_rows)
    _write_csv(output / "per_unique_state_results.csv", per_unique)
    _write_csv(output / "per_protocol_snapshot_results.csv", protocol_results)
    _write_csv(output / "property_check_results.csv", properties)
    _write_csv(output / "planner_equivalence_reference.csv", equivalence_refs)
    _write_csv(output / "solver_scalability.csv", scalability)
    _write_csv(output / "joint_objective_comparison.csv", [{k:v for k,v in row.items() if k in {"unique_state_sha256","method","joint_objective","own_scale_objective","status"}} for row in method_rows])
    _write_csv(output / "detection_probability_comparison.csv", [{k:v for k,v in row.items() if k in {"unique_state_sha256","method","detection_probability","status"}} for row in method_rows])
    _write_csv(output / "response_diagnostics_comparison.csv", method_rows)
    _write_csv(output / "pareto_tradeoff.csv", [{"unique_state_sha256":row["unique_state_sha256"],"method":row["method"],"detection_probability":row.get("detection_probability",""),"conditional_reachable_response_time":row.get("conditional_reachable_response_time",""),"unreachable_detected_mass":row.get("unreachable_detected_mass",""),"joint_objective":row.get("joint_objective","")} for row in method_rows])
    _write_csv(output / "aggregate_joint_objective_by_method.csv", _aggregate_methods(method_rows, "joint_objective"))
    _write_csv(output / "aggregate_detection_probability_by_method.csv", _aggregate_methods(method_rows, "detection_probability"))
    _write_csv(output / "aggregate_by_profile.csv", _aggregate(protocol_results, ["profile"]))
    _write_csv(output / "aggregate_by_snapshot_step.csv", _aggregate(protocol_results, ["requested_snapshot_step"]))
    _write_csv(output / "aggregate_by_trajectory.csv", _aggregate(protocol_results, ["profile","scenario_id","scenario_seed","action_trace"]))
    scenario_aggregate = _aggregate(protocol_results, ["profile","scenario_id","scenario_seed"])
    _write_csv(output / "aggregate_by_scenario.csv", scenario_aggregate)
    _write_csv(output / "aggregate_by_knowledge_mode.csv", _aggregate(protocol_results, ["knowledge_mode"]))
    _write_csv(output / "failure_cases.csv", failures)
    _write_csv(output / "skipped_protocol_snapshots.csv", skipped)

    property_failures = sum(any(not value for key,value in row.items() if key.endswith("_pass")) for row in properties)
    summary_core = {
        "schema":"bser.e1_v2.summary.v1", "formal_training_run":False,
        "protocol_request_count":len(protocol), "valid_protocol_snapshot_count":sum(row["status"]=="VALID" for row in protocol),
        "skipped_protocol_snapshot_count":len(skipped), "raw_unique_planning_state_count":len(unique),
        "solved_unique_planning_state_count":len(per_unique), "trajectory_aggregate_count":len(_aggregate(protocol_results,["profile","scenario_id","scenario_seed","action_trace"])),
        "scenario_primary_sample_count":len(scenario_aggregate), "property_failure_count":property_failures,
        "failure_case_count":len(failures), "primary_statistical_unit":"profile x scenario_seed",
        "passed":len(protocol)==expected and len(per_unique)>0 and not failures and property_failures==0,
    }
    summary = {**summary_core,"deterministic_summary_sha256":_stable_hash(summary_core)}
    _write_json(output / "e1_v2_summary.json", summary)
    _write_json(output / "determinism_manifest.json", {"schema":"bser.e1_v2.determinism.v1","summary_sha256":summary["deterministic_summary_sha256"],"unique_state_hashes":sorted(unique),"result_hashes":{row["unique_state_sha256"]:_stable_hash({k:v for k,v in row.items() if not k.endswith("runtime_seconds")}) for row in per_unique}})
    _write_json(output / "experiment_manifest.json", {"schema":"bser.e1_v2.experiment.v1","profiles":list(sorted({row["profile"] for row in protocol})),"snapshot_steps":[0,10,25,50],"environment_max_steps":400,"protocol_request_count":len(protocol),"deduplicated":True,"formal_training":False,"smoke":bool(smoke)})
    (output / "e1_v2_summary.md").write_text(f"# BSER E1-v2 summary\n\nStatus: **{'PASS' if summary['passed'] else 'FAIL'}**\n\nProtocol requests: {len(protocol)}; raw unique states: {len(unique)}; solved unique states: {len(per_unique)}; scenario-level primary samples: {len(scenario_aggregate)}.\n\nNo formal training was run.\n",encoding="utf-8")
    (output / "test_report.md").write_text(f"# E1-v2 test report\n\nStatus: **{'PASS' if summary['passed'] else 'FAIL'}**\n\nProperties are evaluated once per unique planning state; skips and failures are retained explicitly.\n",encoding="utf-8")
    if not smoke and per_unique:
        m20=[row for row in per_unique if row["profile"]=="M20_MOVING_UNKNOWN_MULTI"]
        pool=m20 or per_unique; median=statistics.median(row["greedy_exact_ratio"] for row in pool); typical=min(pool,key=lambda row:abs(row["greedy_exact_ratio"]-median))
        item=unique[typical["unique_state_sha256"]]
        _write_json(output/"typical_case.json",{"selection_rule":"M20 unique state nearest median greedy/exact ratio","unique_state_sha256":typical["unique_state_sha256"],"representative_protocol_snapshot_id":typical["representative_protocol_snapshot_id"],"cell_centers":item.state.grid.cell_centers.tolist(),"belief":item.state.target_belief.probabilities.tolist(),"occupancy_probability":item.state.occupancy.occupancy_probability.tolist(),"selected_waypoints":json.loads(typical["selected_waypoints"]),"standby_waypoint":json.loads(typical["standby_waypoint"])})
        from chapter3_bser.experiments.plot_e1_v2 import generate_figures
        generate_figures(output)
    if run_sensitivity and per_unique and not smoke:
        from chapter3_bser.experiments.run_s1_sensitivity import run as run_s1
        run_s1([item for key,item in sorted(unique.items()) if item.profile == "M20_MOVING_UNKNOWN_MULTI"], smoke=smoke)
    return summary


def _aggregate_methods(rows, field):
    output=[]
    for method in sorted({row["method"] for row in rows}):
        values=[float(row[field]) for row in rows if row["method"]==method and row.get(field,"")!=""]
        output.append({"method":method,"unique_state_count":len(values),f"mean_{field}":statistics.fmean(values) if values else ""})
    return output


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT); parser.add_argument("--smoke",action="store_true"); parser.add_argument("--skip-sensitivity",action="store_true"); args=parser.parse_args()
    summary=run(args.output_dir,smoke=args.smoke,run_sensitivity=not args.skip_sensitivity); print(json.dumps(summary,sort_keys=True)); raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__": main()
