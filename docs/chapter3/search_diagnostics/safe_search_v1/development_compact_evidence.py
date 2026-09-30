"""Export small, Git-visible evidence after strict full-stage reconciliation.

Run from the repository root as a namespace module. The source run, complete
analysis and window audit are read-only. Cell IDs, paths and raw trace payloads
are deliberately excluded. This never starts a simulator or loads a checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path, PurePosixPath
import subprocess

from chapter3_bser.experiments.safe_search_v1.analyze_development import (
    PLAN, aggregate as aggregate_episodes, choose_variant, outcome, validate_rows,
)
from chapter3_bser.experiments.safe_search_v1.run_development import collect_completed
from chapter3_bser.experiments.safe_search_v1.run_paired import ROOT, VARIANTS, digest, file_hash, read_json
from .development_figures import prepare
from .development_trace_audit import COVERAGE_FIELDS, check_output, summarize_audits

COVERAGE_SCALARS = (
    "schema", "available", "initial_observed_cells", "unique_observed_cells", "grid_cell_count",
    "unique_observed_grid_fraction", "revisit_age_steps", "pre_found_exposure_steps",
    *COVERAGE_FIELDS,
)
FORBIDDEN_KEYS = {
    "observed_cell_indices", "initial_observed_cell_indices", "per_agent_observed_cell_indices",
    "planned_path", "remaining_path_points", "path_points", "trace", "step_trace", "collision_records",
    "pre_found_collision_snapshots", "ground_truth_obstacles", "obstacles", "positions", "velocities",
}


def count(value, label):
    if type(value) is not int or value < 0:
        raise ValueError("invalid nonnegative count: " + label)
    return value


def validate_terminal(row):
    complete = row["complete_episode_row"]
    result = row["episode_result"]
    steps = row["physical_steps"]
    collision = result.get("first_collision_step")
    if collision is not None and (type(collision) is not int or collision != steps):
        raise ValueError("collision-terminal first collision must equal the actual terminal step")
    checks = dict(found=row["found_within_budget"], success=result["success"], found_step=row["found_step"],
        actual_length=steps, episode_length=steps, terminal_step=steps, termination_reason=row["stop_reason"],
        scenario_id=row["scenario_id"], scenario_seed=row["scenario_seed"],
        environment_innovation_seed=row["environment_innovation_seed"],
        episode_id=row["original_episode_index"], episode_index=row["original_episode_index"],
        evaluation_episode_index=row["original_episode_index"], first_collision_step=collision,
        first_collision_agent_ids=result.get("first_collision_agent_ids", []),
        task_protocol="collision_terminal_v1", episode_terminated=True,
        training_update=False, optimizer_update_count=0)
    for name, expected in checks.items():
        if name not in complete or complete[name] != expected:
            raise ValueError("complete_episode_row disagrees with terminal summary: " + name)
    if any(type(complete[name]) is not bool for name in ("found", "success", "training_update", "episode_terminated")):
        raise ValueError("complete episode flags must be booleans")


def validate_coverage(row):
    coverage = row.get("search_coverage")
    if not isinstance(coverage, dict) or coverage.get("schema") != "ch3.safe_search.target_evidence_coverage.v1":
        raise ValueError("requested coverage summary is missing or has an unknown schema")
    if coverage.get("pre_found_exposure_steps") != row["pre_found_exposure_steps"]:
        raise ValueError("coverage exposure differs from terminal search exposure")
    if coverage.get("available") is not True:
        if row.get("effective_search_steps") is not None:
            raise ValueError("unavailable coverage cannot have a numeric effective-search proxy")
        return False
    values = {name: count(coverage[name], name) for name in COVERAGE_FIELDS}
    initial = count(coverage["initial_observed_cells"], "initial_observed_cells")
    unique = count(coverage["unique_observed_cells"], "unique_observed_cells")
    cells = count(coverage["grid_cell_count"], "grid_cell_count")
    if not cells or initial + values["new_cell_observations"] != unique or not initial <= unique <= cells:
        raise ValueError("initial/new/unique/grid cell accounting differs")
    if not math.isclose(coverage["unique_observed_grid_fraction"], unique / cells, abs_tol=1e-12):
        raise ValueError("unique footprint fraction differs from cell counts")
    for name, expected in (("observed_cell_indices", unique), ("initial_observed_cell_indices", initial)):
        indices = coverage.get(name)
        if (not isinstance(indices, list) or len(indices) != expected or len(set(indices)) != expected
                or any(type(i) is not int or not 0 <= i < cells for i in indices)):
            raise ValueError("raw cell list does not reconcile: " + name)
    if not set(coverage["initial_observed_cell_indices"]) <= set(coverage["observed_cell_indices"]):
        raise ValueError("initial footprint is not a subset of total footprint")
    if (values["aged_revisit_cell_observations"] > values["repeated_cell_observations"]
            or not values["effective_observation_steps"] <= values["observation_update_steps"]
            <= values["target_update_call_steps"] <= row["pre_found_exposure_steps"]
            or values["observation_calls"] < values["target_update_call_steps"]
            or values["effective_observation_steps"] > values["new_cell_observations"] + values["aged_revisit_cell_observations"]
            or row.get("effective_search_steps") != values["effective_observation_steps"]):
        raise ValueError("coverage subset/cadence/effective-step accounting differs")
    overlap = coverage.get("inter_agent_overlap")
    agents = coverage.get("per_agent_coverage")
    if overlap is not None:
        if not isinstance(agents, dict) or set(agents) != {"0", "1", "2"}:
            raise ValueError("attributed coverage must contain exactly three searchers")
        total = count(overlap["total_agent_cell_observations"], "total_agent_cell_observations")
        union = count(overlap["simultaneous_union_cell_observations"], "simultaneous_union_cell_observations")
        redundant = count(overlap["redundant_agent_cell_observations"], "redundant_agent_cell_observations")
        if (union != values["new_cell_observations"] + values["repeated_cell_observations"]
                or total - union != redundant or total < union
                or sum(count(a["cell_observations"], "agent cell observations") for a in agents.values()) != total
                or count(overlap["overlapped_cell_observations"], "overlapped_cell_observations") > union
                or overlap["observation_steps"] != values["observation_update_steps"]):
            raise ValueError("new/repeated/agent/union/redundancy coverage totals disagree")
        expected_fraction = redundant / total if total else None
        if overlap.get("redundant_agent_cell_fraction") != expected_fraction:
            raise ValueError("redundant cell fraction differs from its numerator and denominator")
    return True


def validate_windows(episode, row):
    if (episode["outcome"] != outcome(row) or episode["physical_steps"] != row["physical_steps"]
            or episode["pre_found_exposure_steps"] != row["pre_found_exposure_steps"]
            or episode["found_step"] != row["found_step"]):
        raise ValueError("window audit and terminal episode disagree")
    windows = episode["windows"]
    if len(windows) != 4:
        raise ValueError("exactly four physical-step windows are required")
    for index, window in enumerate(windows):
        first, last = 1 + index * 100, 100 + index * 100
        expected = max(0, min(100, row["pre_found_exposure_steps"] - first + 1))
        if (window["first_step"], window["last_step"], window["pre_found_exposure_steps"]) != (first, last, expected):
            raise ValueError("window limits or Found-censored exposure disagree")
    if sum(w["searcher_stall_agent_steps"] for w in windows) != row["searcher_motion_stall_proxy_agent_steps"]:
        raise ValueError("window stall totals disagree with terminal motion proxy")
    if row["search_coverage"]["available"] is True:
        if not all(w["coverage_available"] for w in windows):
            raise ValueError("window coverage availability differs from terminal summary")
        if any(sum(w[name] for w in windows) != row["search_coverage"][name] for name in COVERAGE_FIELDS):
            raise ValueError("window coverage totals disagree with terminal summary")


def compact_coverage(coverage):
    return {**{name: coverage.get(name) for name in COVERAGE_SCALARS},
            "per_agent_coverage": copy.deepcopy(coverage.get("per_agent_coverage")),
            "inter_agent_overlap": copy.deepcopy(coverage.get("inter_agent_overlap")),
            "errors": list(coverage.get("errors", [])),
            "attribution_errors": list(coverage.get("attribution_errors", []))}


def reject_raw_payload(value):
    if isinstance(value, dict):
        if FORBIDDEN_KEYS & value.keys():
            raise ValueError("raw payload leaked into compact evidence: " + str(sorted(FORBIDDEN_KEYS & value.keys())))
        for nested in value.values():
            reject_raw_payload(nested)
    elif isinstance(value, list):
        if len(value) > 100:
            raise ValueError("oversized array leaked into compact evidence")
        for nested in value:
            reject_raw_payload(nested)


def validate_pairing(rows, analysis, identity, plan):
    indexed = {(row["original_episode_index"], row["variant"]): row for row in rows}
    indices = [scene["source_episode_index"] for scene in plan["splits"]["development"]]
    if set(analysis["paired_vs_v0"]) != set(VARIANTS[1:]):
        raise ValueError("all four paired comparisons are required")
    for variant in VARIANTS[1:]:
        pairs = [(indexed[index, "V0"], indexed[index, variant]) for index in indices]
        comparison = analysis["paired_vs_v0"][variant]
        for name, getter in (
            ("found", lambda r: int(r["found_step"] is not None)),
            ("pre_found_collision", lambda r: int(r["pre_found_collision"])),
            ("success", lambda r: int(r["episode_result"]["success"])),
            ("restricted_found_time_400", lambda r: r["found_step"] if r["found_step"] is not None else 400),
            ("pre_found_exposure_steps", lambda r: r["pre_found_exposure_steps"]),
        ):
            checked = comparison[name]
            expected = sum(getter(b) - getter(a) for a, b in pairs) / 20
            interval = checked.get("bootstrap_95_percentile_ci")
            if (checked.get("n_pairs") != 20 or not math.isclose(checked["mean_delta"], expected, abs_tol=1e-12)
                    or checked.get("bootstrap_replicates") != plan["statistics"]["bootstrap_replicates"]
                    or checked.get("bootstrap_seed") != plan["statistics"]["bootstrap_seed"]
                    or not isinstance(interval, list) or len(interval) != 2
                    or not all(math.isfinite(value) for value in interval) or interval[0] > interval[1]):
                raise ValueError("paired estimate or interval metadata differs from registered comparison")
            if name in ("found", "pre_found_collision", "success"):
                if (checked["discordant_0_to_1"] != sum(getter(a) == 0 and getter(b) == 1 for a, b in pairs)
                        or checked["discordant_1_to_0"] != sum(getter(a) == 1 and getter(b) == 0 for a, b in pairs)):
                    raise ValueError("paired discordant counts disagree")
        expected_pairs = [dict(original_episode_index=a["original_episode_index"], scenario_id=a["scenario_id"],
            v0_outcome=outcome(a), variant_outcome=outcome(b),
            found_delta=int(b["found_step"] is not None)-int(a["found_step"] is not None),
            pre_found_collision_delta=int(b["pre_found_collision"])-int(a["pre_found_collision"])) for a, b in pairs]
        if comparison["scenario_pairs"] != expected_pairs:
            raise ValueError("paired scenario rows disagree with original identities/outcomes")
    gates = analysis["selection"]["gates"]
    if set(gates) != set(VARIANTS[1:]):
        raise ValueError("all four development engineering gates are required")
    declared_d1 = {gate["d1_mechanisms_passed"] for gate in gates.values()}
    if len(declared_d1) != 1 or any(type(value) is not bool for value in declared_d1):
        raise ValueError("inconsistent declared D1 mechanism status")
    declared_d1 = declared_d1.pop()
    possible = [choose_variant(analysis["variants"], d1_passed=declared_d1, wall_time_comparable=False)]
    if identity["workers"] == 1:
        possible.append(choose_variant(analysis["variants"], d1_passed=declared_d1, wall_time_comparable=True))
    if analysis["selection"] not in possible:
        raise ValueError("engineering gates/selection do not follow the registered comparison")


def build(identity, rows, root_rows, analysis, audit, plan, sources):
    if root_rows != rows:
        raise ValueError("root episodes.json differs from the 100 reconstructed child/arm records")
    validate_rows(rows, plan)
    prepare(analysis, audit)
    if identity["sources_before"]["inventory"]["sha256"] != analysis["sources"]["source_inventory_sha256"]:
        raise ValueError("final analysis does not identify the verified runtime source")
    actual = {(row["original_episode_index"], row["variant"]): row for row in rows}
    windows = {(row["original_episode_index"], row["variant"]): row for row in audit["episodes"]}
    diagnostics = {(row["original_episode_index"], row["variant"]): row for row in analysis["episode_diagnostics"]}
    measured = 0
    episodes = []
    for key, row in actual.items():
        validate_terminal(row)
        measured += validate_coverage(row)
        validate_windows(windows[key], row)
        diagnostic = diagnostics[key]
        if (diagnostic["stall_agent_steps"] != row["searcher_motion_stall_proxy_agent_steps"]
                or diagnostic["hold_agent_steps"] != row["searcher_hold_agent_steps"]
                or diagnostic["pre_found_exposure_steps"] != row["pre_found_exposure_steps"]
                or diagnostic.get("search_coverage") != row["search_coverage"]):
            raise ValueError("analysis diagnostic differs from the verified arm")
        episodes.append(dict(original_episode_index=key[0], variant=key[1], scenario_id=row["scenario_id"],
            scenario_seed=row["scenario_seed"], environment_innovation_seed=row["environment_innovation_seed"],
            outcome=outcome(row), found=row["found_within_budget"], found_step=row["found_step"],
            success=row["episode_result"]["success"], physical_steps=row["physical_steps"], stop_reason=row["stop_reason"],
            pre_found_exposure_steps=row["pre_found_exposure_steps"],
            stall_agent_steps=row["searcher_motion_stall_proxy_agent_steps"], hold_agent_steps=row["searcher_hold_agent_steps"],
            first_collision_agent_ids=list(row["episode_result"].get("first_collision_agent_ids", [])),
            first_collision_step=row["episode_result"].get("first_collision_step"),
            physical_signature_sha256=row["signature_sha256"], search_coverage=compact_coverage(row["search_coverage"]),
            trailing_search_steps_without_new_or_aged_evidence=windows[key]["trailing_search_steps_without_new_or_aged_evidence"],
            timeout_final100_without_new_or_aged_evidence=windows[key]["timeout_final100_without_new_or_aged_evidence"]))
    for variant in VARIANTS:
        checked = aggregate_episodes([row for row in rows if row["variant"] == variant], {})
        for name, value in checked.items():
            if name != "pre_found_query_reasons" and value != analysis["variants"][variant][name]:
                raise ValueError("variant aggregate differs from verified arms: " + variant + ":" + name)
    reconstructed = summarize_audits(audit["episodes"])
    for name in ("variants", "paired_v0_v4_both_no_found_timeout"):
        if audit[name] != reconstructed[name]:
            raise ValueError("window aggregate differs from its episode records: " + name)
    validate_pairing(rows, analysis, identity, plan)
    runtime_sources = identity["sources_before"]
    result = dict(schema="ch3.safe_search.development_compact_evidence.v1", complete_stage=True,
        stage="D2", baseline="B0_search_prior", scenario_count=20, episode_count=100, horizon=400,
        training=False, checkpoint_loaded=False, formal_thesis_evaluation=False, performance_passed=None,
        sources=sources,
        experiment_identity=dict(seed=identity["seed"], workers=identity["workers"],
            collection_mode=identity["collection_mode"], selected=identity["selected"],
            planned_episode_runs=100, max_physical_steps=40000, source_input_sha256=identity["input_sha256"],
            checkout_profile=runtime_sources["checkout_profile"], evolution_sha256=runtime_sources["evolution_sha256"],
            source_inventory_sha256=runtime_sources["inventory"]["sha256"],
            source_file_count=len(runtime_sources["inventory"]["files"]),
            historical_record_count=runtime_sources["historical_record_count"],
            historical_production_sha256=runtime_sources["historical_production_sha256"]),
        variants=copy.deepcopy(analysis["variants"]), paired_vs_v0=copy.deepcopy(analysis["paired_vs_v0"]),
        selection=copy.deepcopy(analysis["selection"]), episodes=episodes,
        window_aggregates=copy.deepcopy(audit["variants"]),
        paired_v0_v4_both_no_found_timeout=copy.deepcopy(audit["paired_v0_v4_both_no_found_timeout"]),
        metric_definitions=copy.deepcopy(analysis["metric_definitions"]),
        window_definitions=copy.deepcopy(audit["definitions"]),
        caveats=list(analysis["caveats"]) + list(audit["caveats"]),
        validation=dict(root_rows_equal_reconstructed_100_arms=True, full20_by5_terminal_identity_passed=True,
            collision_terminal_step_reconciled=True, complete_episode_row_reconciled=True,
            coverage_exposure_and_cell_accounting_passed=True, measured_coverage_arm_count=measured,
            unavailable_coverage_arm_count=100-measured, variant_aggregates_reconciled=True,
            window_aggregates_reconciled=True, paired_estimates_and_engineering_gates_reconciled=True,
            raw_payload_excluded=True,
            note="Paired means, discordant pairs and engineering gates are recomputed; declared D1 status is preserved. Bootstrap intervals remain hash-bound to the full analysis and bootstrap is not rerun by this export."))
    reject_raw_payload(result)
    return result


def mapped_run_file(original_path, original_root, run_dir):
    original_path, original_root = str(original_path).replace("\\", "/"), str(original_root).replace("\\", "/").rstrip("/")
    prefix = original_root + "/"
    if not original_path.startswith(prefix):
        raise ValueError("audit hash record is outside its original run root")
    relative = PurePosixPath(original_path[len(prefix):])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("invalid run-relative audit input")
    result = (run_dir / Path(*relative.parts)).resolve()
    if run_dir not in result.parents:
        raise ValueError("audit input escapes the selected run directory")
    return result


def export(analysis_path, audit_path, output_json, run_dir=None):
    analysis_path, audit_path = Path(analysis_path).resolve(), Path(audit_path).resolve()
    analysis, audit = read_json(analysis_path), read_json(audit_path)
    run_dir = Path(run_dir or analysis["sources"]["run_directory"]).resolve()
    output = check_output(output_json, run_dir)
    if ROOT / "docs" not in output.parents:
        raise ValueError("Git-visible compact evidence must be written below repository docs/")
    ignored = subprocess.run(["git", "-c", "safe.directory=" + ROOT.as_posix(), "check-ignore", "--quiet", "--", str(output)],
                             cwd=ROOT, capture_output=True, check=False)
    if ignored.returncode == 0:
        raise ValueError("compact evidence output is ignored by Git")
    if ignored.returncode != 1:
        raise RuntimeError("cannot verify Git visibility of compact evidence output")
    plan = read_json(PLAN)
    identity, rows = collect_completed(run_dir, plan=plan)
    root_rows = read_json(run_dir / "episodes.json")
    hashes = {str(path): file_hash(path) for path in (analysis_path, audit_path, PLAN,
                                                    run_dir / "identity.json", run_dir / "episodes.json")}
    if (analysis["sources"]["identity_sha256"] != hashes[str(run_dir / "identity.json")]
            or analysis["sources"]["episodes_sha256"] != hashes[str(run_dir / "episodes.json")]
            or analysis["sources"]["plan_sha256"] != hashes[str(PLAN)]
            or hashes[str(PLAN)] != identity["sources_before"]["inventory"]["files"][PLAN.relative_to(ROOT).as_posix()]):
        raise ValueError("analysis/plan/root artifact hashes disagree")
    expected_audit_inputs = {run_dir / name for name in ("identity.json", "episodes.json")}
    expected_audit_inputs.update(run_dir / f"scene_{row['original_episode_index']:04d}" /
        f"episode_{row['original_episode_index']:04d}" / row["variant"] / "step_trace.jsonl" for row in rows)
    mapped = {}
    for original, expected_hash in audit["sources"]["input_sha256"].items():
        path = mapped_run_file(original, audit["sources"]["run_directory"], run_dir)
        if path in mapped or file_hash(path) != expected_hash:
            raise ValueError("window-audit input hash changed or duplicated")
        mapped[path] = expected_hash
    if set(mapped) != expected_audit_inputs:
        raise ValueError("audit must bind the root identity, episodes and all 100 traces")
    hashes.update({str(path): sha for path, sha in mapped.items()})
    sources = dict(analysis_path=str(analysis_path), analysis_sha256=hashes[str(analysis_path)],
        window_audit_path=str(audit_path), window_audit_sha256=hashes[str(audit_path)],
        run_directory=str(run_dir), root_identity_sha256=hashes[str(run_dir / "identity.json")],
        root_episodes_sha256=hashes[str(run_dir / "episodes.json")], plan_sha256=hashes[str(PLAN)],
        audit_input_hash_manifest_sha256=digest(audit["sources"]["input_sha256"]),
        verified_trace_hash_count=100, export_script_sha256=file_hash(__file__),
        helper_script_sha256={name: file_hash(Path(__file__).with_name(name))
            for name in ("development_figures.py", "development_trace_audit.py")})
    compact = build(identity, rows, root_rows, analysis, audit, plan, sources)
    if any(file_hash(path) != sha for path, sha in hashes.items()):
        raise RuntimeError("compact evidence input changed during export")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(compact, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return compact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--window-audit", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path,
                        help="Optional relocated original run; otherwise use analysis.sources.run_directory")
    args = parser.parse_args(argv)
    export(args.analysis, args.window_audit, args.output_json, args.run_dir)


if __name__ == "__main__":
    main()
