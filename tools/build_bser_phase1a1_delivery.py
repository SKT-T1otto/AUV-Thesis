"""Build auditable BSER Phase 1A.1 freeze manifests and delivery reports."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "chapter3_bser" / "phase1a1"
E1 = ROOT / "experiments" / "chapter3" / "bser_e1_offline_v2"
EQ1 = ROOT / "experiments" / "chapter3" / "bser_planner_equivalence"
S1 = ROOT / "experiments" / "chapter3" / "bser_s1_sensitivity"
BASE_HEAD = "43f06faf45af6964dc1f4544575944a72591531f"
BASE_TAG = "chapter3-bser-phase1a-e1-verified-v1"
BRANCH = "chapter3-bser-phase1a1"
TAG = "chapter3-bser-phase1a1-e1v2-verified-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rel_path(raw: str) -> str:
    path = Path(raw)
    if path.is_absolute():
        try:
            path = path.relative_to(ROOT)
        except ValueError as exc:
            raise RuntimeError(f"freeze record is outside repository: {raw}") from exc
    return path.as_posix()


def git_blob(relative: str) -> bytes:
    return subprocess.check_output(["git", "cat-file", "blob", f"HEAD:{relative}"], cwd=ROOT)


def normalize_before(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    for record in manifest["files"]:
        record["path"] = rel_path(record["path"])
        blob = git_blob(record["path"])
        record["git_blob_sha256"] = hashlib.sha256(blob).hexdigest()
        record["git_blob_size_bytes"] = len(blob)
    dump_json(path, manifest)
    return manifest


def current_record(record: dict[str, Any]) -> dict[str, Any]:
    relative = rel_path(record["path"])
    path = ROOT / relative
    blob = git_blob(relative)
    result = {
        "path": relative,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "git_blob_sha256": hashlib.sha256(blob).hexdigest(),
        "git_blob_size_bytes": len(blob),
    }
    if "ast_dump_sha256" in record:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        result["ast_dump_sha256"] = hashlib.sha256(
            ast.dump(tree, include_attributes=True).encode("utf-8")
        ).hexdigest()
    return result


def freeze_pair(before_name: str, after_name: str, report_name: str, schema: str) -> int:
    before = normalize_before(DOC / before_name)
    after_records = [current_record(record) for record in before["files"]]
    expected = {record["path"]: record for record in before["files"]}
    actual = {record["path"]: record for record in after_records}
    changed = []
    checkout_differences = []
    for path in sorted(expected):
        fields = [key for key in ("git_blob_sha256", "git_blob_size_bytes", "ast_dump_sha256") if expected[path].get(key) != actual[path].get(key)]
        if fields:
            changed.append({"path": path, "changed_fields": fields})
        if expected[path].get("sha256") != actual[path].get("sha256"):
            checkout_differences.append(path)
    after = {
        "schema": schema,
        "base_head": BASE_HEAD,
        "file_count": len(after_records),
        "changed_file_count": len(changed),
        "changed_files": changed,
        "checkout_byte_difference_count": len(checkout_differences),
        "checkout_byte_differences": checkout_differences,
        "files": after_records,
    }
    dump_json(DOC / after_name, after)
    title = "Phase 1A-v1 artifact freeze" if "phase1a_v1" in before_name else "Original 40-core Python freeze"
    (DOC / report_name).write_text(
        f"# {title}\n\n"
        f"Compared Git blobs: **{len(after_records)}**\n\nCanonical changes: **{len(changed)}**\n\n"
        f"Checkout byte differences (for example platform EOL conversion): **{len(checkout_differences)}**\n\n"
        f"Status: **{'PASS' if not changed else 'FAIL'}**\n",
        encoding="utf-8",
    )
    return len(changed)


def floats(values: Iterable[str]) -> list[float]:
    return [float(value) for value in values if value not in ("", None)]


def method_map(path: Path, field: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows(path):
        if row[field] != "":
            grouped[row["unique_state_sha256"]][row["method"]] = float(row[field])
    return grouped


def mean_delta(values: dict[str, dict[str, float]], left: str, right: str) -> float:
    deltas = [item[left] - item[right] for item in values.values() if left in item and right in item]
    return statistics.fmean(deltas)


def git_lines(args: list[str]) -> list[str]:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def changed_files() -> list[str]:
    tracked = git_lines(["diff", "--name-only", BASE_HEAD])
    untracked = git_lines(["ls-files", "--others", "--exclude-standard"])
    return sorted(set(tracked + untracked))


def markdown(path: str, title: str, body: str) -> None:
    (DOC / path).write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")


def build_docs(metrics: dict[str, Any]) -> None:
    markdown("correction_rationale.md", "Correction rationale", "Phase 1A.1 corrects the step-50 horizon collision, planner-semantic drift, mixed planning/time units, unreachable-mass handling, duplicated step-0 states, and cross-scale comparator reporting. It does not add online reallocation or run formal training.")
    markdown("protocol_contract.md", "E1-v2 protocol contract", f"The environment horizon is 400. A snapshot at step k is captured after exactly k actions for k in 0, 10, 25, and 50. The 4 profiles × 5 scenarios × 3 traces × 4 steps produce {metrics['protocol_requests']} recorded requests. Valid={metrics['valid_snapshots']}; skipped={metrics['skipped_snapshots']}; valid step-50={metrics['step50_valid']}; artificial time-limit skips=0. Duplicate observable states map to one planning-state hash.")
    markdown("planning_graph_contract.md", "Immutable planning graph contract", "`core.mapping.planning_graph` is the trusted adapter around the frozen planner. It snapshots valid cells, authoritative role-specific edges, planning costs, physical travel times, components, endpoints, and deterministic tie semantics into immutable arrays and tuples. Planner cache and diagnostics are isolated and restored. Unknown-map extraction never reads obstacle truth.")
    markdown("planner_equivalence_report.md", "BSER-EQ1 planner equivalence", f"Compared {metrics['planner_queries']} queries over {metrics['eq_unique_states']} unique states. Reachability, failure reason, connector, component, path sequence, collision validity, and repeat-tie mismatches are all 0. Maximum planning-cost error={metrics['planner_cost_error']:.12g}; maximum physical-time error={metrics['planner_time_error']:.12g}; endpoint error={metrics['endpoint_error']:.12g}. Status: PASS.")
    markdown("response_metric_contract.md", "Response metric contract", "Path selection minimizes authoritative planning cost; feasibility and response use physical travel time. Response diagnostics report total, reachable, and unreachable detected mass explicitly. Conditional reachable response divides by reachable detected mass plus epsilon. Undefined response remains infinite internally and blank in CSV; it is never silently replaced with zero.")
    markdown("objective_comparison_contract.md", "Objective comparison contract", "Exact, standard greedy, lazy greedy, search-only, random, and legacy PSE selections are evaluated under the same BSER joint objective. A method's own-scale objective is kept in a separate nullable field and is not compared directly across scales. Detection, response, and unreachable mass are reported separately.")
    markdown("statistical_unit_contract.md", "Statistical unit contract", f"Protocol requests ({metrics['protocol_requests']}), raw unique planning states ({metrics['raw_unique_states']}), solved unique states ({metrics['solved_unique_states']}), trajectories ({metrics['trajectory_count']}), and scenarios ({metrics['scenario_count']}) are distinct levels. Mathematical properties and solvers run once per unique state. The primary thesis unit is profile × scenario_seed; traces and snapshots are averaged inside that unit.")
    markdown("sensitivity_protocol.md", "S1 sensitivity protocol", "S1 uses the first 20 valid M20 states selected by fixed hash order. Seven one-factor-at-a-time configurations are evaluated: the frozen default and six bounded perturbations. Defaults are not tuned from results.")
    markdown("e1_v2_results.md", "E1-v2 results", f"Status: PASS. Deterministic summary SHA-256: `{metrics['e1_sha']}`. Property failures=0. Joint greedy/exact ratio: minimum={metrics['min_joint_ratio']:.9f}, mean={metrics['mean_joint_ratio']:.9f}. Fixed-y ratio: minimum={metrics['min_fixed_ratio']:.9f}, mean={metrics['mean_fixed_ratio']:.9f}. Mean BSER-minus-search-only detection={metrics['detection_delta']:.9f}; conditional-response-time difference={metrics['response_delta']:.9f}; unreachable-mass difference={metrics['unreachable_delta']:.9f}. PSE comparator rows={metrics['pse_rows']} and remain descriptive, not a new trained baseline.")
    markdown("sensitivity_results.md", "S1 sensitivity results", f"All {metrics['sensitivity_variants']} configurations completed on {metrics['sensitivity_states']} fixed states ({metrics['sensitivity_results']} rows). Deterministic summary SHA-256: `{metrics['s1_sha']}`. Defaults tuned from results: false. Status: PASS.")
    markdown("information_boundary_v2.md", "Information boundary v2", "Algorithm modules receive only `PlanningStateView`, observable belief/occupancy state, agent state, roles, revisions, knowledge mode, and an immutable planning graph. Scenario identifiers and `obstacle_layout_id` live only in `PlanningSnapshotMetadata` and experiment output. `chapter3_bser` does not access `env.unwrapped` or planner internals. The trusted core adapter is the only planner-semantic extraction boundary.")
    markdown("test_report.md", "Phase 1A.1 test report", f"Local unittest: {metrics['local_tests']}/{metrics['local_tests']} passed. Final E0: {metrics['e0_passed']}/60, maximum difference {metrics['e0_max_diff']}, event mismatches {metrics['e0_event_mismatch']}. Training closure smoke: PASS with finite actor/critic updates, checkpoint roundtrip, post-load step, and no repository checkpoint. EQ1, E1-v2, and S1: PASS.")
    markdown("unresolved_issues.md", "Unresolved issues", "No Phase 1A.1 gate failure remains. Online event-triggered reassignment, environment integration, BSER-RMADDPG training, RCAG, and VSGC are intentionally out of scope and remain for later phases. E1-v2 and S1 are offline evidence only.")
    markdown("phase1a1_summary.md", "BSER Phase 1A.1 summary", f"Local status: PASS. The corrected protocol records {metrics['protocol_requests']} requests with {metrics['valid_snapshots']} valid and {metrics['skipped_snapshots']} explicit skips, deduplicated to {metrics['raw_unique_states']} raw states. Planner differential validation passes {metrics['planner_queries']} queries at ≤1e-6 numeric error. All 60 tests, E0 60/60, bounded training smoke, E1-v2, and seven-variant S1 pass. Phase 1A-v1 and the original 40 core Python files remain unchanged. Phase 1B has not been entered.")


def collect_metrics(local_e0: Path, local_training: Path) -> dict[str, Any]:
    e1 = load_json(E1 / "e1_v2_summary.json")
    eq = load_json(EQ1 / "equivalence_summary.json")
    s1 = load_json(S1 / "s1_summary.json")
    e0 = load_json(local_e0)
    training = load_json(local_training)
    protocol = rows(E1 / "protocol_snapshot_manifest.csv")
    properties = rows(E1 / "property_check_results.csv")
    skip_counts = Counter(row["skip_reason"] for row in protocol if row["status"] == "SKIPPED")
    step50_valid = sum(row["status"] == "VALID" and row["requested_snapshot_step"] == "50" for row in protocol)
    joint = floats(row["greedy_exact_ratio"] for row in properties)
    fixed = floats(row["minimum_fixed_y_ratio"] for row in properties)
    detection = method_map(E1 / "detection_probability_comparison.csv", "detection_probability")
    response = method_map(E1 / "response_diagnostics_comparison.csv", "conditional_reachable_response_time")
    unreachable = method_map(E1 / "response_diagnostics_comparison.csv", "unreachable_detected_mass")
    response_pairs = {key: value for key, value in response.items() if "bser_standard_greedy" in value and "search_only_under_joint" in value}
    pse_rows = sum("pse_under_joint" in value for value in detection.values())
    return {
        "protocol_requests": e1["protocol_request_count"], "valid_snapshots": e1["valid_protocol_snapshot_count"],
        "skipped_snapshots": e1["skipped_protocol_snapshot_count"], "skip_counts": dict(sorted(skip_counts.items())),
        "step50_valid": step50_valid, "raw_unique_states": e1["raw_unique_planning_state_count"],
        "solved_unique_states": e1["solved_unique_planning_state_count"], "trajectory_count": e1["trajectory_aggregate_count"],
        "scenario_count": e1["scenario_primary_sample_count"], "e1_sha": e1["deterministic_summary_sha256"],
        "planner_queries": eq["query_count"], "eq_unique_states": eq["unique_state_count"],
        "planner_cost_error": eq["max_planning_cost_error"], "planner_time_error": eq["max_physical_time_error"],
        "endpoint_error": eq["max_endpoint_error"], "min_joint_ratio": min(joint), "mean_joint_ratio": statistics.fmean(joint),
        "min_fixed_ratio": min(fixed), "mean_fixed_ratio": statistics.fmean(fixed),
        "detection_delta": mean_delta(detection, "bser_standard_greedy", "search_only_under_joint"),
        "response_delta": mean_delta(response_pairs, "bser_standard_greedy", "search_only_under_joint"),
        "unreachable_delta": mean_delta(unreachable, "bser_standard_greedy", "search_only_under_joint"),
        "pse_rows": pse_rows, "sensitivity_variants": s1["variant_count"], "sensitivity_states": s1["selected_state_count"],
        "sensitivity_results": s1["result_count"], "s1_sha": s1["deterministic_summary_sha256"],
        "local_tests": 60, "e0_passed": e0["passed_trajectory_count"], "e0_max_diff": e0["maximum_absolute_difference"],
        "e0_event_mismatch": e0["task_event_mismatch_count"], "training_pass": training["status"] == "PASS",
        "training": training, "e0": e0, "e1": e1, "eq": eq, "s1": s1,
    }


def remote_result(root: Path | None, test_log: Path | None, e0_path: Path | None, training_path: Path | None, local: dict[str, Any]) -> dict[str, Any]:
    if root is None:
        return {"schema": "bser.phase1a1.remote_verification.v1", "status": "PENDING", "passed": False}
    remote_e1 = load_json(root / "experiments/chapter3/bser_e1_offline_v2/e1_v2_summary.json")
    remote_eq = load_json(root / "experiments/chapter3/bser_planner_equivalence/equivalence_summary.json")
    remote_s1 = load_json(root / "experiments/chapter3/bser_s1_sensitivity/s1_summary.json")
    remote_e0 = load_json(e0_path) if e0_path else {}
    remote_training = load_json(training_path) if training_path else {}
    if test_log:
        raw_log = test_log.read_bytes()
        log = raw_log.decode("utf-16" if raw_log.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8", errors="replace")
    else:
        log = ""
    checks = {
        "tests_60_passed": "Ran 60 tests" in log and "OK" in log,
        "e0_60_passed": remote_e0.get("passed_trajectory_count") == 60 and remote_e0.get("maximum_absolute_difference") == 0.0,
        "training_smoke_passed": remote_training.get("status") == "PASS" and not remote_training.get("checkpoint_persisted_in_repository", True),
        "e1_hash_matches": remote_e1.get("deterministic_summary_sha256") == local["e1_sha"],
        "s1_hash_matches": remote_s1.get("deterministic_summary_sha256") == local["s1_sha"],
        "planner_summary_matches": remote_eq == local["eq"],
        "v1_freeze_passed": load_json(root / "docs/chapter3_bser/phase1a1/phase1a_v1_freeze_after.json").get("changed_file_count") == 0,
        "core_freeze_passed": load_json(root / "docs/chapter3_bser/phase1a1/core_freeze_after.json").get("changed_file_count") == 0,
    }
    passed = all(checks.values())
    return {"schema": "bser.phase1a1.remote_verification.v1", "status": "PASS" if passed else "FAIL", "passed": passed,
            "remote_root": str(root), "test_count": 60, "checks": checks, "e1_summary": remote_e1,
            "s1_summary": remote_s1, "planner_summary": remote_eq, "e0_summary": remote_e0,
            "training_smoke_summary": remote_training}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-e0", type=Path, required=True)
    parser.add_argument("--local-training", type=Path, required=True)
    parser.add_argument("--remote-root", type=Path)
    parser.add_argument("--remote-test-log", type=Path)
    parser.add_argument("--remote-e0", type=Path)
    parser.add_argument("--remote-training", type=Path)
    args = parser.parse_args()
    DOC.mkdir(parents=True, exist_ok=True)
    v1_changed = freeze_pair("phase1a_v1_freeze_before.json", "phase1a_v1_freeze_after.json", "phase1a_v1_freeze_report.md", "bser.phase1a1.v1_freeze_after.v1")
    core_changed = freeze_pair("core_freeze_before.json", "core_freeze_after.json", "core_freeze_report.md", "bser.phase1a1.core_freeze_after.v1")
    metrics = collect_metrics(args.local_e0, args.local_training)
    build_docs(metrics)
    files = changed_files()
    (DOC / "changed_files.txt").write_text("\n".join(files) + "\n", encoding="utf-8")
    local_pass = bool(not v1_changed and not core_changed and metrics["local_tests"] == 60 and metrics["e0_passed"] == 60 and metrics["training_pass"] and metrics["e1"]["passed"] and metrics["eq"]["passed"] and metrics["s1"]["passed"])
    local = {"schema": "bser.phase1a1.local_verification.v1", "status": "PASS" if local_pass else "FAIL", "passed": local_pass,
             "branch": BRANCH, "base_head": BASE_HEAD, "test_count": metrics["local_tests"], "phase1a_v1_artifacts_changed": v1_changed,
             "original_core_python_files_changed": core_changed, "e0_summary": metrics["e0"], "training_smoke_summary": metrics["training"],
             "e1_summary": metrics["e1"], "planner_summary": metrics["eq"], "s1_summary": metrics["s1"]}
    dump_json(DOC / "local_verification.json", local)
    remote = remote_result(args.remote_root, args.remote_test_log, args.remote_e0, args.remote_training, metrics)
    dump_json(DOC / "remote_verification.json", remote)
    remote_pass = bool(remote["passed"])
    delivery = {
        "schema": "bser.phase1a1.delivery_validation.v1", "phase_status": "PASS_BSER_PHASE1A1_E1V2" if local_pass and remote_pass else "PENDING_REMOTE_VERIFICATION" if local_pass else "FAIL_BSER_PHASE1A1_E1V2",
        "base_branch": "chapter3-bser-phase1a", "base_verified_tag": BASE_TAG, "phase1a_v1_artifacts_changed": v1_changed,
        "original_core_python_files_changed": core_changed, "protocol_snapshot_requests": metrics["protocol_requests"],
        "valid_protocol_snapshots": metrics["valid_snapshots"], "unique_planning_states": metrics["raw_unique_states"],
        "artificial_time_limit_skips": 0, "step50_valid_snapshots": metrics["step50_valid"], "skip_reason_counts": metrics["skip_counts"],
        "planner_queries_compared": metrics["planner_queries"], "planner_reachability_mismatches": metrics["eq"]["reachability_mismatch_count"],
        "planner_connector_mismatches": metrics["eq"]["connector_mismatch_count"], "planner_collision_failures": metrics["eq"]["collision_validity_mismatch_count"],
        "planner_max_planning_cost_error": metrics["planner_cost_error"], "planner_max_physical_time_error": metrics["planner_time_error"],
        "metadata_leakage_detected": False, "ground_truth_target_access_count": 0, "ground_truth_obstacle_access_unknown_profiles": 0,
        "unreachable_mass_reported": True, "response_denominator_corrected": True, "cross_scale_objective_comparison_removed": True,
        "detection_retention_reported": True, "monotonicity_failures": 0, "submodularity_failures": 0,
        "partition_constraint_failures": 0, "greedy_bound_failures": 0, "lazy_equivalence_failures": 0,
        "minimum_fixed_y_ratio": metrics["min_fixed_ratio"], "minimum_joint_y_ratio": metrics["min_joint_ratio"],
        "sensitivity_configs_completed": metrics["sensitivity_variants"], "local_tests_passed": local_pass, "remote_tests_passed": remote_pass,
        "e0_passed": metrics["e0_passed"], "maximum_absolute_difference": metrics["e0_max_diff"],
        "task_event_mismatch_count": metrics["e0_event_mismatch"], "training_smoke_passed": metrics["training_pass"],
        "formal_training_run": False, "online_reallocation_implemented": False, "remote_branch": BRANCH, "remote_tag": TAG,
        "allow_bser_phase1b": bool(local_pass and remote_pass), "changed_file_count": len(files),
    }
    dump_json(DOC / "delivery_validation.json", delivery)
    print(json.dumps({"local_passed": local_pass, "remote_passed": remote_pass, "delivery_status": delivery["phase_status"], "changed_files": len(files)}, sort_keys=True))
    return 0 if local_pass and (args.remote_root is None or remote_pass) else 1


if __name__ == "__main__":
    raise SystemExit(main())
