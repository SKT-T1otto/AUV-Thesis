"""Run the five preregistered Phase 1B.3A diagnosis-only cases."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable
import unittest

from chapter3_bser.diagnostics.event_semantics import Phase1B3ADiagnosticRecorder
from chapter3_bser.experiments.phase1b1_pilot.run_pilot import (
    MemoizedAllocator,
    _episode,
    _write_csv,
    _write_json,
)
from chapter3_bser.online.config import load_phase1b2_config
from core.scenarios.ch3_generator_impl import build_scenario_manifests


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "docs2" / "phase1b3a_diagnosis"
METHOD = "Event-BSER-phase1b2_corrected"
CASES = ((2729, 2), (2731, 1), (2731, 3), (2732, 3), (2733, 0))
BEHAVIOR_FIELDS = (
    "success",
    "found_step",
    "executor_arrival_step",
    "completion_step",
    "collision_count",
    "executor_invalid_count",
    "waypoint_stale_count",
    "optimizer_invocation_count",
    "accepted_replan_count",
    "rejected_replan_count",
    "waypoint_switch_count",
    "total_switch_distance",
    "path_tracking_error",
    "event_counts",
    "reject_reason_counts",
    "target_route_source_counts",
)
FLOAT_FIELDS = frozenset({"path_tracking_error", "total_switch_distance"})
JSON_FIELDS = frozenset({"event_counts", "reject_reason_counts", "target_route_source_counts"})
SPECIALIZED_TEST_MODULES = (
    "tests.test_phase1b3a_reason_partition",
    "tests.test_phase1b3a_behavior_preservation",
    "tests.test_phase1b3a_duplicate_event_keys",
    "tests.test_phase1b3a_public_information_only",
    "tests.test_phase1b3a_case_manifest",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _optional_number(value: Any) -> Any:
    if value in (None, ""):
        return ""
    return int(value)


def _behavior_equal(field: str, expected: Any, actual: Any) -> bool:
    if field == "success":
        return _as_bool(expected) == _as_bool(actual)
    if field in FLOAT_FIELDS:
        return math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=1e-9)
    if field in JSON_FIELDS:
        return json.loads(str(expected)) == json.loads(str(actual))
    return _optional_number(expected) == _optional_number(actual)


def _reason_summary(rows: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["event_type"] == event_type]
    total = len(selected)
    primary = Counter(row["primary_classification"] for row in selected)
    flags = Counter(flag for row in selected for flag in json.loads(row.get("reason_flags", "[]")))
    output = [
        {
            "event_type": event_type,
            "reason_kind": "primary_classification",
            "reason": reason,
            "count": count,
            "share": count / total if total else 0.0,
            "event_instance_total": total,
        }
        for reason, count in sorted(primary.items())
    ]
    output.extend(
        {
            "event_type": event_type,
            "reason_kind": "reason_flag",
            "reason": reason,
            "count": count,
            "share": count / total if total else 0.0,
            "event_instance_total": total,
        }
        for reason, count in sorted(flags.items())
    )
    return output


def _phase_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(
        (row["mission_phase"], row["event_type"], row["primary_classification"])
        for row in rows
    )
    return [
        {
            "mission_phase": phase,
            "event_type": event_type,
            "primary_classification": primary,
            "count": count,
        }
        for (phase, event_type, primary), count in sorted(counts.items())
    ]


def _duplicate_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for event_type in ("EXECUTOR_INVALID", "WAYPOINT_STALE"):
        selected = [row for row in rows if row["event_type"] == event_type]
        # assignment_version restarts at one for each episode, so event keys
        # live in an episode-local namespace when multiple cases are combined.
        unique = len(
            {
                (
                    int(row["scenario_seed"]),
                    int(row["episode_index"]),
                    row["event_key"],
                )
                for row in selected
            }
        )
        repeated = len(selected) - unique
        consecutive = 0
        maximum_run = 0
        without_assignment = 0
        without_map = 0
        without_phase = 0
        streams: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            streams[(int(row["scenario_seed"]), int(row["episode_index"]), int(row["agent_id"]))].append(row)
        for stream in streams.values():
            stream.sort(key=lambda row: int(row["step"]))
            run = 0
            previous = None
            for row in stream:
                if previous is None:
                    run = 1
                else:
                    if int(row["assignment_version"]) == int(previous["assignment_version"]):
                        without_assignment += 1
                    if int(row["map_revision"]) == int(previous["map_revision"]):
                        without_map += 1
                    if row["mission_phase"] == previous["mission_phase"]:
                        without_phase += 1
                    same_consecutive = (
                        row["event_key"] == previous["event_key"]
                        and int(row["step"]) == int(previous["step"]) + 1
                    )
                    if same_consecutive:
                        run += 1
                        consecutive += 1
                    else:
                        run = 1
                maximum_run = max(maximum_run, run)
                previous = row
        output.append(
            {
                "event_type": event_type,
                "total_event_count": len(selected),
                "unique_event_key_count": unique,
                "repeated_same_key_count": repeated,
                "repeated_same_key_share": repeated / len(selected) if selected else 0.0,
                "consecutive_repeat_count": consecutive,
                "maximum_consecutive_run_length": maximum_run,
                "repeated_without_assignment_change_count": without_assignment,
                "repeated_without_map_change_count": without_map,
                "repeated_without_phase_change_count": without_phase,
            }
        )
    return output


def _rejection_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(
        (row["rejection_reason"], row["mission_phase"], row["allocation_scope"])
        for row in rows
    )
    return [
        {
            "rejection_reason": reason,
            "mission_phase": phase,
            "allocation_scope": scope,
            "count": count,
        }
        for (reason, phase, scope), count in sorted(counts.items())
    ]


def _counts(rows: list[dict[str, Any]], event_type: str) -> Counter[str]:
    return Counter(
        row["primary_classification"] for row in rows if row["event_type"] == event_type
    )


def _make_recommendations(
    event_rows: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
    rejection_rows: list[dict[str, Any]],
    target_summary: dict[str, Any],
    behavior_preserved: bool,
) -> dict[str, Any]:
    if not behavior_preserved:
        return {
            "schema": "bser.phase1b3a.decision.v1",
            "status": "WITHHELD_BEHAVIOR_CHANGED",
            "recommendations": [],
        }
    executor = _counts(event_rows, "EXECUTOR_INVALID")
    waypoint = _counts(event_rows, "WAYPOINT_STALE")
    executor_total = sum(executor.values())
    waypoint_total = sum(waypoint.values())
    duplicates = {row["event_type"]: row for row in duplicate_rows}
    search_executor_missing = sum(
        int(row["count"])
        for row in _rejection_summary(rejection_rows)
        if row["rejection_reason"] == "ATOMIC_REJECT_MISSING_LOCAL_CANDIDATES"
        and row["mission_phase"] == "SEARCH"
        and row["allocation_scope"] == "executor_only"
    )
    rules = [
        {
            "rule": 1,
            "triggered": executor["SOFT_RESPONSE_DEGRADED_ONLY"] / executor_total >= 0.50 if executor_total else False,
            "recommendation": "Split EXECUTOR_RESPONSE_DEGRADED from hard route invalidation.",
        },
        {
            "rule": 2,
            "triggered": duplicates["EXECUTOR_INVALID"]["repeated_same_key_share"] >= 0.50,
            "recommendation": "Use assignment_version plus map_revision for executor-event deduplication.",
        },
        {
            "rule": 3,
            "triggered": (
                waypoint["FINAL_WAYPOINT_REACHED"] / waypoint_total >= 0.50
                and duplicates["WAYPOINT_STALE"]["consecutive_repeat_count"] > 0
            ) if waypoint_total else False,
            "recommendation": "Split edge-triggered SEARCH_WAYPOINT_REACHED from WAYPOINT_STALE.",
        },
        {
            "rule": 4,
            "triggered": waypoint["HARD_PATH_INVALID"] > 0,
            "recommendation": "Create a separate SEARCH_WAYPOINT_UNREACHABLE event.",
        },
        {
            "rule": 5,
            "triggered": search_executor_missing > 0,
            "recommendation": "Add replan_executor_standby() without search-candidate dependency.",
        },
        {
            "rule": 6,
            "triggered": int(target_summary["public_target_lock_violation_count"]) > 0,
            "recommendation": "Treat public-target lock violations as a Phase 1B.3B blocker.",
        },
        {
            "rule": 7,
            "triggered": any(
                row["primary_classification"] == "SOFT_RESPONSE_DEGRADED_ONLY"
                and (int(row["scenario_seed"]), int(row["episode_index"])) in {(2731, 1), (2732, 3)}
                and bool(row["current_query_reachable"])
                for row in event_rows
            ),
            "recommendation": "Do not describe reachable soft degradation in successful episodes as executor failure.",
        },
    ]
    return {
        "schema": "bser.phase1b3a.decision.v1",
        "status": "RECOMMENDATIONS_ONLY",
        "recommendations": rules,
        "triggered_rule_numbers": [row["rule"] for row in rules if row["triggered"]],
    }


def _target_summary(recorders: Iterable[Phase1B3ADiagnosticRecorder]) -> dict[str, Any]:
    sources: Counter[str] = Counter()
    received = violations = standby = 0
    for recorder in recorders:
        value = recorder.target_summary()
        received += int(value["public_target_received_count"])
        violations += int(value["public_target_lock_violation_count"])
        standby += int(value["standby_source_after_public_handoff_count"])
        sources.update(value["execute_phase_target_source_counts"])
    return {
        "public_target_received_count": received,
        "execute_phase_target_source_counts": dict(sorted(sources.items())),
        "public_target_lock_violation_count": violations,
        "standby_source_after_public_handoff_count": standby,
    }


def _changed_files() -> list[str]:
    outputs = [
        "baseline_cases.csv", "diagnostic_episode_metrics.csv", "event_step_diagnostics.csv",
        "executor_invalid_reason_summary.csv", "waypoint_reason_summary.csv",
        "duplicate_event_summary.csv", "phase_reason_summary.csv", "rejection_reason_summary.csv",
        "collision_case_trace.csv", "config_snapshot.json", "experiment_manifest.json",
        "delivery_validation.json", "diagnostic_findings.md", "decision_recommendation.json",
        "failure_cases.csv", "test_report.json", "changed_files.txt",
        "specialized_test_report.json", "active_test_report.json",
    ]
    source = [
        "chapter3_bser/controllers/path_tracker.py",
        "chapter3_bser/experiments/phase1b1_pilot/run_pilot.py",
        "chapter3_bser/diagnostics/__init__.py",
        "chapter3_bser/diagnostics/event_semantics.py",
        "chapter3_bser/experiments/phase1b3a_diagnosis/__init__.py",
        "chapter3_bser/experiments/phase1b3a_diagnosis/run_diagnosis.py",
        "tests/test_phase1b3a_reason_partition.py",
        "tests/test_phase1b3a_behavior_preservation.py",
        "tests/test_phase1b3a_duplicate_event_keys.py",
        "tests/test_phase1b3a_public_information_only.py",
        "tests/test_phase1b3a_case_manifest.py",
    ]
    return source + [f"docs2/phase1b3a_diagnosis/{name}" for name in outputs]


def _diagnostic_case_worker(
    job: tuple[int, int, dict[str, Any]],
) -> tuple[dict[str, Any], Phase1B3ADiagnosticRecorder]:
    seed, episode_index, scenario = job
    config = load_phase1b2_config()
    recorder = Phase1B3ADiagnosticRecorder(seed, episode_index, config)
    metric, _ = _episode(
        METHOD,
        scenario,
        episode_index,
        400,
        execution_consistent=True,
        allocator=MemoizedAllocator(),
        diagnostic_recorder=recorder,
    )
    return metric, recorder


def run(output: Path = OUTPUT, workers: int = 4) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    baseline_path = output / "baseline_cases.csv"
    if not baseline_path.is_file():
        raise FileNotFoundError("baseline_cases.csv must be frozen before diagnostic code runs")
    baseline = _read_csv(baseline_path)
    baseline_by_case = {
        (int(row["scenario_seed"]), int(row["episode_index"])): row for row in baseline
    }
    if tuple(baseline_by_case) != CASES:
        raise RuntimeError("frozen baseline case manifest does not exactly match preregistration")
    config = load_phase1b2_config()
    manifest = build_scenario_manifests(
        count=5,
        generator_seed=2729,
        split="validation",
        profiles=("M20_MOVING_UNKNOWN_MULTI",),
    )["M20_MOVING_UNKNOWN_MULTI"]
    scenarios = {int(row["scenario_seed"]): row for row in manifest["scenarios"]}
    metrics: list[dict[str, Any]] = []
    recorders: list[Phase1B3ADiagnosticRecorder] = []
    failures: list[dict[str, Any]] = []
    jobs = [(seed, episode_index, scenarios[seed]) for seed, episode_index in CASES]
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 4, len(jobs)))) as pool:
        futures = {pool.submit(_diagnostic_case_worker, job): job[:2] for job in jobs}
        for future in as_completed(futures):
            seed, episode_index = futures[future]
            try:
                metric, recorder = future.result()
                metrics.append(metric)
                recorders.append(recorder)
                print(
                    json.dumps(
                        {
                            "phase1b3a_case_complete": [seed, episode_index],
                            "runtime_seconds": metric["runtime_seconds"],
                            "status": metric["status"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except Exception as exc:
                failures.append(
                    {
                        "failure_kind": "RUN_FAILURE",
                        "scenario_seed": seed,
                        "episode_index": episode_index,
                        "field": "",
                        "expected": "",
                        "actual": "",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                print(
                    json.dumps(
                        {
                            "phase1b3a_case_failed": [seed, episode_index],
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    metrics.sort(key=lambda row: CASES.index((int(row["scenario_seed"]), int(row["episode_index"]))))
    for metric in metrics:
        case = (int(metric["scenario_seed"]), int(metric["episode_index"]))
        expected = baseline_by_case[case]
        for field in BEHAVIOR_FIELDS:
            if not _behavior_equal(field, expected[field], metric[field]):
                failures.append(
                    {
                        "failure_kind": "BEHAVIOR_MISMATCH",
                        "scenario_seed": case[0],
                        "episode_index": case[1],
                        "field": field,
                        "expected": expected[field],
                        "actual": metric[field],
                        "error_type": "",
                        "message": "diagnostic run changed a frozen behavior field",
                    }
                )
    event_rows = [row for recorder in recorders for row in recorder.event_rows]
    collision_rows = [row for recorder in recorders for row in recorder.collision_rows]
    rejection_rows = [row for recorder in recorders for row in recorder.rejection_rows]
    event_rows.sort(key=lambda row: (int(row["scenario_seed"]), int(row["episode_index"]), int(row["step"]), row["event_type"], int(row["agent_id"])))
    collision_rows.sort(key=lambda row: (int(row["scenario_seed"]), int(row["episode_index"]), int(row["step"]), int(row["agent_id"])))
    executor_summary = _reason_summary(event_rows, "EXECUTOR_INVALID")
    waypoint_summary = _reason_summary(event_rows, "WAYPOINT_STALE")
    duplicate_summary = _duplicate_summary(event_rows)
    phase_summary = _phase_summary(event_rows)
    rejection_summary = _rejection_summary(rejection_rows)
    target_summary = _target_summary(recorders)
    behavior_preserved = not any(row["failure_kind"] == "BEHAVIOR_MISMATCH" for row in failures)
    executor_counts = _counts(event_rows, "EXECUTOR_INVALID")
    waypoint_counts = _counts(event_rows, "WAYPOINT_STALE")
    executor_total = sum(executor_counts.values())
    waypoint_total = sum(waypoint_counts.values())
    duplicates = {row["event_type"]: row for row in duplicate_summary}
    raw_waypoint_step_total = sum(int(row["waypoint_stale_count"]) for row in metrics)
    preregistered_metrics = {
        "executor": {
            "executor_invalid_total": executor_total,
            "installed_assignment_invalid_count": executor_counts["INSTALLED_ASSIGNMENT_INVALID"],
            "hard_route_invalid_count": executor_counts["HARD_ROUTE_INVALID"],
            "soft_response_degraded_only_count": executor_counts["SOFT_RESPONSE_DEGRADED_ONLY"],
            "diagnostic_inconsistency_count": executor_counts["DIAGNOSTIC_INCONSISTENCY"],
            "soft_response_degraded_share": executor_counts["SOFT_RESPONSE_DEGRADED_ONLY"] / executor_total if executor_total else 0.0,
            "hard_route_invalid_share": executor_counts["HARD_ROUTE_INVALID"] / executor_total if executor_total else 0.0,
            "duplicate_executor_event_share": duplicates["EXECUTOR_INVALID"]["repeated_same_key_share"],
            "max_executor_invalid_run_length": duplicates["EXECUTOR_INVALID"]["maximum_consecutive_run_length"],
        },
        "waypoint": {
            "waypoint_stale_total": waypoint_total,
            "raw_boolean_step_event_total": raw_waypoint_step_total,
            "final_waypoint_reached_count": waypoint_counts["FINAL_WAYPOINT_REACHED"],
            "hard_path_invalid_count": waypoint_counts["HARD_PATH_INVALID"],
            "no_active_assignment_count": waypoint_counts["NO_ACTIVE_ASSIGNMENT"],
            "diagnostic_inconsistency_count": waypoint_counts["DIAGNOSTIC_INCONSISTENCY"],
            "final_waypoint_reached_share": waypoint_counts["FINAL_WAYPOINT_REACHED"] / waypoint_total if waypoint_total else 0.0,
            "duplicate_waypoint_event_share": duplicates["WAYPOINT_STALE"]["repeated_same_key_share"],
            "max_waypoint_stale_run_length": duplicates["WAYPOINT_STALE"]["maximum_consecutive_run_length"],
        },
        "execution_target": target_summary,
        "replanning": {
            "optimizer_invocation_count": sum(int(row["optimizer_invocation_count"]) for row in metrics),
            "accepted_replan_count": sum(int(row["accepted_replan_count"]) for row in metrics),
            "actual_decision_rejection_count": sum(int(row["rejected_replan_count"]) for row in metrics),
            "atomic_reject_missing_local_candidates_count": sum(
                int(row["count"]) for row in rejection_summary
                if row["rejection_reason"] == "ATOMIC_REJECT_MISSING_LOCAL_CANDIDATES"
            ),
        },
    }
    recommendations = _make_recommendations(
        event_rows, duplicate_summary, rejection_rows, target_summary, behavior_preserved
    )
    completed = len(metrics)
    coverage_executor = executor_total == sum(int(row["executor_invalid_count"]) for row in metrics)
    coverage_waypoint = waypoint_total >= raw_waypoint_step_total and all(
        row.get("primary_classification") for row in event_rows if row["event_type"] == "WAYPOINT_STALE"
    )
    status = (
        "PENDING_PHASE1B3A_TESTS"
        if completed == 5 and not failures and coverage_executor and coverage_waypoint
        else "FAIL_PHASE1B3A_BEHAVIOR_CHANGED"
        if not behavior_preserved
        else "FAIL_PHASE1B3A_DIAGNOSIS"
    )
    case_collision = [row for row in collision_rows if (int(row["scenario_seed"]), int(row["episode_index"])) == (2731, 3)]
    valid_collision_paths = sum(not bool(row["remaining_path_invalidated"]) for row in case_collision)
    findings = [
        "# BSER Phase 1B.3A diagnostic findings",
        "",
        f"Status: **{status}** (final test gate not yet applied).",
        "",
        f"Completed {completed}/5 preregistered condition-episodes; run or behavior failures: {len(failures)}.",
        f"Frozen behavior preserved: {behavior_preserved}.",
        "",
        "## Executor invalid semantics",
        "",
        f"Classified {executor_total} executor events: installed-invalid {executor_counts['INSTALLED_ASSIGNMENT_INVALID']}, hard-route-invalid {executor_counts['HARD_ROUTE_INVALID']}, soft-response-degraded-only {executor_counts['SOFT_RESPONSE_DEGRADED_ONLY']}, inconsistency {executor_counts['DIAGNOSTIC_INCONSISTENCY']}.",
        "",
        "## Waypoint stale semantics",
        "",
        f"The detector emitted {raw_waypoint_step_total} boolean step-events; per-agent expansion produced {waypoint_total} event instances. Final-waypoint-reached {waypoint_counts['FINAL_WAYPOINT_REACHED']}, hard-path-invalid {waypoint_counts['HARD_PATH_INVALID']}, no-active-assignment {waypoint_counts['NO_ACTIVE_ASSIGNMENT']}, inconsistency {waypoint_counts['DIAGNOSTIC_INCONSISTENCY']}.",
        "",
        "## Public handoff and collision tail",
        "",
        f"Public target received events: {target_summary['public_target_received_count']}; lock violations: {target_summary['public_target_lock_violation_count']}; standby sources after handoff: {target_summary['standby_source_after_public_handoff_count']}.",
        f"Case 2731/3 has {len(case_collision)} collision-agent trace rows; {valid_collision_paths} retain a remaining path with no newly occupied cell and {len(case_collision)-valid_collision_paths} have an invalidated remaining path.",
        "",
        "Recommendations are preregistered-rule outputs only; Phase 1B.3B mechanisms were not implemented.",
    ]
    snapshot = {
        "schema": "bser.phase1b3a.config.v1",
        "phase1b2_corrected": config,
        "diagnostic_policy": {
            "cases": [list(value) for value in CASES],
            "method": METHOD,
            "profile": "M20_MOVING_UNKNOWN_MULTI",
            "max_steps": 400,
            "formal_training": False,
            "oracle_access": False,
            "behavior_float_tolerance": 1e-9,
        },
    }
    protocol = {
        "schema": "bser.phase1b3a.experiment.v1",
        "method": METHOD,
        "cases": [list(value) for value in CASES],
        "profile": "M20_MOVING_UNKNOWN_MULTI",
        "max_steps": 400,
        "state_refresh_interval": 20,
        "planned_condition_episode_count": 5,
        "formal_training": False,
        "oracle_access": False,
        "diagnosis_only": True,
    }
    delivery = {
        "schema": "bser.phase1b3a.delivery.v1",
        "status": status,
        "completed_condition_episode_count": completed,
        "failure_count": len(failures),
        "behavior_preserved": behavior_preserved,
        "executor_reason_coverage_100_percent": coverage_executor,
        "waypoint_reason_coverage_100_percent": coverage_waypoint,
        "tests_finalized": False,
        "preregistered_metrics": preregistered_metrics,
        "phase_reason_counts": phase_summary,
    }
    diagnostic_metrics = []
    by_case_recorder = {(row.scenario_seed, row.episode_index): row for row in recorders}
    for metric in metrics:
        case = (int(metric["scenario_seed"]), int(metric["episode_index"]))
        recorder = by_case_recorder[case]
        executor_case = sum(row["event_type"] == "EXECUTOR_INVALID" for row in recorder.event_rows)
        waypoint_case = sum(row["event_type"] == "WAYPOINT_STALE" for row in recorder.event_rows)
        diagnostic_metrics.append(
            {
                **metric,
                "diagnostic_executor_instance_count": executor_case,
                "diagnostic_waypoint_agent_instance_count": waypoint_case,
                "assignment_version_final": recorder.version.assignment_version,
                "target_diagnostics": json.dumps(recorder.target_summary(), sort_keys=True),
            }
        )
    _write_csv(output / "diagnostic_episode_metrics.csv", diagnostic_metrics)
    _write_csv(output / "event_step_diagnostics.csv", event_rows)
    _write_csv(output / "executor_invalid_reason_summary.csv", executor_summary)
    _write_csv(output / "waypoint_reason_summary.csv", waypoint_summary)
    _write_csv(output / "duplicate_event_summary.csv", duplicate_summary)
    _write_csv(output / "phase_reason_summary.csv", phase_summary)
    _write_csv(output / "rejection_reason_summary.csv", rejection_summary)
    _write_csv(output / "collision_case_trace.csv", collision_rows)
    _write_json(output / "config_snapshot.json", snapshot)
    _write_json(output / "experiment_manifest.json", protocol)
    _write_json(output / "delivery_validation.json", delivery)
    (output / "diagnostic_findings.md").write_text("\n".join(findings) + "\n", encoding="utf-8")
    _write_json(output / "decision_recommendation.json", recommendations)
    _write_csv(
        output / "failure_cases.csv",
        failures,
        fields=["failure_kind", "scenario_seed", "episode_index", "field", "expected", "actual", "error_type", "message"],
    )
    _write_json(
        output / "test_report.json",
        {"schema": "bser.phase1b3a.test_report.v1", "status": "PENDING_TESTS"},
    )
    (output / "changed_files.txt").write_text("\n".join(_changed_files()) + "\n", encoding="utf-8")
    print(json.dumps(delivery, sort_keys=True))
    return delivery


def run_specialized_tests(output: Path = OUTPUT) -> dict[str, Any]:
    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromName(name) for name in SPECIALIZED_TEST_MODULES
    )
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {
        "schema": "bser.phase1b3a.specialized_tests.v1",
        "modules": list(SPECIALIZED_TEST_MODULES),
        "tests_run": int(result.testsRun),
        "failure_count": len(result.failures),
        "error_count": len(result.errors),
        "skipped_count": len(result.skipped),
        "passed_count": int(result.testsRun) - len(result.failures) - len(result.errors) - len(result.skipped),
        "runtime_seconds": time.perf_counter() - started,
        "passed": result.wasSuccessful(),
    }
    _write_json(Path(output) / "specialized_test_report.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return payload


def finalize_tests(output: Path = OUTPUT) -> dict[str, Any]:
    output = Path(output)
    specialized = json.loads((output / "specialized_test_report.json").read_text(encoding="utf-8"))
    active = json.loads((output / "active_test_report.json").read_text(encoding="utf-8"))
    delivery = json.loads((output / "delivery_validation.json").read_text(encoding="utf-8"))
    diagnostic_metrics = _read_csv(output / "diagnostic_episode_metrics.csv")
    target_sources: Counter[str] = Counter()
    target_received = violations = standby = 0
    for row in diagnostic_metrics:
        target = json.loads(row["target_diagnostics"])
        target_sources.update(target["execute_phase_target_source_counts"])
        target_received += int(target["public_target_received_count"])
        violations += int(target["public_target_lock_violation_count"])
        standby += int(target["standby_source_after_public_handoff_count"])
    target_summary = {
        "public_target_received_count": target_received,
        "execute_phase_target_source_counts": dict(sorted(target_sources.items())),
        "public_target_lock_violation_count": violations,
        "standby_source_after_public_handoff_count": standby,
    }
    delivery["preregistered_metrics"]["execution_target"] = target_summary
    recommendations = json.loads((output / "decision_recommendation.json").read_text(encoding="utf-8"))
    for rule in recommendations.get("recommendations", []):
        if int(rule["rule"]) == 6:
            rule["triggered"] = target_summary["public_target_lock_violation_count"] > 0
    recommendations["triggered_rule_numbers"] = [
        int(rule["rule"]) for rule in recommendations.get("recommendations", []) if rule["triggered"]
    ]
    _write_json(output / "decision_recommendation.json", recommendations)
    passed = bool(specialized["passed"] and active["passed"])
    experiment_ready = bool(
        delivery["completed_condition_episode_count"] == 5
        and delivery["failure_count"] == 0
        and delivery["behavior_preserved"]
        and delivery["executor_reason_coverage_100_percent"]
        and delivery["waypoint_reason_coverage_100_percent"]
    )
    status = "PASS_PHASE1B3A_DIAGNOSIS" if passed and experiment_ready else "FAIL_PHASE1B3A_DIAGNOSIS"
    report = {
        "schema": "bser.phase1b3a.test_report.v1",
        "status": status,
        "specialized": specialized,
        "active_suite": active,
        "failure_count": int(specialized["failure_count"]) + int(active["failure_count"]),
        "error_count": int(specialized["error_count"]) + int(active["error_count"]),
    }
    delivery["status"] = status
    delivery["tests_finalized"] = True
    delivery["specialized_tests_passed"] = bool(specialized["passed"])
    delivery["active_tests_passed"] = bool(active["passed"])
    _write_json(output / "test_report.json", report)
    _write_json(output / "delivery_validation.json", delivery)
    findings_path = output / "diagnostic_findings.md"
    findings = findings_path.read_text(encoding="utf-8")
    findings = findings.replace("PENDING_PHASE1B3A_TESTS", status).replace(" (final test gate not yet applied)", "")
    findings_path.write_text(findings, encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run-specialized-tests", action="store_true")
    parser.add_argument("--finalize-tests", action="store_true")
    args = parser.parse_args()
    if args.run_specialized_tests:
        result = run_specialized_tests(args.output_dir)
        raise SystemExit(0 if result["passed"] else 1)
    if args.finalize_tests:
        result = finalize_tests(args.output_dir)
        raise SystemExit(0 if result["status"] == "PASS_PHASE1B3A_DIAGNOSIS" else 1)
    result = run(args.output_dir, workers=args.workers)
    raise SystemExit(0 if result["status"] == "PENDING_PHASE1B3A_TESTS" else 1)


if __name__ == "__main__":
    main()
