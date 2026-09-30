"""Offline baseline and replay audit; never starts a simulator or changes inputs.

Run as ``python -B -m docs.chapter3.search_diagnostics.safe_search_v1.
development_baseline_audit --help`` from the repository root.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from chapter3_bser.experiments.safe_search_v1.analyze_development import (
    collision_role, outcome, validate_rows,
)
from chapter3_bser.experiments.safe_search_v1.run_development import (
    BASELINE, INDICES, PLAN, collect_completed,
)
from chapter3_bser.experiments.safe_search_v1.run_paired import (
    ROOT, VARIANTS, digest, file_hash, read_json,
)
from docs.chapter3.search_diagnostics.safe_search_v1 import development_trace_audit as window_audit

CASE = ROOT / "docs/chapter3/search_diagnostics/safe_search_v1/public_map_case_analysis.json"
OUTCOME_FIELDS = ("episode_index", "scenario_id", "scenario_seed",
    "environment_innovation_seed", "found", "found_step", "success", "actual_length",
    "episode_length", "termination_reason", "first_collision_step", "first_collision_agent_ids")


class Inputs:
    """Detect changes while reading, without copying raw data into the report."""
    def __init__(self):
        self.hashes = {}

    def watch(self, path):
        path = Path(path).resolve()
        sha = file_hash(path)
        if str(path) in self.hashes and self.hashes[str(path)] != sha:
            raise RuntimeError(f"input changed while reading: {path}")
        self.hashes[str(path)] = sha
        return sha

    def read(self, path):
        self.watch(path)
        return read_json(path)

    def verify(self):
        if any(file_hash(path) != sha for path, sha in self.hashes.items()):
            raise RuntimeError("audit input or audit code changed while reading")


def key_outcome(row):
    result = {name: row[name] for name in OUTCOME_FIELDS}
    if type(result["found"]) is not bool or type(result["success"]) is not bool:
        raise ValueError("outcome flags must be booleans")
    if result["found"] != (result["found_step"] is not None):
        raise ValueError("Found and Found step disagree")
    if type(result["actual_length"]) is not int or not 1 <= result["actual_length"] <= 400:
        raise ValueError("invalid episode length")
    if result["episode_length"] != result["actual_length"]:
        raise ValueError("episode length fields disagree")
    return result


def compare_outcomes(old, new):
    old, new = key_outcome(old), key_outcome(new)
    differences = [name for name in OUTCOME_FIELDS if old[name] != new[name]]
    return dict(equal=not differences, differing_fields=differences, historical=old, current=new)


def terminal_arm(row):
    steps = row.get("physical_steps")
    return (row.get("terminal") is True and row.get("full_episode_completed") is True
        and row.get("full_episode_requested") is True and type(steps) is int
        and 1 <= steps <= 400 and row.get("stop_reason") in
        ("success", "timeout", "obstacle_collision")
        and (row["stop_reason"] != "timeout" or steps == 400)
        and isinstance(row.get("complete_episode_row"), dict))


def arm_key(row):
    return row["original_episode_index"], row["variant"]


def checked_signature(row):
    signature = row["signature_sha256"]
    if not isinstance(signature, str) or len(signature) != 64 or any(
            c not in "0123456789abcdef" for c in signature):
        raise ValueError("invalid ordered physical-signature digest")
    return signature


def historical_comparison(inputs, historical, run, rows):
    identity = inputs.read(historical / "identity.json")
    summary = inputs.read(historical / "summary.json")
    episodes = inputs.read(historical / "episodes.json")
    if (summary.get("evaluation_complete") is not True or summary.get("n_valid_episodes") != 100
            or summary.get("n_expected_episodes") != 100 or len(episodes) != 100
            or identity.get("baseline") != BASELINE
            or identity.get("sources_before") != identity.get("sources_after")):
        raise ValueError("historical baseline lacks complete, consistent 100-episode evidence")
    indexed = {row["episode_index"]: row for row in episodes}
    if set(indexed) != set(range(100)):
        raise ValueError("historical episodes require unique original indices 0..99")
    old_config_path, old_manifest_path = historical / "resolved_config.json", historical / "evaluation_manifest.json"
    old_config, old_manifest = inputs.read(old_config_path), inputs.read(old_manifest_path)
    config_pairs, manifest_pairs, outcome_pairs = [], [], []
    for row in rows:
        if row["variant"] != "V0":
            continue
        index = row["original_episode_index"]
        child = run / f"scene_{index:04d}"
        config_path, manifest_path = child / "resolved_config.json", child / "evaluation_manifest.json"
        config, manifest, child_identity = (inputs.read(p) for p in
            (config_path, manifest_path, child / "identity.json"))
        config_pairs.append(dict(original_episode_index=index,
            native_runtime_config_json_equal=old_config["native_runtime_config"] == config,
            common_task_conditions_json_equal=old_config["common_task_conditions"] == child_identity["common_task_conditions"],
            current_file_sha256=file_hash(config_path), current_native_config_object_sha256=digest(config)))
        manifest_pairs.append(dict(original_episode_index=index, json_equal=old_manifest == manifest,
            current_file_sha256=file_hash(manifest_path), current_object_sha256=digest(manifest)))
        pair = compare_outcomes(indexed[index], row["complete_episode_row"])
        pair.update(original_episode_index=index,
            historical_scene_sha256=digest(old_manifest["scenarios"][index]),
            current_scene_sha256=digest(manifest["scenarios"][index]))
        pair["scene_content_equal"] = pair["historical_scene_sha256"] == pair["current_scene_sha256"]
        outcome_pairs.append(pair)
    return dict(n_selected_pairs=len(outcome_pairs), added_to_current_denominator=0,
        all_key_outcomes_equal=all(p["equal"] for p in outcome_pairs),
        all_scene_contents_equal=all(p["scene_content_equal"] for p in outcome_pairs),
        all_native_configs_and_task_conditions_equal=all(p["native_runtime_config_json_equal"]
            and p["common_task_conditions_json_equal"] for p in config_pairs),
        all_manifests_json_equal=all(p["json_equal"] for p in manifest_pairs),
        historical_identity_sha256=file_hash(historical / "identity.json"),
        historical_checkout=identity.get("checkout_before"),
        historical_production_source_sha256=identity["sources_before"].get("production_source_sha256"),
        historical_native_config_object_sha256=digest(old_config["native_runtime_config"]),
        historical_config_file_sha256=file_hash(old_config_path),
        historical_manifest_file_sha256=file_hash(old_manifest_path),
        historical_manifest_object_sha256=digest(old_manifest),
        config_comparisons=config_pairs, manifest_comparisons=manifest_pairs, outcome_comparisons=outcome_pairs,
        historical_software_versions=None,
        limitation="Historical Python, NumPy and Torch versions were not recorded; equal selected outcomes do not establish cross-platform stepwise or RNG equivalence. Config file hashes compare a historical wrapper to a native config; use native JSON-object equality.")


def previous_comparison(inputs, previous, current):
    identity = inputs.read(previous / "identity.json")
    failure = inputs.read(previous / "failure.json")
    pairs, excluded = [], []
    seen = set()
    for path in sorted(previous.glob("scene_*/episode_*/V*/summary.json")):
        row = inputs.read(path)
        if not terminal_arm(row):
            excluded.append(dict(path=str(path.relative_to(previous)), reason="not a complete terminal arm"))
            continue
        key = arm_key(row)
        if key in seen or key not in current or row.get("baseline") != BASELINE:
            raise ValueError("duplicate, unexpected or wrong-baseline previous terminal arm")
        expected_path = previous / f"scene_{key[0]:04d}" / f"episode_{key[0]:04d}" / key[1] / "summary.json"
        if path != expected_path:
            raise ValueError("previous terminal arm path and identity disagree")
        seen.add(key)
        new = current[key]
        same_identity = all(row[name] == new[name] for name in
            ("original_episode_index", "scenario_id", "scenario_seed", "environment_innovation_seed", "variant", "baseline"))
        old_sha, new_sha = checked_signature(row), checked_signature(new)
        pairs.append(dict(original_episode_index=key[0], variant=key[1], same_identity=same_identity,
            previous_summary_sha256=file_hash(path), previous_signature_sha256=old_sha,
            current_signature_sha256=new_sha, equal_ordered_signature_digest=old_sha == new_sha,
            equal_key_outcome=key_outcome(row["complete_episode_row"]) == key_outcome(new["complete_episode_row"]),
            physical_steps=row["physical_steps"]))
    if not pairs or identity.get("experiment_complete") is True or failure.get("experiment_complete") is not False:
        raise ValueError("previous directory must contain failed-stage evidence and completed arms")
    return dict(n_complete_terminal_arm_pairs=len(pairs), added_to_current_denominator=0,
        previous_stage_complete=False, previous_failure_type=failure.get("exception_type"),
        previous_identity_sha256=file_hash(previous / "identity.json"),
        previous_failure_sha256=file_hash(previous / "failure.json"),
        all_completed_arm_signatures_equal=all(p["same_identity"] and p["equal_ordered_signature_digest"]
            and p["equal_key_outcome"] for p in pairs), pairs=pairs, excluded_summaries=excluded,
        limitation="Only complete terminal arms are compared. The failed scene10/V3 prefix is not evidence of old full-episode reward or RNG equivalence.")


def telemetry_comparison(inputs, directory, root_identity, current):
    identity, summary = (inputs.read(directory / name) for name in ("identity.json", "summary.json"))
    row = current[10, "V3"]
    if (identity.get("experiment_complete") is not True
            or identity.get("source_and_input_verification_passed") is not True
            or identity.get("sources_before") != root_identity["sources_before"]
            or identity.get("sources_after") != root_identity["sources_before"]
            or summary.get("experiment_complete") is not True
            or summary.get("source_and_input_verification_passed") is not True
            or summary.get("full_task_terminal_both") is not True
            or summary.get("equal_at_every_signature") is not True
            or identity.get("baseline") != BASELINE or identity.get("variant") != "V3"
            or identity.get("original_episode_index") != 10):
        raise ValueError("full telemetry equivalence identity or verification failed")
    selected = next(s for s in root_identity["selected"] if s["original_episode_index"] == 10)
    if identity.get("scenario") != selected:
        raise ValueError("telemetry equivalence scenario differs from the frozen selection")
    signatures, arm_summaries = {}, {}
    for name in ("observed", "unobserved"):
        signatures[name] = inputs.read(directory / name / "signatures.json")
        arm_summaries[name] = inputs.read(directory / name / "summary.json")
        arm = arm_summaries[name]
        if (not terminal_arm(arm) or arm_key(arm) != (10, "V3")
                or len(signatures[name]) != arm["physical_steps"] + 1
                or digest(signatures[name]) != checked_signature(arm)
                or checked_signature(arm) != summary[name + "_signature_sha256"]
                or len(signatures[name]) != summary["signature_count_" + name]):
            raise ValueError("telemetry signatures and full terminal summary disagree")
    return dict(original_episode_index=10, variant="V3", added_to_current_denominator=0,
        observed_and_unobserved_signature_lists_equal=signatures["observed"] == signatures["unobserved"],
        signature_count=len(signatures["observed"]), current_signature_sha256=checked_signature(row),
        observed_signature_sha256=checked_signature(arm_summaries["observed"]),
        unobserved_signature_sha256=checked_signature(arm_summaries["unobserved"]),
        current_matches_both_signatures=all(checked_signature(row) == checked_signature(a) for a in arm_summaries.values()),
        current_matches_both_key_outcomes=all(key_outcome(row["complete_episode_row"])
            == key_outcome(a["complete_episode_row"]) for a in arm_summaries.values()),
        old_failed_public_prefix=summary["old_failed_run_public_prefix"],
        equivalence_identity_sha256=file_hash(directory / "identity.json"),
        equivalence_summary_sha256=file_hash(directory / "summary.json"))


def case_comparison(inputs, path, current):
    case = inputs.read(path)
    scope, equivalence = case["scope"], case["replay_equivalence"]
    if (scope.get("original_episode_index") != 7 or scope.get("variant") != "V4"
            or scope.get("baseline") != BASELINE
            or scope.get("independent_diagnostic_replay_not_additional_evaluation_arm") is not True
            or equivalence.get("included_in_D2_100_arms") is not False
            or equivalence.get("equal_ordered_signature_digest") is not True):
        raise ValueError("unexpected public-map replay case or equivalence status")
    for source, sha in case["input_sha256"].items():
        if inputs.watch(ROOT / source) != sha:
            raise ValueError("public-map case input no longer matches its recorded hash")
    row = current[7, "V4"]
    return dict(original_episode_index=7, variant="V4", added_to_current_denominator=0,
        case_file_sha256=file_hash(path), current_signature_sha256=checked_signature(row),
        reference_signature_sha256=equivalence["reference_signature_sha256"],
        replay_signature_sha256=equivalence["replay_signature_sha256"],
        current_matches_reference_and_replay=(checked_signature(row) == equivalence["reference_signature_sha256"]
            == equivalence["replay_signature_sha256"]),
        counts_match=(equivalence["observed_transition_count"] == row["physical_steps"]
            and equivalence["compared_signature_count"] == row["physical_steps"] + 1),
        physical_steps=row["physical_steps"])


def mechanism_trace(rows):
    """Count query-bearing decisions, not persisted terminal decision labels."""
    reasons, samples = Counter(), Counter()
    atomic_steps = atomic_unchanged = labelled_without_queries = 0
    last_goal_change = {str(agent): 0 for agent in range(3)}
    final100_distance = {str(agent): 0.0 for agent in range(3)}
    for row in rows:
        if not row["search_transition"]:
            continue
        before = {a["agent_id"]: a for a in row["before"]["agents"]}
        after = {a["agent_id"]: a for a in row["after"]["agents"]}
        for agent in range(3):
            if before[agent]["semantic_waypoint"] != after[agent]["semantic_waypoint"]:
                last_goal_change[str(agent)] = row["step_after"]
            if 301 <= row["step_after"] <= 400:
                final100_distance[str(agent)] += math.dist(before[agent]["position"], after[agent]["position"])
        query_count = 0
        for query in row["queries"]["counts"]:
            if type(query["count"]) is not int or query["count"] <= 0:
                raise ValueError("query count must be a positive integer")
            reasons[query["reason"]] += query["count"]
            query_count += query["count"]
        for sample in row["queries"]["samples"]:
            samples[sample["reason"], sample.get("start_endpoint_present")] += 1
        if row["after"]["decision_reason"] == "ATOMIC_REJECT_MISSING_SEARCH_ROUTE":
            if query_count:
                atomic_steps += 1
                atomic_unchanged += all(before[a]["semantic_waypoint"] == after[a]["semantic_waypoint"] for a in range(3))
            else:
                labelled_without_queries += 1
    return dict(query_reasons=dict(reasons), endpoint_samples=samples,
        actual_query_bearing_atomic_rejection_steps=atomic_steps,
        atomic_rejection_steps_with_all_searcher_semantic_goals_unchanged=atomic_unchanged,
        atomic_label_steps_without_queries_excluded=labelled_without_queries,
        last_semantic_goal_change_step_by_searcher=last_goal_change,
        final100_motion_distance_by_searcher=final100_distance)


def baseline_mechanisms(inputs, run, rows):
    """Descriptive V0 mechanisms on all 20 selected scenes; no causal estimate."""
    arms = [r for r in rows if r["variant"] == "V0"]
    if (len(arms) != len(INDICES) or {r["original_episode_index"] for r in arms} != set(INDICES)
            or any(not terminal_arm(r) for r in arms)):
        raise ValueError("baseline mechanism audit requires all 20 full terminal V0 arms")
    inputs.watch(window_audit.__file__)
    counts = Counter(outcome(r) for r in arms)
    collisions, file_hashes, timeout_episodes, timeout_windows = [], [], [], []
    all_queries, timeout_queries, timeout_samples, timeout_counts, timeout_reasons = (Counter() for _ in range(5))
    scene0 = None
    for terminal in arms:
        index = terminal["original_episode_index"]
        path = run / f"scene_{index:04d}" / f"episode_{index:04d}" / "V0"
        if inputs.read(path / "summary.json") != terminal:
            raise ValueError("baseline arm differs from the validated terminal summary")
        trace_sha = inputs.watch(path / "step_trace.jsonl")
        trace = list(window_audit.trace_rows(path / "step_trace.jsonl"))
        windows = window_audit.audit_trace(trace, terminal)
        mechanism = mechanism_trace(trace)
        all_queries.update(mechanism["query_reasons"])
        file_hashes.append(dict(original_episode_index=index, terminal_summary_sha256=file_hash(path / "summary.json"),
            trace_sha256=trace_sha))
        collision_step = terminal["episode_result"].get("first_collision_step")
        if collision_step is not None:
            collisions.append(dict(original_episode_index=index, first_collision_step=collision_step,
                first_collision_agent_ids=terminal["episode_result"]["first_collision_agent_ids"],
                role=collision_role(terminal), pre_found=terminal["pre_found_collision"]))
        if outcome(terminal) != "no_found_timeout":
            continue
        controller = terminal["controller"]
        timeout_counts.update(controller.get("allocation_counts", {}))
        timeout_reasons.update(controller.get("allocation_reasons", {}))
        timeout_queries.update(mechanism["query_reasons"])
        timeout_samples.update(mechanism["endpoint_samples"])
        timeout_windows.append(windows)
        episode = dict(original_episode_index=index,
            partial_allocation_attempts=controller.get("allocation_counts", {}).get("partial_allocation_attempts", 0),
            atomic_missing_search_route_rejections=controller.get("allocation_reasons", {}).get("ATOMIC_REJECT_MISSING_SEARCH_ROUTE", 0),
            partial_search_proposals=controller.get("allocation_counts", {}).get("partial_search_proposals", 0),
            candidate_shortage_agent_generation_records=controller.get("allocation_counts", {}).get("candidate_shortage_events", 0),
            query_reasons=mechanism["query_reasons"],
            actual_query_bearing_atomic_rejection_steps=mechanism["actual_query_bearing_atomic_rejection_steps"],
            atomic_rejection_steps_with_all_searcher_semantic_goals_unchanged=mechanism["atomic_rejection_steps_with_all_searcher_semantic_goals_unchanged"],
            atomic_label_steps_without_queries_excluded=mechanism["atomic_label_steps_without_queries_excluded"],
            final100=windows["windows"][-1],
            trailing_steps_without_new_or_aged_evidence=windows["trailing_search_steps_without_new_or_aged_evidence"],
            last_semantic_goal_change_step_by_searcher=mechanism["last_semantic_goal_change_step_by_searcher"],
            final100_motion_distance_by_searcher=mechanism["final100_motion_distance_by_searcher"])
        if episode["atomic_missing_search_route_rejections"] != episode["actual_query_bearing_atomic_rejection_steps"]:
            raise ValueError("controller atomic rejection count differs from query-bearing trace steps")
        timeout_episodes.append(episode)
        if index == 0:
            scene0 = dict(episode, trailing_steps_without_target_update_call=windows["trailing_search_steps_without_target_update_call"],
                trailing_steps_without_nonempty_footprint_update=windows["trailing_search_steps_without_nonempty_footprint_update"],
                unique_observed_cells=terminal["search_coverage"]["unique_observed_cells"],
                grid_cell_count=terminal["search_coverage"]["grid_cell_count"],
                accepted_replan_steps=controller["replan_steps"])
    n = len(arms)
    found, success = sum(r["found_step"] is not None for r in arms), sum(outcome(r) == "success" for r in arms)
    attempts, rejected = timeout_counts["partial_allocation_attempts"], timeout_reasons["ATOMIC_REJECT_MISSING_SEARCH_ROUTE"]
    combined_windows = window_audit.aggregate(timeout_windows)
    return dict(scope="V0 only, complete fixed development selection; diagnostic outcome strata are descriptive",
        n_scenarios=n, found_count=found, found_rate=found / n, success_count=success, success_rate=success / n,
        outcome_partition=dict(counts), first_collisions=collisions,
        pre_found_collision_roles={role: sum(c["pre_found"] and c["role"] == role for c in collisions)
            for role in ("searcher", "executor", "mixed")},
        all_V0_pre_found_query_reasons=dict(all_queries),
        no_found_timeouts=dict(n_scenarios=len(timeout_episodes),
            original_episode_indices=[e["original_episode_index"] for e in timeout_episodes],
            partial_allocation_attempts=attempts, atomic_missing_search_route_rejections=rejected,
            atomic_rejection_fraction=rejected / attempts if attempts else None,
            partial_search_proposals=timeout_counts["partial_search_proposals"],
            candidate_shortage_agent_generation_records=timeout_counts["candidate_shortage_events"],
            actual_query_bearing_atomic_rejection_steps=sum(e["actual_query_bearing_atomic_rejection_steps"] for e in timeout_episodes),
            atomic_rejection_steps_with_all_searcher_semantic_goals_unchanged=sum(e["atomic_rejection_steps_with_all_searcher_semantic_goals_unchanged"] for e in timeout_episodes),
            atomic_label_steps_without_queries_excluded=sum(e["atomic_label_steps_without_queries_excluded"] for e in timeout_episodes),
            query_reasons=dict(timeout_queries), endpoint_diagnostic_samples=[dict(reason=key[0], start_endpoint_present=key[1], sample_count=count)
                for key, count in sorted(timeout_samples.items(), key=lambda item: repr(item[0]))],
            final100=combined_windows["windows"][-1],
            final100_without_new_or_aged_evidence_count=combined_windows["timeout_final100_without_new_or_aged_evidence_count"],
            final100_coverage_available_count=combined_windows["timeout_final100_coverage_available_count"],
            episodes=timeout_episodes), scene0_example=scene0, file_sha256=file_hashes,
        definitions=dict(atomic="Count steps with an actual recorded query and ATOMIC_REJECT_MISSING_SEARCH_ROUTE after the step; labels inherited by a query-free terminal step are excluded.",
            candidate_shortage="Agent-level candidate-generation shortage records, not episodes or independent failure events.",
            samples="Capped diagnostic sample entries, distinct from complete query counters and not independent trials.",
            goal_retention="Exact equality of all three public semantic waypoints across the query-bearing rejection transition.",
            final100="Physical transitions 301..400 in no-Found timeouts; stall denominator is three searchers times exposure.",
            aged="Repeat cell evidence after the configured revisit age; effective means a step with new or aged evidence, not target detection probability."),
        limitations=["Timeout-only strata are selected by the outcome and do not estimate overall treatment effects.",
            "An absent cached start connector does not establish physical map unreachability.",
            "No new or aged evidence is not zero observation calls or zero repeated evidence.",
            "Goal retention and stalls support a failure mechanism but do not establish that fixing it will cause Found."])


def audit(run, historical, previous, equivalence, case=CASE):
    run, historical, previous, equivalence = map(lambda p: Path(p).resolve(), (run, historical, previous, equivalence))
    inputs = Inputs()
    for path in (Path(__file__), PLAN, run / "identity.json", run / "episodes.json", run / "summary.json"):
        inputs.watch(path)
    for index in INDICES:
        child = run / f"scene_{index:04d}"
        for name in ("identity.json", "summary.json", "episodes.json"):
            inputs.watch(child / name)
        for variant in VARIANTS:
            inputs.watch(child / f"episode_{index:04d}" / variant / "summary.json")
    identity, rows = collect_completed(run)
    validate_rows(rows, inputs.read(PLAN))
    if inputs.read(run / "episodes.json") != rows:
        raise ValueError("root episodes do not exactly equal validated complete child arms")
    if file_hash(PLAN) != identity["sources_before"]["inventory"]["files"][PLAN.relative_to(ROOT).as_posix()]:
        raise ValueError("plan differs from the sealed experiment plan")
    if identity["input_sha256"].get(str(PLAN)) != file_hash(PLAN):
        raise ValueError("root experiment input hash does not bind the current plan")
    current = {arm_key(row): row for row in rows}
    old = historical_comparison(inputs, historical, run, rows)
    prior = previous_comparison(inputs, previous, current)
    telemetry = telemetry_comparison(inputs, equivalence, identity, current)
    public_map = case_comparison(inputs, Path(case).resolve(), current)
    mechanisms = baseline_mechanisms(inputs, run, rows)
    passed = (old["all_key_outcomes_equal"] and old["all_scene_contents_equal"]
        and old["all_native_configs_and_task_conditions_equal"] and old["all_manifests_json_equal"]
        and prior["all_completed_arm_signatures_equal"]
        and telemetry["observed_and_unobserved_signature_lists_equal"]
        and telemetry["current_matches_both_signatures"] and telemetry["current_matches_both_key_outcomes"]
        and public_map["current_matches_reference_and_replay"] and public_map["counts_match"])
    environment = inputs.read(run / "scene_0000/identity.json").get("environment")
    inputs.verify()
    return dict(schema="ch3.safe_search.development_baseline_audit.v1", audit_complete=True,
        all_requested_comparisons_equal=passed, current_complete_arm_count=100, current_independent_scene_count=20,
        current_source_inventory_sha256=identity["sources_before"]["inventory"]["sha256"],
        current_environment=environment,
        baseline_mechanisms=mechanisms,
        historical_baseline=old, previous_failed_development=prior,
        telemetry_equivalence_scene10_V3=telemetry, public_map_replay_scene7_V4=public_map,
        sources=dict(run_directory=str(run), historical_directory=str(historical), previous_directory=str(previous),
            equivalence_directory=str(equivalence), audited_file_count=len(inputs.hashes),
            audited_path_hash_inventory_sha256=digest(inputs.hashes), audit_script_sha256=file_hash(__file__),
            all_audited_files_unchanged=True),
        limitations=["This fixed, selected development set has 20 paired scenes; 100 arms are not 100 independent scenes.",
            "Historical baseline and diagnostic replays add zero episodes to current evaluation denominators.",
            "Historical Linux key-outcome reproduction is weaker than stepwise physical/RNG equivalence.",
            "Signature equality checks deterministic replay consistency; it does not establish performance or causal attribution."])


def output_path(path, protected_inputs):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("audit output must be a new file")
    for protected in [*(Path(p).resolve() for p in protected_inputs),
            *(ROOT / name for name in ("outputs", "3090结果", ".git", "core", "chapter3_bser", "configs", "tests", "scripts", "tools"))]:
        if path == protected or protected in path.parents:
            raise ValueError("cannot write audit report inside retained inputs or production source")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "historical-dir", "previous-dir", "equivalence-dir", "output-json"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--public-map-case", type=Path, default=CASE)
    args = parser.parse_args(argv)
    output = output_path(args.output_json, (args.run_dir, args.historical_dir, args.previous_dir, args.equivalence_dir))
    error = None
    try:
        result = audit(args.run_dir, args.historical_dir, args.previous_dir, args.equivalence_dir, args.public_map_case)
    except Exception as exc:
        error = exc
        result = dict(schema="ch3.safe_search.development_baseline_audit.v1", audit_complete=False,
            all_requested_comparisons_equal=False, error_type=type(exc).__name__, error=str(exc),
            audit_script_sha256=file_hash(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    if error is not None:
        raise error
    if not result["all_requested_comparisons_equal"]:
        raise SystemExit("Audit completed with differences; inspect the saved report.")


if __name__ == "__main__":
    main()
