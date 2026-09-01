"""S2-A aggregation, exact pairing, baseline strata, funnel, and preflight."""

from __future__ import annotations

import itertools
import math
import statistics
from typing import Any, Iterable, Mapping

from chapter3_bser.experiments.phase1c_prrac.evaluation_metrics import mcnemar_exact_p_value

from .types import SearchRecoveryVariant
from .types_v2 import SearchRecoveryVariantV2


def _numbers(rows: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None and value != "" and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _mean(rows: Iterable[Mapping[str, Any]], key: str):
    values = _numbers(rows, key)
    return None if not values else float(statistics.fmean(values))


def _median(rows: Iterable[Mapping[str, Any]], key: str):
    values = _numbers(rows, key)
    return None if not values else float(statistics.median(values))


def _rate(n: int, d: int):
    return None if d == 0 else n / d


def _percentile(rows: Iterable[Mapping[str, Any]], key: str, percentile: float):
    values = sorted(_numbers(rows, key))
    if not values:
        return None
    position = (len(values) - 1) * float(percentile)
    lower = int(math.floor(position)); upper = int(math.ceil(position))
    return float(values[lower] if lower == upper else values[lower] + (values[upper] - values[lower]) * (position - lower))


def aggregate_search_collision_recovery(rows: list[dict[str, Any]], info: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(info)
    found = [row for row in rows if bool(row.get("found"))]
    contact = [row for row in rows if bool(row.get("contact_episode"))]
    success = [row for row in rows if bool(row.get("success"))]
    collisions = [row for row in rows if bool(row.get("searcher_collision_episode_pre_found"))]
    result.update({
        "evaluation_episodes": len(rows),
        "found_count": len(found), "found_rate": _rate(len(found), len(rows)),
        "contact_count": len(contact), "contact_rate": _rate(len(contact), len(rows)),
        "contact_if_found_rate": _rate(len(contact), len(found)),
        "success_count": len(success), "success_rate": _rate(len(success), len(rows)),
        "success_if_found_rate": _rate(len(success), len(found)),
        "success_if_contact_rate": _rate(len(success), len(contact)),
        "pre_found_collision_episode_count": len(collisions),
        "pre_found_collision_episode_rate": _rate(len(collisions), len(rows)),
        "mean_found_step_if_found": _mean(found, "found_step"),
        "median_found_step_if_found": _median(found, "found_step"),
    })
    for prefix, key in (("pre_found_collision_count", "searcher_collision_count_pre_found_total"), ("pre_found_collision_max_streak", "searcher_collision_max_streak_pre_found")):
        for suffix, selected in (("all", rows), ("if_collision", collisions)):
            result[f"{prefix}_sum_{suffix}"] = sum(_numbers(selected, key))
            result[f"{prefix}_mean_{suffix}"] = _mean(selected, key)
            result[f"{prefix}_median_{suffix}"] = _median(selected, key)
            result[f"{prefix}_p50_{suffix}"] = result[f"{prefix}_median_{suffix}"]
            result[f"{prefix}_p90_{suffix}"] = _percentile(selected, key, .90)
            result[f"{prefix}_p95_{suffix}"] = _percentile(selected, key, .95)
        # Required compatibility names mean the all-episode distribution.
        for statistic_name in ("sum", "mean", "median", "p50", "p90", "p95"):
            result[f"{prefix}_{statistic_name}"] = result[f"{prefix}_{statistic_name}_all"]
    for key in (
        "route_refresh_attempt_count", "route_refresh_success_count", "route_refresh_failure_count",
        "egress_attempt_count", "egress_success_count", "egress_failure_count",
        "search_recovery_entry_count", "recovery_no_egress_count",
        "forced_public_refresh_count", "route_refresh_identical_to_base_count",
        "local_connector_attempt_count", "local_connector_plan_count", "local_connector_reached_count",
        "local_connector_collision_count", "graph_reconnect_attempt_count",
        "graph_reconnect_success_count", "graph_reconnect_failure_count",
        "recovery_state_non_normal_step_count", "recovery_plan_active_step_count",
        "recovery_guidance_changed_step_count", "recovery_effective_intervention_count",
        "path_changed_step_count", "recovery_failed_pass_through_count",
        "tier0_count", "tier1_count", "tier2_count", "tier3_count",
        "executor_collision_count_post_found", "post_found_safe_hold_step_count",
        "post_found_route_inactive_step_count",
    ):
        result[key] = sum(int(row.get(key) or 0) for row in rows)
    result["recovery_active_rate"] = _mean(rows, "search_recovery_active_rate")
    result["mean_recovery_duration"] = _mean(rows, "recovery_duration_mean")
    result["effective_recovery_active_rate"] = _mean(rows, "effective_recovery_active_rate")
    result["recovery_effective_intervention_episode_count"] = sum(
        bool(row.get("recovery_effective_intervention_episode")) for row in rows
    )
    result["tracking_waypoint_delta_norm_sum"] = sum(_numbers(rows, "tracking_waypoint_delta_norm_sum"))
    result["tracking_waypoint_delta_norm_mean"] = _mean(rows, "tracking_waypoint_delta_norm_mean")
    result["tracking_waypoint_delta_norm_max"] = max(_numbers(rows, "tracking_waypoint_delta_norm_max"), default=0.0)
    failure_distribution: dict[str, int] = {}
    for row in rows:
        row_distribution = row.get("failure_reason_distribution")
        if isinstance(row_distribution, Mapping):
            for reason, count in row_distribution.items():
                failure_distribution[str(reason)] = failure_distribution.get(str(reason), 0) + int(count)
            continue
        for agent_id in range(3):
            reason = row.get(f"last_recovery_failure_reason_agent_{agent_id}")
            if reason:
                failure_distribution[str(reason)] = failure_distribution.get(str(reason), 0) + 1
    result["failure_reason_distribution"] = failure_distribution
    result["candidate_tier_distribution"] = {
        f"tier{tier}": result[f"tier{tier}_count"] for tier in range(4)
    }
    for output, key in (
        ("post_found_collision_max_streak_mean", "executor_collision_max_streak_post_found"),
        ("first_contact_step_mean_if_contact", "first_contact_step"),
        ("found_to_first_contact_steps_mean_if_contact", "found_to_first_contact_steps"),
        ("post_found_safe_hold_max_streak_mean", "post_found_safe_hold_max_streak"),
        ("post_found_safe_hold_terminal_streak_mean", "post_found_safe_hold_terminal_streak"),
        ("post_found_route_inactive_max_streak_mean", "post_found_route_inactive_max_streak"),
        ("post_found_route_inactive_terminal_streak_mean", "post_found_route_inactive_terminal_streak"),
    ):
        result[output] = _mean(rows, key)
    return result


def _paired_groups(rows: list[dict[str, Any]]):
    keys = ("checkpoint", "checkpoint_config_hash", "checkpoint_runtime_revision", "evaluation_runtime_revision", "runtime_integration_mode", "execution_variant", "evaluation_mode", "manifest_sha256", "search_collision_recovery_schema", "search_collision_recovery_config_hash", "activation_diagnostics_schema", "activation_artifact_revision", "s2a1_activation_artifact_revision", "report_schema")
    groups: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in keys), {}).setdefault(str(row["search_recovery_variant"]), []).append(row)
    return keys, groups


def paired_search_collision_recovery_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys, groups = _paired_groups(rows)
    output = []
    order = [item.value for item in SearchRecoveryVariant] + [item.value for item in SearchRecoveryVariantV2]
    for root, variants in sorted(groups.items(), key=lambda item: str(item[0])):
        available = [name for name in order if name in variants]
        for left_name, right_name in itertools.combinations(available, 2):
            left = {str(row["scenario_id"]): row for row in variants[left_name]}
            right = {str(row["scenario_id"]): row for row in variants[right_name]}
            if set(left) != set(right):
                raise ValueError("paired search recovery variants require identical scenario_id sets")
            for identity in sorted(left):
                if int(left[identity]["scenario_seed"]) != int(right[identity]["scenario_seed"]):
                    raise ValueError(f"paired search recovery scenario_seed mismatch for scenario_id={identity}")
            pairs = [(left[key], right[key]) for key in sorted(left)]
            result = dict(zip(keys, root)); result.update({"left_search_recovery_variant": left_name, "right_search_recovery_variant": right_name, "paired_scenario_count": len(pairs), "continuous_difference_direction": "right_minus_left"})
            for label, field in (("found", "found"), ("success", "success"), ("contact", "contact_episode"), ("pre_found_collision", "searcher_collision_episode_pre_found")):
                both = sum(bool(a.get(field)) and bool(b.get(field)) for a, b in pairs)
                left_only = sum(bool(a.get(field)) and not bool(b.get(field)) for a, b in pairs)
                right_only = sum(not bool(a.get(field)) and bool(b.get(field)) for a, b in pairs)
                result.update({f"both_{label}": both, f"left_only_{label}": left_only, f"right_only_{label}": right_only, f"neither_{label}": len(pairs)-both-left_only-right_only, f"{label}_mcnemar_exact_p_value": mcnemar_exact_p_value(left_only, right_only)})
            for label, field, better in (("pre_found_collision_count", "searcher_collision_count_pre_found_total", "lower"), ("pre_found_collision_max_streak", "searcher_collision_max_streak_pre_found", "lower"), ("found_step", "found_step", "lower"), ("searcher_distance_travelled", "searcher_distance_travelled_pre_found", "report_only"), ("map_known_fraction_gain", "map_known_fraction_gain_pre_found", "higher")):
                differences = [float(b[field])-float(a[field]) for a,b in pairs if a.get(field) is not None and b.get(field) is not None]
                result[f"{label}_mean_difference_right_minus_left"] = None if not differences else float(statistics.fmean(differences))
                result[f"{label}_median_difference_right_minus_left"] = None if not differences else float(statistics.median(differences))
                result[f"{label}_better_direction"] = better
                if better == "lower":
                    result[f"{label}_left_better_count"] = sum(value > 0 for value in differences)
                    result[f"{label}_right_better_count"] = sum(value < 0 for value in differences)
                elif better == "higher":
                    result[f"{label}_left_better_count"] = sum(value < 0 for value in differences)
                    result[f"{label}_right_better_count"] = sum(value > 0 for value in differences)
                else:
                    result[f"{label}_left_better_count"] = None
                    result[f"{label}_right_better_count"] = None
                result[f"{label}_equal_count"] = sum(value == 0 for value in differences)
            output.append(result)
    return output


def paired_search_collision_recovery_baseline_strata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys, groups = _paired_groups(rows); output=[]
    for root, variants in sorted(groups.items(), key=lambda item: str(item[0])):
        baseline_name = (SearchRecoveryVariantV2.S2A1_C0_BASELINE.value
                         if SearchRecoveryVariantV2.S2A1_C0_BASELINE.value in variants
                         else SearchRecoveryVariant.S2A_C0_BASELINE.value)
        if baseline_name not in variants: continue
        baseline = {str(row["scenario_id"]): row for row in variants[baseline_name]}
        candidate_names = ((SearchRecoveryVariantV2.S2A1_C1_FORCED_REFRESH.value, SearchRecoveryVariantV2.S2A1_C2_LOCAL_CONNECTOR.value)
                           if baseline_name == SearchRecoveryVariantV2.S2A1_C0_BASELINE.value
                           else (SearchRecoveryVariant.S2A_C1_ROUTE_REFRESH.value, SearchRecoveryVariant.S2A_C2_EGRESS_ROUTE.value))
        for candidate_name in candidate_names:
            if candidate_name not in variants: continue
            candidate = {str(row["scenario_id"]): row for row in variants[candidate_name]}
            if set(baseline) != set(candidate): raise ValueError("baseline strata require identical scenario_id sets")
            for identity in sorted(baseline):
                if int(baseline[identity]["scenario_seed"]) != int(candidate[identity]["scenario_seed"]):
                    raise ValueError(f"baseline strata scenario_seed mismatch for scenario_id={identity}")
            for stratum, collision in (("BASELINE_COLLISION", True), ("BASELINE_NO_COLLISION", False)):
                ids = [key for key,row in baseline.items() if bool(row.get("searcher_collision_episode_pre_found")) is collision]
                base_rows=[baseline[key] for key in ids]; candidate_rows=[candidate[key] for key in ids]
                item=dict(zip(keys,root)); item.update({"baseline_search_recovery_variant":baseline_name,"candidate_search_recovery_variant":candidate_name,"stratum":stratum,"stratum_definition":f"{baseline_name}.searcher_collision_episode_pre_found","scenario_count":len(ids)})
                for outcome in ("found","success"):
                    br=_rate(sum(bool(r.get(outcome)) for r in base_rows),len(ids)); cr=_rate(sum(bool(r.get(outcome)) for r in candidate_rows),len(ids)); item[f"baseline_{outcome}_rate"]=br; item[f"candidate_{outcome}_rate"]=cr; item[f"{outcome}_difference"]=None if br is None else cr-br
                item["baseline_collision_count_mean"]=_mean(base_rows,"searcher_collision_count_pre_found_total"); item["candidate_collision_count_mean"]=_mean(candidate_rows,"searcher_collision_count_pre_found_total")
                item["baseline_max_streak_mean"]=_mean(base_rows,"searcher_collision_max_streak_pre_found"); item["candidate_max_streak_mean"]=_mean(candidate_rows,"searcher_collision_max_streak_pre_found")
                output.append(item)
    return output


def search_collision_recovery_failure_funnel(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_keys=("checkpoint","execution_variant","evaluation_mode","search_recovery_variant","manifest_sha256")
    groups={}
    for row in rows: groups.setdefault(tuple(row.get(key) for key in group_keys),[]).append(row)
    output=[]
    for key, values in sorted(groups.items(),key=lambda item:str(item[0])):
        categories={"FOUND":lambda r:bool(r.get("found")),"NOT_FOUND_WITH_PREFIND_COLLISION":lambda r:not bool(r.get("found")) and bool(r.get("searcher_collision_episode_pre_found")),"NOT_FOUND_WITHOUT_PREFIND_COLLISION":lambda r:not bool(r.get("found")) and not bool(r.get("searcher_collision_episode_pre_found")),"RECOVERY_TRIGGERED_FOUND":lambda r:int(r.get("search_recovery_entry_count") or 0)>0 and bool(r.get("found")),"RECOVERY_TRIGGERED_NOT_FOUND":lambda r:int(r.get("search_recovery_entry_count") or 0)>0 and not bool(r.get("found")),"RECOVERY_NOT_TRIGGERED_FOUND":lambda r:int(r.get("search_recovery_entry_count") or 0)==0 and bool(r.get("found")),"RECOVERY_NOT_TRIGGERED_NOT_FOUND":lambda r:int(r.get("search_recovery_entry_count") or 0)==0 and not bool(r.get("found"))}
        for name,predicate in categories.items():
            count=sum(predicate(row) for row in values); item=dict(zip(group_keys,key)); item.update({"category":name,"count":count,"rate":_rate(count,len(values))}); output.append(item)
    return output


def validate_s2a_baseline_regression(rows: list[dict[str, Any]], *, expected_scenario_ids: Iterable[str], expected_found: int = 48, expected_success: int = 26) -> dict[str, Any]:
    baseline=[row for row in rows if row.get("search_recovery_variant")==SearchRecoveryVariant.S2A_C0_BASELINE.value]
    if not baseline:
        baseline=[row for row in rows if row.get("search_recovery_variant")==SearchRecoveryVariantV2.S2A1_C0_BASELINE.value]
    checks={"row_count_100":len(baseline)==100,"scenario_ids_match":{str(row["scenario_id"]) for row in baseline}=={str(value) for value in expected_scenario_ids},"found_48":sum(bool(row.get("found")) for row in baseline)==expected_found,"success_26":sum(bool(row.get("success")) for row in baseline)==expected_success,"runtime_provenance":all(row.get("runtime_integration_mode")=="native" and row.get("execution_variant")=="B1_ATOMIC_LAST_VALID" and row.get("evaluation_mode")=="full_prrac" and row.get("checkpoint_runtime_revision")=="dynamic_public_intercept_v3_atomic_continuity" and row.get("evaluation_runtime_revision")=="dynamic_public_intercept_v3_atomic_continuity" and bool(row.get("manifest_sha256")) and bool(row.get("search_collision_recovery_config_hash")) for row in baseline),"single_manifest":len({row.get("manifest_sha256") for row in baseline})==1,"single_checkpoint":len({row.get("checkpoint") for row in baseline})==1}
    return {"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"result_rows_preserved":True}


__all__=("aggregate_search_collision_recovery","paired_search_collision_recovery_comparisons","paired_search_collision_recovery_baseline_strata","search_collision_recovery_failure_funnel","validate_s2a_baseline_regression")
