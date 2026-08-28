"""Deterministic aggregation for SEARCH continuity episode records."""

from __future__ import annotations

import statistics
from typing import Any, Iterable, Mapping

from chapter3_bser.experiments.phase1c_prrac.evaluation_metrics import (
    mcnemar_exact_p_value,
)

from .types import nullable_rate


def _values(rows: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _mean(rows: Iterable[Mapping[str, Any]], key: str):
    values = _values(rows, key)
    return None if not values else float(statistics.fmean(values))


def _median(rows: Iterable[Mapping[str, Any]], key: str):
    values = _values(rows, key)
    return None if not values else float(statistics.median(values))


def aggregate_search_continuity(
    rows: list[dict[str, Any]], info: Mapping[str, Any]
) -> dict[str, Any]:
    count = len(rows)
    found = [row for row in rows if bool(row.get("found"))]
    success = [row for row in rows if bool(row.get("success"))]
    contact = [row for row in rows if bool(row.get("contact_episode"))]
    collision = [row for row in rows if bool(row.get("searcher_collision_episode_pre_found"))]
    no_collision = [row for row in rows if not bool(row.get("searcher_collision_episode_pre_found"))]
    def found_before(row: Mapping[str, Any], fraction: float) -> bool:
        found_step = row.get("found_step")
        max_steps = row.get("max_steps")
        return bool(
            row.get("found")
            and found_step is not None
            and max_steps is not None
            and float(found_step) <= fraction * float(max_steps)
        )

    half = sum(found_before(row, 0.5) for row in rows)
    three_quarter = sum(found_before(row, 0.75) for row in rows)
    result = dict(info)
    result.update({
        "evaluation_episodes": count,
        "found_count": len(found),
        "found_rate": nullable_rate(len(found), count),
        "success_count": len(success),
        "success_rate": nullable_rate(len(success), count),
        "contact_if_found_rate": nullable_rate(
            sum(bool(row.get("contact_episode")) for row in found), len(found)
        ),
        "success_if_found_rate": nullable_rate(
            sum(bool(row.get("success")) for row in found), len(found)
        ),
        "median_found_step_if_found": _median(found, "found_step"),
        "mean_found_step_if_found": _mean(found, "found_step"),
        "found_before_half_horizon_count": half,
        "found_before_half_horizon_rate": nullable_rate(half, count),
        "found_before_three_quarter_horizon_count": three_quarter,
        "found_before_three_quarter_horizon_rate": nullable_rate(three_quarter, count),
        "pre_found_collision_episode_count": len(collision),
        "pre_found_collision_episode_rate": nullable_rate(len(collision), count),
        "found_rate_if_pre_found_collision": nullable_rate(sum(bool(r.get("found")) for r in collision), len(collision)),
        "found_rate_if_no_pre_found_collision": nullable_rate(sum(bool(r.get("found")) for r in no_collision), len(no_collision)),
    })
    for output, source in (
        ("mean_searcher_route_active_rate_pre_found", "searcher_route_active_rate_pre_found"),
        ("mean_searcher_hold_rate_pre_found", "searcher_hold_rate_pre_found"),
        ("mean_searcher_distance_travelled_pre_found", "searcher_distance_travelled_pre_found"),
        ("mean_map_known_fraction_gain_pre_found", "map_known_fraction_gain_pre_found"),
        ("mean_target_belief_entropy_delta_pre_found", "target_belief_entropy_delta_pre_found"),
        ("mean_target_belief_peak_delta_pre_found", "target_belief_peak_delta_pre_found"),
        ("mean_searcher_raw_residual_norm_pre_found", "searcher_raw_residual_norm_mean_pre_found"),
        ("mean_searcher_applied_residual_norm_pre_found", "searcher_applied_residual_norm_mean_pre_found"),
        ("mean_searcher_residual_negative_alignment_rate_pre_found", "searcher_residual_negative_alignment_rate_pre_found"),
        ("mean_searcher_assignment_switch_count_pre_found", "searcher_assignment_switch_count_pre_found"),
        ("mean_searcher_tracking_subgoal_switch_count_pre_found", "searcher_tracking_subgoal_switch_count_pre_found"),
        ("mean_searcher_residual_suppressed_env_step_count_pre_found", "searcher_residual_suppressed_env_step_count_pre_found"),
        ("mean_searcher_residual_suppressed_agent_step_count_pre_found", "searcher_residual_suppressed_agent_step_count_pre_found"),
        ("mean_searcher_raw_action_norm_pre_found", "searcher_raw_action_norm_pre_found"),
        ("mean_searcher_applied_action_norm_pre_found", "searcher_applied_action_norm_pre_found"),
        ("mean_searcher_residual_alignment_zero_navigation_count_pre_found", "searcher_residual_alignment_zero_navigation_count_pre_found"),
        ("mean_searcher_residual_alignment_zero_residual_count_pre_found", "searcher_residual_alignment_zero_residual_count_pre_found"),
    ):
        result[output] = _mean(rows, source)
    return result


def paired_searcher_residual_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "checkpoint", "checkpoint_config_hash", "checkpoint_runtime_revision",
        "evaluation_runtime_revision", "runtime_integration_mode", "execution_variant",
        "manifest_sha256", "search_continuity_diagnostics_hash",
    )
    groups: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in keys), {}).setdefault(str(row["evaluation_mode"]), []).append(row)
    output = []
    for group_key, modes in sorted(groups.items(), key=lambda item: str(item[0])):
        if "full_prrac" not in modes or "searcher_residual_off" not in modes:
            continue
        full = {str(row["scenario_id"]): row for row in modes["full_prrac"]}
        off = {str(row["scenario_id"]): row for row in modes["searcher_residual_off"]}
        if set(full) != set(off):
            raise ValueError("paired searcher residual comparison requires identical scenario_id sets")
        pairs = [(full[key], off[key]) for key in sorted(full)]
        result = dict(zip(keys, group_key))
        result["paired_scenario_count"] = len(pairs)
        for outcome in ("found", "success"):
            both = sum(bool(a.get(outcome)) and bool(b.get(outcome)) for a, b in pairs)
            full_only = sum(bool(a.get(outcome)) and not bool(b.get(outcome)) for a, b in pairs)
            off_only = sum(not bool(a.get(outcome)) and bool(b.get(outcome)) for a, b in pairs)
            result[f"both_{outcome}"] = both
            result[f"full_only_{outcome}"] = full_only
            result[f"searcher_off_only_{outcome}"] = off_only
            result[f"neither_{outcome}"] = len(pairs) - both - full_only - off_only
            result[f"{outcome}_mcnemar_exact_p_value"] = mcnemar_exact_p_value(full_only, off_only)
        full_summary = aggregate_search_continuity(list(full.values()), {})
        off_summary = aggregate_search_continuity(list(off.values()), {})
        for output_name, source in (
            ("found_rate_difference", "found_rate"),
            ("success_rate_difference", "success_rate"),
            ("contact_if_found_rate_difference", "contact_if_found_rate"),
            ("pre_found_collision_episode_rate_difference", "pre_found_collision_episode_rate"),
            ("mean_found_step_difference", "mean_found_step_if_found"),
            ("median_found_step_difference", "median_found_step_if_found"),
            ("searcher_route_active_rate_difference", "mean_searcher_route_active_rate_pre_found"),
            ("map_known_fraction_gain_difference", "mean_map_known_fraction_gain_pre_found"),
            ("searcher_distance_travelled_difference", "mean_searcher_distance_travelled_pre_found"),
            ("searcher_applied_action_norm_difference", "mean_searcher_applied_action_norm_pre_found"),
            ("searcher_assignment_switch_count_difference", "mean_searcher_assignment_switch_count_pre_found"),
            ("searcher_tracking_subgoal_switch_count_difference", "mean_searcher_tracking_subgoal_switch_count_pre_found"),
        ):
            left, right = full_summary.get(source), off_summary.get(source)
            result[output_name] = None if left is None or right is None else float(right) - float(left)
        output.append(result)
    return output


def search_failure_funnel(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_keys = (
        "checkpoint", "checkpoint_runtime_revision", "evaluation_runtime_revision",
        "runtime_integration_mode", "execution_variant", "evaluation_mode",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in group_keys), []).append(row)
    output = []
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        categories = {
            "FOUND": [row for row in values if bool(row.get("found"))],
            "NOT_FOUND_WITH_PREFIND_COLLISION": [row for row in values if not bool(row.get("found")) and bool(row.get("searcher_collision_episode_pre_found"))],
            "NOT_FOUND_WITHOUT_PREFIND_COLLISION": [row for row in values if not bool(row.get("found")) and not bool(row.get("searcher_collision_episode_pre_found"))],
        }
        for category, selected in categories.items():
            row = dict(zip(group_keys, key))
            row.update({"category": category, "count": len(selected), "rate": nullable_rate(len(selected), len(values))})
            for output_name, source in (
                ("mean_route_active_rate", "searcher_route_active_rate_pre_found"),
                ("mean_hold_rate", "searcher_hold_rate_pre_found"),
                ("mean_distance_travelled", "searcher_distance_travelled_pre_found"),
                ("mean_map_known_gain", "map_known_fraction_gain_pre_found"),
                ("mean_belief_entropy_delta", "target_belief_entropy_delta_pre_found"),
                ("mean_residual_norm", "searcher_applied_residual_norm_mean_pre_found"),
                ("mean_searcher_assignment_switch_count_pre_found", "searcher_assignment_switch_count_pre_found"),
                ("mean_searcher_tracking_subgoal_switch_count_pre_found", "searcher_tracking_subgoal_switch_count_pre_found"),
                ("mean_searcher_residual_suppressed_env_step_count_pre_found", "searcher_residual_suppressed_env_step_count_pre_found"),
                ("mean_searcher_residual_suppressed_agent_step_count_pre_found", "searcher_residual_suppressed_agent_step_count_pre_found"),
                ("mean_searcher_raw_action_norm_pre_found", "searcher_raw_action_norm_pre_found"),
                ("mean_searcher_applied_action_norm_pre_found", "searcher_applied_action_norm_pre_found"),
                ("mean_searcher_residual_alignment_zero_navigation_count_pre_found", "searcher_residual_alignment_zero_navigation_count_pre_found"),
                ("mean_searcher_residual_alignment_zero_residual_count_pre_found", "searcher_residual_alignment_zero_residual_count_pre_found"),
            ):
                row[output_name] = _mean(selected, source)
            output.append(row)
    return output


__all__ = (
    "aggregate_search_continuity",
    "paired_searcher_residual_comparisons",
    "search_failure_funnel",
)
