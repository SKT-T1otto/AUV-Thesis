"""Pure aggregation helpers for deterministic PRRAC checkpoint evaluation."""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Mapping

import numpy as np


STAGES = ("search", "intercept", "hold")
SELECTION_RULE = (
    "success_rate desc; success_if_found_rate desc; contact_if_found_rate desc; "
    "collision_episode_rate asc; mean_assignment_unreachable_if_found asc; "
    "checkpoint_episode asc"
)


def safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if int(denominator) <= 0 else int(numerator) / int(denominator)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    successes = int(successes)
    total = int(total)
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def router_class_metrics(matrix: Iterable[Iterable[int]]) -> dict[str, Any]:
    values = np.asarray(tuple(tuple(row) for row in matrix), dtype=np.int64)
    if values.shape != (3, 3):
        raise ValueError("router confusion matrix must be 3x3")
    total = int(values.sum())
    result: dict[str, Any] = {
        "router_accuracy": None if total == 0 else float(np.trace(values) / total)
    }
    recalls: list[float] = []
    for index, name in enumerate(STAGES):
        actual = int(values[index, :].sum())
        predicted = int(values[:, index].sum())
        true_positive = int(values[index, index])
        recall = None if actual == 0 else true_positive / actual
        precision = None if predicted == 0 else true_positive / predicted
        result[f"router_recall_{name}"] = recall
        result[f"router_precision_{name}"] = precision
        if recall is not None:
            recalls.append(float(recall))
    result["router_balanced_accuracy"] = (
        None if not recalls else float(statistics.fmean(recalls))
    )
    return result


def failure_stage(row: Mapping[str, Any]) -> str:
    if bool(row.get("success")):
        return "SUCCESS"
    if not bool(row.get("found")):
        return "NOT_FOUND"
    if not bool(row.get("contact_episode")):
        return "FOUND_NO_CONTACT"
    if not bool(row.get("hold_episode")):
        return "CONTACT_NO_HOLD"
    return "HOLD_NO_SUCCESS"


def _numbers(rows: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def _mean(rows: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values = _numbers(rows, key)
    return None if not values else float(statistics.fmean(values))


def _median(rows: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values = _numbers(rows, key)
    return None if not values else float(statistics.median(values))


def _sum_confusions(rows: Iterable[Mapping[str, Any]]) -> list[list[int]]:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for row in rows:
        value = row.get("router_confusion_matrix")
        if value is None:
            continue
        candidate = np.asarray(value, dtype=np.int64)
        if candidate.shape != (3, 3):
            raise ValueError("episode router confusion matrix must be 3x3")
        matrix += candidate
    return [[int(item) for item in row] for row in matrix.tolist()]


def aggregate_checkpoint(
    rows: list[dict[str, Any]], checkpoint_info: Mapping[str, Any]
) -> dict[str, Any]:
    count = len(rows)
    found_rows = [row for row in rows if bool(row.get("found"))]
    contact_rows = [row for row in rows if bool(row.get("contact_episode"))]
    hold_rows = [row for row in rows if bool(row.get("hold_episode"))]
    success_rows = [row for row in rows if bool(row.get("success"))]
    found_count = len(found_rows)
    contact_count = len(contact_rows)
    hold_count = len(hold_rows)
    success_count = len(success_rows)
    collision_count = sum(bool(row.get("collision_episode")) for row in rows)
    post_collision_count = sum(
        int(row.get("post_found_collision_count") or 0) > 0 for row in rows
    )
    result = dict(checkpoint_info)
    result.update(
        {
            "evaluation_episodes": count,
            "found_count": found_count,
            "contact_count": contact_count,
            "hold_count": hold_count,
            "success_count": success_count,
            "found_rate": safe_rate(found_count, count),
            "contact_rate": safe_rate(contact_count, count),
            "hold_rate": safe_rate(hold_count, count),
            "success_rate": safe_rate(success_count, count),
            "contact_if_found_rate": safe_rate(contact_count, found_count),
            "hold_if_contact_rate": safe_rate(hold_count, contact_count),
            "success_if_found_rate": safe_rate(success_count, found_count),
            "success_if_contact_rate": safe_rate(success_count, contact_count),
            "collision_episode_rate": safe_rate(collision_count, count),
            "post_found_collision_episode_rate": safe_rate(post_collision_count, count),
        }
    )
    for prefix, successes, total in (
        ("found_rate", found_count, count),
        ("success_rate", success_count, count),
        ("success_if_found", success_count, found_count),
        ("contact_if_found", contact_count, found_count),
    ):
        low, high = wilson_interval(successes, total)
        result[f"{prefix}_ci_low"] = low
        result[f"{prefix}_ci_high"] = high

    conditional_fields = (
        ("executor_invalid_count", "executor_invalid_count"),
        ("assignment_unreachable", "executor_invalid_assignment_unreachable_count"),
        ("executor_min_distance", "executor_min_distance_to_target"),
        ("executor_final_distance", "executor_final_distance_to_target"),
    )
    for output_name, source_name in conditional_fields:
        result[f"mean_{output_name}_if_found"] = _mean(found_rows, source_name)
        result[f"median_{output_name}_if_found"] = _median(found_rows, source_name)
    result["mean_executor_replan_count_if_found"] = _mean(
        found_rows, "executor_replan_count"
    )
    result["mean_post_found_residual_ratio_if_found"] = _mean(
        found_rows, "executor_residual_ratio_post_found"
    )
    result["mean_handoff_delay_if_found"] = _mean(found_rows, "handoff_delay")
    result["mean_found_to_success_steps_if_success"] = _mean(
        success_rows, "found_to_success_steps"
    )
    matrix = _sum_confusions(rows)
    result["router_confusion_matrix"] = matrix
    result.update(router_class_metrics(matrix))
    for key in (
        "gate_mean",
        "gate_p10",
        "gate_p90",
        "alignment_negative_rate",
    ):
        result[key] = _mean(rows, key)
    return result


def mcnemar_exact_p_value(base_only: int, candidate_only: int) -> float:
    base_only = int(base_only)
    candidate_only = int(candidate_only)
    discordant = base_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(base_only, candidate_only) + 1)
    ) / (2.0**discordant)
    return min(1.0, 2.0 * tail)


def paired_checkpoint_comparison(
    base_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    base_checkpoint: str,
    candidate_checkpoint: str,
    evaluation_mode: str,
) -> dict[str, Any]:
    base = {str(row["scenario_id"]): row for row in base_rows}
    candidate = {str(row["scenario_id"]): row for row in candidate_rows}
    if set(base) != set(candidate):
        raise ValueError("paired checkpoint comparison requires identical scenario_id sets")
    pairs = [(base[key], candidate[key]) for key in sorted(base)]
    both = sum(bool(left.get("success")) and bool(right.get("success")) for left, right in pairs)
    base_only = sum(bool(left.get("success")) and not bool(right.get("success")) for left, right in pairs)
    candidate_only = sum(not bool(left.get("success")) and bool(right.get("success")) for left, right in pairs)
    neither = len(pairs) - both - base_only - candidate_only
    base_summary = aggregate_checkpoint([left for left, _ in pairs], {})
    candidate_summary = aggregate_checkpoint([right for _, right in pairs], {})

    def difference(name: str) -> float | None:
        left = base_summary.get(name)
        right = candidate_summary.get(name)
        return None if left is None or right is None else float(right) - float(left)

    return {
        "base_checkpoint": str(base_checkpoint),
        "candidate_checkpoint": str(candidate_checkpoint),
        "evaluation_mode": str(evaluation_mode),
        "paired_scenario_count": len(pairs),
        "both_success": both,
        "base_only_success": base_only,
        "candidate_only_success": candidate_only,
        "neither_success": neither,
        "success_rate_difference": difference("success_rate"),
        "found_rate_difference": difference("found_rate"),
        "contact_if_found_difference": difference("contact_if_found_rate"),
        "collision_rate_difference": difference("collision_episode_rate"),
        "assignment_unreachable_mean_difference": difference(
            "mean_assignment_unreachable_if_found"
        ),
        "executor_min_distance_mean_difference": difference(
            "mean_executor_min_distance_if_found"
        ),
        "mcnemar_exact_p_value": mcnemar_exact_p_value(base_only, candidate_only),
    }


def recommend_checkpoint(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row for row in summary_rows if row.get("evaluation_mode") == "full_prrac"
    ]
    if not eligible:
        return {
            "selection_rule": SELECTION_RULE,
            "recommended_checkpoint": None,
            "recommended_checkpoint_episode": None,
            "performance_passed": None,
        }

    def descending(value: Any) -> float:
        return -float(value) if value is not None and value != "" else math.inf

    def ascending(value: Any) -> float:
        return float(value) if value is not None and value != "" else math.inf

    selected = min(
        eligible,
        key=lambda row: (
            descending(row.get("success_rate")),
            descending(row.get("success_if_found_rate")),
            descending(row.get("contact_if_found_rate")),
            ascending(row.get("collision_episode_rate")),
            ascending(row.get("mean_assignment_unreachable_if_found")),
            int(row.get("checkpoint_episode", 0)),
        ),
    )
    return {
        "selection_rule": SELECTION_RULE,
        "recommended_checkpoint": selected.get("checkpoint"),
        "recommended_checkpoint_episode": int(selected.get("checkpoint_episode", 0)),
        "performance_passed": None,
    }


__all__ = (
    "SELECTION_RULE",
    "STAGES",
    "aggregate_checkpoint",
    "failure_stage",
    "mcnemar_exact_p_value",
    "paired_checkpoint_comparison",
    "recommend_checkpoint",
    "router_class_metrics",
    "safe_rate",
    "wilson_interval",
)
