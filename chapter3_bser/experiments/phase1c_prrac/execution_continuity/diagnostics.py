"""Episode and aggregate diagnostics for execution-continuity ablations."""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
import statistics
from typing import Any, Iterable, Mapping

from chapter3_bser.experiments.phase1c_prrac.evaluation_metrics import (
    aggregate_checkpoint,
    mcnemar_exact_p_value,
    wilson_interval,
)

from .types import (
    ExecutionContinuityDetectionV3,
    ExecutionNavigationPlanV3,
    ExecutionVariant,
    NavigationMode,
    ResidualSuppressionDiagnostics,
)


def _rate(numerator: int, denominator: int) -> float | None:
    return None if int(denominator) == 0 else int(numerator) / int(denominator)


def _mean(values: Iterable[Any]) -> float | None:
    numbers = []
    for value in values:
        if value is None or value == "":
            continue
        number = float(value)
        if math.isfinite(number):
            numbers.append(number)
    return None if not numbers else float(statistics.fmean(numbers))


@dataclass
class ExecutionContinuityDiagnostics:
    variant: ExecutionVariant
    post_found_step_count: int = 0
    exact_public_target_plan_count: int = 0
    exact_public_target_unreachable_count: int = 0
    reachable_proxy_plan_count: int = 0
    reachable_proxy_active_step_count: int = 0
    last_valid_route_plan_count: int = 0
    last_valid_route_active_step_count: int = 0
    safe_hold_entry_count: int = 0
    safe_hold_active_step_count: int = 0
    executor_route_active_post_found_steps: int = 0
    executor_route_inactive_post_found_steps: int = 0
    executor_invalid_count_post_found: int = 0
    assignment_unreachable_count_post_found: int = 0
    navigation_endpoint_switch_count: int = 0
    semantic_target_update_count: int = 0
    executor_residual_suppressed_step_count: int = 0
    executor_collision_count_post_found: int = 0
    executor_collision_max_streak_post_found: int = 0
    executor_first_collision_step_post_found: int | None = None
    executor_last_collision_step_post_found: int | None = None
    post_found_safe_hold_max_streak: int = 0
    post_found_safe_hold_terminal_streak: int = 0
    post_found_route_inactive_max_streak: int = 0
    post_found_route_inactive_terminal_streak: int = 0
    _executor_collision_streak: int = 0
    _safe_hold_streak: int = 0
    _route_inactive_streak: int = 0
    proxy_distances: list[float] = field(default_factory=list)
    suppressed_raw_norms: list[float] = field(default_factory=list)
    suppressed_applied_norms: list[float] = field(default_factory=list)
    event_counts: dict[str, int] = field(default_factory=dict)
    _last_plan: ExecutionNavigationPlanV3 | None = None

    def observe_plan(self, plan: ExecutionNavigationPlanV3, *, semantic_updated: bool) -> None:
        previous = self._last_plan
        if previous is not None and previous.navigation_endpoint != plan.navigation_endpoint:
            self.navigation_endpoint_switch_count += 1
        if semantic_updated:
            self.semantic_target_update_count += 1
        if plan.exact_public_target_unreachable:
            self.exact_public_target_unreachable_count += 1
        if plan.navigation_mode is NavigationMode.EXACT_PUBLIC_TARGET:
            self.exact_public_target_plan_count += 1
        elif plan.navigation_mode is NavigationMode.REACHABLE_PUBLIC_PROXY:
            self.reachable_proxy_plan_count += 1
        elif plan.navigation_mode is NavigationMode.LAST_VALID_ROUTE:
            self.last_valid_route_plan_count += 1
        elif plan.navigation_mode is NavigationMode.SAFE_HOLD and (
            previous is None or previous.navigation_mode is not NavigationMode.SAFE_HOLD
        ):
            self.safe_hold_entry_count += 1
        self._last_plan = plan

    def observe_step(
        self,
        *,
        post_found: bool,
        plan: ExecutionNavigationPlanV3 | None,
        detection: ExecutionContinuityDetectionV3 | None,
        suppression: ResidualSuppressionDiagnostics,
        legacy_route_active: bool | None = None,
        legacy_invalid_reason: str = "",
        executor_collision: bool = False,
        transition_step: int | None = None,
    ) -> None:
        if not post_found:
            return
        self.post_found_step_count += 1
        active = bool(
            legacy_route_active
            if plan is None and legacy_route_active is not None
            else plan is not None and plan.reachable and not plan.safe_hold
        )
        self.executor_route_active_post_found_steps += int(active)
        self.executor_route_inactive_post_found_steps += int(not active)
        self._route_inactive_streak = 0 if active else self._route_inactive_streak + 1
        self.post_found_route_inactive_max_streak = max(
            self.post_found_route_inactive_max_streak, self._route_inactive_streak
        )
        self.post_found_route_inactive_terminal_streak = self._route_inactive_streak
        if executor_collision:
            self.executor_collision_count_post_found += 1
            self._executor_collision_streak += 1
            self.executor_collision_max_streak_post_found = max(
                self.executor_collision_max_streak_post_found, self._executor_collision_streak
            )
            if self.executor_first_collision_step_post_found is None:
                self.executor_first_collision_step_post_found = transition_step
            self.executor_last_collision_step_post_found = transition_step
        else:
            self._executor_collision_streak = 0
        if plan is not None:
            if plan.navigation_mode is NavigationMode.REACHABLE_PUBLIC_PROXY:
                self.reachable_proxy_active_step_count += 1
                if plan.proxy_distance_to_semantic_target is not None:
                    self.proxy_distances.append(float(plan.proxy_distance_to_semantic_target))
            elif plan.navigation_mode is NavigationMode.LAST_VALID_ROUTE:
                self.last_valid_route_active_step_count += 1
            elif plan.navigation_mode is NavigationMode.SAFE_HOLD:
                self.safe_hold_active_step_count += 1
        safe_hold = bool(plan is not None and plan.navigation_mode is NavigationMode.SAFE_HOLD)
        self._safe_hold_streak = self._safe_hold_streak + 1 if safe_hold else 0
        self.post_found_safe_hold_max_streak = max(
            self.post_found_safe_hold_max_streak, self._safe_hold_streak
        )
        self.post_found_safe_hold_terminal_streak = self._safe_hold_streak
        if detection is not None:
            for name in detection.events:
                self.event_counts[str(name)] = self.event_counts.get(str(name), 0) + 1
            self.executor_invalid_count_post_found += int(detection.route_invalid)
            self.assignment_unreachable_count_post_found += int(
                "ASSIGNMENT_UNREACHABLE" in detection.events
            )
        elif legacy_invalid_reason:
            self.executor_invalid_count_post_found += int(
                legacy_invalid_reason
                in {"ASSIGNMENT_UNREACHABLE", "QUERY_UNREACHABLE", "PLANNING_COST_INCREASE"}
            )
            self.assignment_unreachable_count_post_found += int(
                legacy_invalid_reason == "ASSIGNMENT_UNREACHABLE"
            )
        if suppression.suppressed:
            self.executor_residual_suppressed_step_count += 1
            self.suppressed_raw_norms.append(float(suppression.raw_norm))
            self.suppressed_applied_norms.append(float(suppression.applied_norm))

    def summary(self) -> dict[str, Any]:
        denominator = self.post_found_step_count
        return {
            "post_found_step_count": denominator,
            "exact_public_target_plan_count": self.exact_public_target_plan_count,
            "exact_public_target_unreachable_count": self.exact_public_target_unreachable_count,
            "reachable_proxy_plan_count": self.reachable_proxy_plan_count,
            "reachable_proxy_active_step_count": self.reachable_proxy_active_step_count,
            "last_valid_route_plan_count": self.last_valid_route_plan_count,
            "last_valid_route_active_step_count": self.last_valid_route_active_step_count,
            "safe_hold_entry_count": self.safe_hold_entry_count,
            "safe_hold_active_step_count": self.safe_hold_active_step_count,
            "post_found_safe_hold_step_count": self.safe_hold_active_step_count,
            "post_found_safe_hold_max_streak": self.post_found_safe_hold_max_streak,
            "post_found_safe_hold_terminal_streak": self.post_found_safe_hold_terminal_streak,
            "executor_route_active_post_found_steps": self.executor_route_active_post_found_steps,
            "executor_route_inactive_post_found_steps": self.executor_route_inactive_post_found_steps,
            "post_found_route_inactive_step_count": self.executor_route_inactive_post_found_steps,
            "post_found_route_inactive_max_streak": self.post_found_route_inactive_max_streak,
            "post_found_route_inactive_terminal_streak": self.post_found_route_inactive_terminal_streak,
            "executor_collision_episode_post_found": bool(self.executor_collision_count_post_found),
            "executor_collision_count_post_found": self.executor_collision_count_post_found,
            "executor_collision_max_streak_post_found": self.executor_collision_max_streak_post_found,
            "executor_first_collision_step_post_found": self.executor_first_collision_step_post_found,
            "executor_last_collision_step_post_found": self.executor_last_collision_step_post_found,
            "executor_route_active_rate_post_found": _rate(self.executor_route_active_post_found_steps, denominator),
            "executor_invalid_count_post_found": self.executor_invalid_count_post_found,
            "assignment_unreachable_count_post_found": self.assignment_unreachable_count_post_found,
            "executor_invalid_rate_post_found": _rate(self.executor_invalid_count_post_found, denominator),
            "assignment_unreachable_rate_post_found": _rate(self.assignment_unreachable_count_post_found, denominator),
            "navigation_endpoint_switch_count": self.navigation_endpoint_switch_count,
            "semantic_target_update_count": self.semantic_target_update_count,
            "proxy_distance_to_semantic_target_mean": _mean(self.proxy_distances),
            "proxy_distance_to_semantic_target_max": None if not self.proxy_distances else max(self.proxy_distances),
            "executor_residual_suppressed_step_count": self.executor_residual_suppressed_step_count,
            "executor_residual_raw_norm_when_suppressed": _mean(self.suppressed_raw_norms),
            "executor_residual_applied_norm_when_suppressed": _mean(self.suppressed_applied_norms),
            "execution_continuity_event_counts": dict(sorted(self.event_counts.items())),
        }


def aggregate_execution_variant(rows: list[dict[str, Any]], info: Mapping[str, Any]) -> dict[str, Any]:
    result = aggregate_checkpoint(rows, info)
    found = [row for row in rows if bool(row.get("found"))]
    contact = [row for row in rows if bool(row.get("contact_episode"))]
    weighted_steps = sum(int(row.get("post_found_step_count") or 0) for row in rows)

    def weighted_rate(count_key: str) -> float | None:
        count = sum(int(row.get(count_key) or 0) for row in rows)
        return _rate(count, weighted_steps)

    result.update(
        {
            "executor_route_active_rate_post_found": weighted_rate("executor_route_active_post_found_steps"),
            "executor_invalid_rate_post_found": weighted_rate("executor_invalid_count_post_found"),
            "assignment_unreachable_rate_post_found": weighted_rate("assignment_unreachable_count_post_found"),
            "mean_exact_plan_count_if_found": _mean(row.get("exact_public_target_plan_count") for row in found),
            "mean_proxy_plan_count_if_found": _mean(row.get("reachable_proxy_plan_count") for row in found),
            "mean_last_valid_plan_count_if_found": _mean(row.get("last_valid_route_plan_count") for row in found),
            "mean_safe_hold_active_rate_if_found": _mean(
                _rate(int(row.get("safe_hold_active_step_count") or 0), int(row.get("post_found_step_count") or 0))
                for row in found
            ),
            "mean_executor_residual_ratio_post_found": _mean(row.get("executor_residual_ratio_post_found") for row in found),
            "mean_executor_residual_suppressed_steps_if_found": _mean(
                row.get("executor_residual_suppressed_step_count") for row in found
            ),
            "executor_collision_episode_post_found_count": sum(bool(row.get("executor_collision_episode_post_found")) for row in rows),
            "executor_collision_episode_post_found_rate": _rate(sum(bool(row.get("executor_collision_episode_post_found")) for row in rows), len(rows)),
            "executor_collision_count_post_found_sum": sum(int(row.get("executor_collision_count_post_found") or 0) for row in rows),
            "executor_collision_max_streak_post_found_mean": _mean(row.get("executor_collision_max_streak_post_found") for row in rows),
            "first_contact_step_mean_if_contact": _mean(row.get("first_contact_step") for row in contact),
            "found_to_first_contact_steps_mean_if_contact": _mean(row.get("found_to_first_contact_steps") for row in contact),
            "post_found_safe_hold_terminal_streak_mean": _mean(row.get("post_found_safe_hold_terminal_streak") for row in rows),
            "post_found_route_inactive_terminal_streak_mean": _mean(row.get("post_found_route_inactive_terminal_streak") for row in rows),
        }
    )
    found_count = sum(bool(row.get("found")) for row in rows)
    contact_count = sum(bool(row.get("contact_episode")) for row in rows)
    success_count = sum(bool(row.get("success")) for row in rows)
    for prefix, numerator, denominator in (
        ("found_rate", found_count, len(rows)),
        ("contact_rate", contact_count, len(rows)),
        ("success_rate", success_count, len(rows)),
        ("contact_if_found_rate", contact_count, found_count),
        ("success_if_found_rate", success_count, found_count),
    ):
        low, high = wilson_interval(numerator, denominator)
        result[f"{prefix}_ci_low"] = low
        result[f"{prefix}_ci_high"] = high
    return result


def paired_execution_variant_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["checkpoint"]),
            str(row["evaluation_mode"]),
            str(row["execution_variant"]),
            str(row["manifest_sha256"]),
        )
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    roots = sorted({(checkpoint, mode, manifest) for checkpoint, mode, _, manifest in grouped})
    order = [item.value for item in ExecutionVariant]
    for checkpoint, mode, manifest in roots:
        available = [variant for variant in order if (checkpoint, mode, variant, manifest) in grouped]
        for base_name, candidate_name in itertools.combinations(available, 2):
            base_rows = grouped[(checkpoint, mode, base_name, manifest)]
            candidate_rows = grouped[(checkpoint, mode, candidate_name, manifest)]
            base = {str(row["scenario_id"]): row for row in base_rows}
            candidate = {str(row["scenario_id"]): row for row in candidate_rows}
            if set(base) != set(candidate):
                raise ValueError("paired execution variants require identical scenario manifests")
            pairs = [(base[key], candidate[key]) for key in sorted(base)]
            both = sum(bool(left.get("success")) and bool(right.get("success")) for left, right in pairs)
            base_only = sum(bool(left.get("success")) and not bool(right.get("success")) for left, right in pairs)
            candidate_only = sum(not bool(left.get("success")) and bool(right.get("success")) for left, right in pairs)
            base_summary = aggregate_execution_variant([left for left, _ in pairs], {})
            candidate_summary = aggregate_execution_variant([right for _, right in pairs], {})

            def difference(field: str) -> float | None:
                left = base_summary.get(field)
                right = candidate_summary.get(field)
                return None if left is None or right is None else float(right) - float(left)

            output.append(
                {
                    "checkpoint": checkpoint,
                    "evaluation_mode": mode,
                    "manifest_sha256": manifest,
                    "base_execution_variant": base_name,
                    "candidate_execution_variant": candidate_name,
                    "paired_scenario_count": len(pairs),
                    "both_success": both,
                    "base_only_success": base_only,
                    "candidate_only_success": candidate_only,
                    "neither_success": len(pairs) - both - base_only - candidate_only,
                    "success_rate_difference": difference("success_rate"),
                    "found_rate_difference": difference("found_rate"),
                    "contact_rate_difference": difference("contact_rate"),
                    "contact_if_found_difference": difference("contact_if_found_rate"),
                    "collision_rate_difference": difference("collision_episode_rate"),
                    "executor_route_active_rate_difference": difference("executor_route_active_rate_post_found"),
                    "executor_invalid_rate_difference": difference("executor_invalid_rate_post_found"),
                    "assignment_unreachable_rate_difference": difference("assignment_unreachable_rate_post_found"),
                    "executor_min_distance_difference": difference("mean_executor_min_distance_if_found"),
                    "executor_final_distance_difference": difference("mean_executor_final_distance_if_found"),
                    "mcnemar_exact_p_value": mcnemar_exact_p_value(base_only, candidate_only),
                }
            )
    return output


__all__ = (
    "ExecutionContinuityDiagnostics",
    "aggregate_execution_variant",
    "paired_execution_variant_comparisons",
)
