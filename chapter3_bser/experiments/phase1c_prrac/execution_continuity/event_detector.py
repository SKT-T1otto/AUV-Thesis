"""Endpoint-aware event semantics for the opt-in v3 runtime overlay."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from core.mapping.planning_state import PlanningStateView
from core.mapping.travel_cost_service import TravelCostService

from .types import ExecutionContinuityDetectionV3, ExecutionNavigationPlanV3, NavigationMode


def _relative_change(current: float, installed: float) -> float:
    if not np.isfinite(current) or not np.isfinite(installed):
        return 0.0
    return max(0.0, (float(current) - float(installed)) / max(abs(float(installed)), 1e-12))


class ExecutionContinuityEventDetector:
    """Validate the installed navigation endpoint, never the unreachable semantic target."""

    def __init__(
        self,
        *,
        executor_cost_increase_threshold: float = 0.15,
        defer_stale_endpoint_invalid: bool = True,
    ) -> None:
        self.executor_cost_increase_threshold = float(executor_cost_increase_threshold)
        self.defer_stale_endpoint_invalid = bool(defer_stale_endpoint_invalid)

    def detect(
        self,
        state: PlanningStateView,
        plan: ExecutionNavigationPlanV3,
        semantic_target: Iterable[float],
        *,
        retry_due: bool,
        service: TravelCostService | None = None,
    ) -> ExecutionContinuityDetectionV3:
        target = np.asarray(tuple(semantic_target), dtype=np.float64).reshape(3)
        shift = float(np.linalg.norm(target - np.asarray(plan.semantic_target, dtype=np.float64)))
        if plan.navigation_mode is NavigationMode.SAFE_HOLD:
            events = ["SAFE_HOLD_RETRY_PENDING"]
            if retry_due:
                events.append("EXECUTOR_ROUTE_RETRY")
            return ExecutionContinuityDetectionV3(
                events=tuple(events),
                route_invalid=False,
                retry_due=bool(retry_due),
                validity_evaluated=False,
                validity_deferred=False,
                executor_invalid_reason="SAFE_HOLD_RETRY_PENDING",
                query_failure_reason="",
                current_planning_cost=None,
                planning_cost_relative_change=0.0,
                public_target_shift=shift,
            )

        executor = state.agents[state.executor_id]
        if not plan.reachable:
            return ExecutionContinuityDetectionV3(
                events=("ASSIGNMENT_UNREACHABLE",),
                route_invalid=True,
                retry_due=bool(retry_due),
                validity_evaluated=False,
                validity_deferred=False,
                executor_invalid_reason="ASSIGNMENT_UNREACHABLE",
                query_failure_reason=str(plan.failure_reason or ""),
                current_planning_cost=None,
                planning_cost_relative_change=0.0,
                public_target_shift=shift,
            )
        query = (service or TravelCostService(state)).query(
            executor.position, plan.navigation_endpoint, executor
        )
        events: list[str] = []
        if plan.navigation_mode is NavigationMode.REACHABLE_PUBLIC_PROXY:
            events.append("PROXY_ACTIVE")
        elif plan.navigation_mode is NavigationMode.LAST_VALID_ROUTE:
            events.append("LAST_VALID_ACTIVE")
        relative = _relative_change(float(query.planning_cost), float(plan.planning_cost))
        if (
            not query.reachable
            and self.defer_stale_endpoint_invalid
            and str(query.failure_reason or "")
            in {"no_start_connector", "no_goal_connector"}
        ):
            reason = "STALE_ENDPOINT_SNAPSHOT_DEFERRED"
            events.append(reason)
            invalid = False
            validity_evaluated = False
            validity_deferred = True
        elif not query.reachable:
            reason = "QUERY_UNREACHABLE"
            events.append(reason)
            invalid = True
            validity_evaluated = True
            validity_deferred = False
        elif relative > self.executor_cost_increase_threshold:
            reason = "PLANNING_COST_INCREASE"
            events.append(reason)
            invalid = True
            validity_evaluated = True
            validity_deferred = False
        else:
            reason = "VALID"
            invalid = False
            validity_evaluated = True
            validity_deferred = False
        return ExecutionContinuityDetectionV3(
            events=tuple(events),
            route_invalid=invalid,
            retry_due=bool(retry_due),
            validity_evaluated=validity_evaluated,
            validity_deferred=validity_deferred,
            executor_invalid_reason=reason,
            query_failure_reason=str(query.failure_reason or ""),
            current_planning_cost=float(query.planning_cost),
            planning_cost_relative_change=relative,
            public_target_shift=shift,
        )


__all__ = ("ExecutionContinuityEventDetector",)
