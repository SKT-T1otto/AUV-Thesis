"""Deterministic, public-information-only execution-continuity planning."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from core.mapping.planning_state import PlanningStateView
from core.mapping.travel_cost_service import PathQueryResult, TravelCostService

from .config import OVERLAY_SCHEMA, parse_execution_variant
from .types import ExecutionNavigationPlanV3, ExecutionVariant, NavigationMode, Vector3


def _vector3(value: Iterable[float]) -> Vector3:
    result = tuple(float(item) for item in value)
    if len(result) != 3 or not np.all(np.isfinite(result)):
        raise ValueError("execution-continuity coordinates must be finite 3D values")
    return result


def _path(value: np.ndarray) -> tuple[Vector3, ...]:
    array = np.asarray(value, dtype=np.float64).reshape(-1, 3)
    return tuple(_vector3(row) for row in array)


def _plan_from_query(
    *,
    variant: ExecutionVariant,
    semantic_target: Vector3,
    endpoint: Vector3,
    mode: NavigationMode,
    query: PathQueryResult,
    source: str,
    preserved: bool,
    exact_reachable: bool,
    exact_unreachable: bool,
    proxy_attempted: bool,
) -> ExecutionNavigationPlanV3:
    if not query.reachable:
        raise ValueError("reachable execution plan requires a reachable path query")
    proxy_distance = (
        float(np.linalg.norm(np.asarray(endpoint) - np.asarray(semantic_target)))
        if mode is NavigationMode.REACHABLE_PUBLIC_PROXY
        else None
    )
    return ExecutionNavigationPlanV3(
        schema=OVERLAY_SCHEMA,
        variant=variant,
        semantic_target=semantic_target,
        navigation_endpoint=endpoint,
        navigation_mode=mode,
        reachable=True,
        path=_path(query.path_points),
        path_cell_indices=tuple(int(item) for item in query.path_cell_indices),
        planning_cost=float(query.planning_cost),
        estimated_arrival_time=float(query.physical_travel_time),
        source=source,
        proxy_distance_to_semantic_target=proxy_distance,
        preserved_from_previous=bool(preserved),
        safe_hold=False,
        failure_reason=None,
        exact_public_target_reachable=bool(exact_reachable),
        exact_public_target_unreachable=bool(exact_unreachable),
        proxy_attempted=bool(proxy_attempted),
    )


def assign_reachable_public_proxy(
    state: PlanningStateView,
    semantic_target: Iterable[float],
    *,
    service: TravelCostService | None = None,
) -> tuple[Vector3, PathQueryResult, int] | None:
    """Return the deterministic best reachable grid center with strict progress."""

    target = np.asarray(_vector3(semantic_target), dtype=np.float64)
    executor = state.agents[state.executor_id]
    start = np.asarray(executor.position, dtype=np.float64)
    current_distance = float(np.linalg.norm(start - target))
    centers = np.asarray(state.grid.cell_centers, dtype=np.float64).reshape(-1, 3)
    valid = np.asarray(state.planning_graph.valid_mask, dtype=np.bool_).reshape(-1)
    finite = np.all(np.isfinite(centers), axis=1)
    distances = np.linalg.norm(centers - target, axis=1)
    progress = distances < current_distance - 1e-9
    candidates = np.flatnonzero(valid & finite & progress)
    if candidates.size == 0:
        return None

    planner = service or TravelCostService(state)
    single = planner.single_source(executor.position, executor)
    reachable = np.asarray(single.reachable_mask, dtype=np.bool_)
    costs = np.asarray(single.planning_cost_by_cell, dtype=np.float64)
    candidates = candidates[reachable[candidates] & np.isfinite(costs[candidates])]
    if candidates.size == 0:
        return None
    ordered = sorted(
        (int(index) for index in candidates),
        key=lambda index: (float(distances[index]), float(costs[index]), index),
    )
    for index in ordered:
        endpoint = _vector3(centers[index])
        query = planner.query(executor.position, endpoint, executor)
        if query.reachable:
            return endpoint, query, index
    return None


def _safe_hold(
    *,
    variant: ExecutionVariant,
    semantic_target: Vector3,
    hold_position: Vector3,
    exact_unreachable: bool,
    proxy_attempted: bool,
    failure_reason: str,
) -> ExecutionNavigationPlanV3:
    return ExecutionNavigationPlanV3(
        schema=OVERLAY_SCHEMA,
        variant=variant,
        semantic_target=semantic_target,
        navigation_endpoint=hold_position,
        navigation_mode=NavigationMode.SAFE_HOLD,
        reachable=False,
        path=(),
        path_cell_indices=(),
        planning_cost=math.inf,
        estimated_arrival_time=math.inf,
        source="EXECUTION_CONTINUITY_SAFE_HOLD",
        proxy_distance_to_semantic_target=None,
        preserved_from_previous=False,
        safe_hold=True,
        failure_reason=str(failure_reason),
        exact_public_target_reachable=False,
        exact_public_target_unreachable=bool(exact_unreachable),
        proxy_attempted=bool(proxy_attempted),
    )


def plan_atomic_execution_continuity(
    state: PlanningStateView,
    semantic_target: Iterable[float],
    previous_plan: ExecutionNavigationPlanV3 | None,
    variant: str | ExecutionVariant,
    *,
    service: TravelCostService | None = None,
) -> ExecutionNavigationPlanV3:
    """Atomically choose exact, proxy, refreshed last-valid, or explicit hold."""

    selected = parse_execution_variant(variant)
    if selected is ExecutionVariant.B0_LEGACY_V2_1:
        raise ValueError("B0 must use the unmodified legacy controller")
    target = _vector3(semantic_target)
    executor = state.agents[state.executor_id]
    hold_position = _vector3(executor.position)
    planner = service or TravelCostService(state)
    exact = planner.query(executor.position, target, executor)
    if exact.reachable:
        return _plan_from_query(
            variant=selected,
            semantic_target=target,
            endpoint=target,
            mode=NavigationMode.EXACT_PUBLIC_TARGET,
            query=exact,
            source="EXECUTION_CONTINUITY_EXACT_PUBLIC_TARGET",
            preserved=False,
            exact_reachable=True,
            exact_unreachable=False,
            proxy_attempted=False,
        )

    proxy_attempted = selected in {
        ExecutionVariant.B2_REACHABLE_PROXY,
        ExecutionVariant.B3_PROXY_SAFE_SUPPRESSION,
    }
    if proxy_attempted:
        proxy = assign_reachable_public_proxy(state, target, service=planner)
        if proxy is not None:
            endpoint, query, index = proxy
            return _plan_from_query(
                variant=selected,
                semantic_target=target,
                endpoint=endpoint,
                mode=NavigationMode.REACHABLE_PUBLIC_PROXY,
                query=query,
                source=f"EXECUTION_CONTINUITY_REACHABLE_PROXY:{index}",
                preserved=False,
                exact_reachable=False,
                exact_unreachable=True,
                proxy_attempted=True,
            )

    if (
        previous_plan is not None
        and previous_plan.navigation_mode
        in {
            NavigationMode.EXACT_PUBLIC_TARGET,
            NavigationMode.REACHABLE_PUBLIC_PROXY,
            NavigationMode.LAST_VALID_ROUTE,
        }
    ):
        endpoint = _vector3(previous_plan.navigation_endpoint)
        refreshed = planner.query(executor.position, endpoint, executor)
        if refreshed.reachable:
            return _plan_from_query(
                variant=selected,
                semantic_target=target,
                endpoint=endpoint,
                mode=NavigationMode.LAST_VALID_ROUTE,
                query=refreshed,
                source="EXECUTION_CONTINUITY_LAST_VALID_ROUTE",
                preserved=True,
                exact_reachable=False,
                exact_unreachable=True,
                proxy_attempted=proxy_attempted,
            )

    return _safe_hold(
        variant=selected,
        semantic_target=target,
        hold_position=hold_position,
        exact_unreachable=True,
        proxy_attempted=proxy_attempted,
        failure_reason=str(exact.failure_reason or "NO_REACHABLE_EXECUTION_ROUTE"),
    )


__all__ = (
    "assign_reachable_public_proxy",
    "plan_atomic_execution_continuity",
)
