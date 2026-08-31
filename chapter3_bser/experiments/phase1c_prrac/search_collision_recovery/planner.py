"""Deterministic recovery planning over public immutable planning views only."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from core.mapping.planning_state import PlanningStateView
from core.mapping.travel_cost_service import TravelCostService

from .types import RecoveryNavigationPlan


def _vector3(value: Iterable[float]) -> tuple[float, float, float]:
    result = tuple(float(item) for item in value)
    if len(result) != 3 or not np.all(np.isfinite(result)):
        raise ValueError("recovery coordinates must be finite three-dimensional values")
    return result


def _plan(candidate_id: str, semantic_waypoint: Iterable[float], endpoint: Iterable[float], query, *, source: str, cell_index: int | None) -> RecoveryNavigationPlan:
    if not query.reachable:
        raise ValueError("a recovery plan requires a reachable public path query")
    return RecoveryNavigationPlan(
        semantic_search_candidate_id=str(candidate_id),
        semantic_search_waypoint=_vector3(semantic_waypoint),
        navigation_endpoint=_vector3(endpoint),
        path=tuple(_vector3(point) for point in np.asarray(query.path_points).reshape(-1, 3)),
        path_cell_indices=tuple(int(item) for item in np.asarray(query.path_cell_indices).reshape(-1)),
        planning_cost=float(query.planning_cost),
        physical_travel_time=float(query.physical_travel_time),
        endpoint_cell_index=None if cell_index is None else int(cell_index),
        source=str(source),
    )


def plan_route_refresh(state: PlanningStateView, agent_id: int, candidate_id: str, semantic_waypoint: Iterable[float], *, service: TravelCostService | None = None):
    agent = {int(item.agent_id): item for item in state.agents}[int(agent_id)]
    planner = service or TravelCostService(state)
    query = planner.query(agent.position, semantic_waypoint, agent)
    if not query.reachable:
        return None, str(query.failure_reason or "ROUTE_REFRESH_UNREACHABLE")
    return _plan(candidate_id, semantic_waypoint, semantic_waypoint, query, source="SEARCH_COLLISION_ROUTE_REFRESH", cell_index=None), None


def _hop_counts(predecessor: np.ndarray, reachable: np.ndarray) -> np.ndarray:
    hops = np.full(predecessor.shape, np.iinfo(np.int32).max, dtype=np.int64)
    for index in np.flatnonzero(reachable):
        seen: set[int] = set()
        cursor = int(index)
        count = 0
        while int(predecessor[cursor]) >= 0 and cursor not in seen:
            seen.add(cursor)
            cursor = int(predecessor[cursor])
            count += 1
        if cursor not in seen:
            hops[index] = count
    return hops


def select_egress_route(
    state: PlanningStateView,
    agent_id: int,
    candidate_id: str,
    semantic_waypoint: Iterable[float],
    *,
    failed_endpoint_cells: Iterable[int] = (),
    failed_direction: Iterable[float] | None = None,
    service: TravelCostService | None = None,
):
    """Choose the first reachable known-free endpoint under the frozen ordering."""

    agent = {int(item.agent_id): item for item in state.agents}[int(agent_id)]
    planner = service or TravelCostService(state)
    single = planner.single_source(agent.position, agent)
    centers = np.asarray(state.grid.cell_centers, dtype=np.float64).reshape(-1, 3)
    valid = np.asarray(state.planning_graph.valid_mask, dtype=np.bool_).reshape(-1)
    free = np.asarray(state.occupancy.free_mask, dtype=np.bool_).reshape(-1)
    occupied = np.asarray(state.occupancy.occupied_mask, dtype=np.bool_).reshape(-1)
    reachable = np.asarray(single.reachable_mask, dtype=np.bool_).reshape(-1)
    costs = np.asarray(single.planning_cost_by_cell, dtype=np.float64).reshape(-1)
    predecessor = np.asarray(single.predecessor_by_cell, dtype=np.int64).reshape(-1)
    hops = _hop_counts(predecessor, reachable)
    start = np.asarray(agent.position, dtype=np.float64)
    semantic = np.asarray(_vector3(semantic_waypoint), dtype=np.float64)
    distances_from_start = np.linalg.norm(centers - start, axis=1)
    finite = np.all(np.isfinite(centers), axis=1) & np.isfinite(costs)
    allowed = valid & free & reachable & finite & (hops > 0) & (distances_from_start > 1e-12)
    failed = {int(value) for value in failed_endpoint_cells}
    if failed:
        allowed[list(index for index in failed if 0 <= index < allowed.size)] = False
    occupied_centers = centers[occupied & np.all(np.isfinite(centers), axis=1)]
    if occupied_centers.size:
        clearance = np.min(np.linalg.norm(centers[:, None, :] - occupied_centers[None, :, :], axis=2), axis=1)
    else:
        clearance = np.zeros(centers.shape[0], dtype=np.float64)
    occupancy_risk = np.asarray(state.occupancy.occupancy_probability, dtype=np.float64).reshape(-1)
    direction = np.zeros(3, dtype=np.float64) if failed_direction is None else np.asarray(tuple(failed_direction), dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        direction = np.asarray(agent.velocity, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
    direction = direction / norm if norm > 1e-12 else np.zeros(3, dtype=np.float64)
    projection = (centers - start) @ direction
    semantic_distance = np.linalg.norm(centers - semantic, axis=1)
    ordered = sorted(
        (int(index) for index in np.flatnonzero(allowed)),
        key=lambda index: (
            int(hops[index]),
            -float(clearance[index]),
            float(occupancy_risk[index]),
            float(projection[index]),
            float(semantic_distance[index]),
            float(costs[index]),
            index,
        ),
    )
    for index in ordered:
        endpoint = _vector3(centers[index])
        query = planner.query(agent.position, endpoint, agent)
        if query.reachable:
            return _plan(candidate_id, semantic_waypoint, endpoint, query, source="SEARCH_COLLISION_EGRESS_ROUTE", cell_index=index), None
    return None, "RECOVERY_NO_EGRESS"


__all__ = ("plan_route_refresh", "select_egress_route")
