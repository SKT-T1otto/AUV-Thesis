"""S2-A.1 recovery planning using only immutable public planning views."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from typing import Iterable, Optional

import numpy as np

from core.mapping.planning_state import PlanningStateView
from core.mapping.travel_cost_service import TravelCostService

from .planner import _plan, _vector3
from .types_v2 import (
    LastCollisionFreeState,
    LocalConnectorCandidate,
    LocalConnectorPlan,
    PublicSegmentAudit,
    RecoveryPlanningAudit,
)


EPSILON = 1e-9


def canonical_path_hash(path: Iterable[Iterable[float]]) -> str:
    """Return a stable cross-process SHA-256 for a finite three-dimensional path."""

    canonical = []
    for point in path:
        values = np.asarray(tuple(point), dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(values)):
            raise ValueError("canonical paths must contain only finite values")
        canonical.append([float(f"{value:.12g}") for value in values])
    encoded = json.dumps(canonical, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _bounds(state: PlanningStateView) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(state.grid.origin, dtype=np.float64)
    upper = lower + np.asarray(state.grid.shape, dtype=np.float64) * np.asarray(state.grid.spacing, dtype=np.float64)
    return lower, upper


def _in_bounds(state: PlanningStateView, point: np.ndarray) -> bool:
    lower, upper = _bounds(state)
    return bool(np.all(point >= lower - EPSILON) and np.all(point <= upper + EPSILON))


def public_cell_index(state: PlanningStateView, point: Iterable[float]) -> Optional[int]:
    value = np.asarray(tuple(point), dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(value)) or not _in_bounds(state, value):
        return None
    centers = np.asarray(state.grid.cell_centers, dtype=np.float64).reshape(-1, 3)
    if not centers.size:
        return None
    return int(np.argmin(np.einsum("ij,ij->i", centers - value, centers - value)))


def audit_public_segment(
    planning_state: PlanningStateView,
    start: Iterable[float],
    endpoint: Iterable[float],
) -> PublicSegmentAudit:
    """Audit a straight connector against public occupancy only."""

    start_value = np.asarray(_vector3(start), dtype=np.float64)
    endpoint_value = np.asarray(_vector3(endpoint), dtype=np.float64)
    distance = float(np.linalg.norm(endpoint_value - start_value))
    spacing = np.asarray(planning_state.grid.spacing, dtype=np.float64)
    positive_spacing = spacing[spacing > EPSILON]
    if not positive_spacing.size:
        raise ValueError("public grid must expose at least one positive spacing")
    sample_spacing = 0.5 * float(np.min(positive_spacing))
    sample_count = max(2, int(math.ceil(distance / sample_spacing)) + 1)
    points = np.linspace(start_value, endpoint_value, sample_count, dtype=np.float64)
    occupied = np.asarray(planning_state.occupancy.occupied_mask, dtype=np.bool_).reshape(-1)
    unknown = np.asarray(planning_state.occupancy.unknown_mask, dtype=np.bool_).reshape(-1)
    free = np.asarray(planning_state.occupancy.free_mask, dtype=np.bool_).reshape(-1)
    occupied_count = unknown_count = free_count = 0
    reason: Optional[str] = None
    if not _in_bounds(planning_state, endpoint_value):
        reason = "LOCAL_CONNECTOR_ENDPOINT_OUT_OF_BOUNDS"
    endpoint_index = public_cell_index(planning_state, endpoint_value)
    if reason is None and (endpoint_index is None or bool(occupied[endpoint_index])):
        reason = "LOCAL_CONNECTOR_ENDPOINT_OCCUPIED"
    for sample_index, point in enumerate(points):
        cell_index = public_cell_index(planning_state, point)
        if cell_index is None:
            if reason is None:
                reason = "LOCAL_CONNECTOR_SEGMENT_OUT_OF_BOUNDS"
            continue
        is_occupied = bool(occupied[cell_index])
        # Only the first sample may occupy the collision cell.
        if is_occupied and sample_index != 0:
            occupied_count += 1
            if reason is None:
                reason = "LOCAL_CONNECTOR_NO_PUBLIC_SEGMENT"
        elif bool(unknown[cell_index]):
            unknown_count += 1
        elif bool(free[cell_index]):
            free_count += 1
    return PublicSegmentAudit(
        start=_vector3(start_value),
        endpoint=_vector3(endpoint_value),
        sampled_path=tuple(_vector3(point) for point in points),
        sample_count=int(sample_count),
        occupied_sample_count=int(occupied_count),
        unknown_sample_count=int(unknown_count),
        known_free_sample_count=int(free_count),
        known_free_fraction=float(free_count / sample_count),
        accepted=reason is None,
        rejection_reason=reason,
    )


def _distance_to_mask(centers: np.ndarray, point: np.ndarray, mask: np.ndarray) -> Optional[float]:
    selected = centers[np.asarray(mask, dtype=np.bool_)]
    return None if not selected.size else float(np.min(np.linalg.norm(selected - point, axis=1)))


def _start_connector_count(state: PlanningStateView, agent_id: int) -> int:
    agent = {int(item.agent_id): item for item in state.agents}[int(agent_id)]
    point = np.asarray(agent.position, dtype=np.float64)
    role_class = "executor" if agent.role.lower().startswith("exec") else "searcher"
    for endpoint in state.planning_graph.endpoint_connectors:
        endpoint_class = "executor" if endpoint.role.lower().startswith("exec") else "searcher"
        if endpoint_class == role_class and np.allclose(endpoint.point, point, atol=1e-12, rtol=0.0):
            return len(endpoint.connectors)
    index = public_cell_index(state, point)
    if index is None:
        return 0
    center = np.asarray(state.grid.cell_centers, dtype=np.float64).reshape(-1, 3)[index]
    valid = np.asarray(state.planning_graph.valid_mask, dtype=np.bool_).reshape(-1)
    labels = np.asarray(state.planning_graph.component_labels, dtype=np.int64).reshape(-1)
    return int(np.allclose(center, point, atol=1e-12, rtol=0.0) and valid[index] and labels[index] >= 0)


def _audit_base(
    state: PlanningStateView,
    agent_id: int,
    semantic_waypoint: Iterable[float],
    base_tracking_waypoint: Iterable[float],
    last_safe: LastCollisionFreeState | None,
    *,
    attempt_id: int,
    stage: str,
    forced_refresh: bool,
) -> RecoveryPlanningAudit:
    agent = {int(item.agent_id): item for item in state.agents}[int(agent_id)]
    current = np.asarray(agent.position, dtype=np.float64)
    centers = np.asarray(state.grid.cell_centers, dtype=np.float64).reshape(-1, 3)
    valid = np.asarray(state.planning_graph.valid_mask, dtype=np.bool_).reshape(-1)
    known = np.asarray(state.occupancy.known_mask, dtype=np.bool_).reshape(-1)
    free = np.asarray(state.occupancy.free_mask, dtype=np.bool_).reshape(-1)
    occupied = np.asarray(state.occupancy.occupied_mask, dtype=np.bool_).reshape(-1)
    probability = np.asarray(state.occupancy.occupancy_probability, dtype=np.float64).reshape(-1)
    current_index = public_cell_index(state, current)
    return RecoveryPlanningAudit(
        planning_state_step=int(state.step), planning_state_revision=int(state.map_revision),
        forced_public_refresh=bool(forced_refresh), current_position=_vector3(current),
        last_collision_free_position=None if last_safe is None else last_safe.position,
        semantic_waypoint=_vector3(semantic_waypoint), base_tracking_waypoint=_vector3(base_tracking_waypoint),
        current_cell_index=current_index,
        current_cell_valid=False if current_index is None else bool(valid[current_index]),
        current_cell_known=False if current_index is None else bool(known[current_index]),
        current_cell_free=False if current_index is None else bool(free[current_index]),
        current_cell_occupied=False if current_index is None else bool(occupied[current_index]),
        current_cell_occupancy_probability=None if current_index is None else float(probability[current_index]),
        start_connector_count=_start_connector_count(state, agent_id), reachable_cell_count=0,
        valid_cell_count=int(np.count_nonzero(valid)), known_free_cell_count=int(np.count_nonzero(free)),
        valid_known_free_cell_count=int(np.count_nonzero(valid & free)),
        valid_nonoccupied_cell_count=int(np.count_nonzero(valid & ~occupied)),
        local_candidate_count_before_filter=0, local_candidate_count_after_filter=0,
        rejected_same_position_count=0, rejected_occupied_segment_count=0,
        rejected_out_of_bounds_count=0, rejected_failed_endpoint_count=0,
        nearest_valid_cell_distance=_distance_to_mask(centers, current, valid),
        nearest_known_free_cell_distance=_distance_to_mask(centers, current, free),
        nearest_nonoccupied_cell_distance=_distance_to_mask(centers, current, ~occupied),
        travel_cost_failure_reason=None, final_failure_stage=None, final_failure_reason=None,
        agent_id=int(agent_id), attempt_id=int(attempt_id), planning_stage=str(stage),
    )


def plan_forced_route_refresh(
    state: PlanningStateView,
    agent_id: int,
    candidate_id: str,
    semantic_waypoint: Iterable[float],
    base_tracking_waypoint: Iterable[float],
    last_safe: LastCollisionFreeState | None,
    *,
    attempt_id: int,
    service: TravelCostService | None = None,
):
    audit = _audit_base(state, agent_id, semantic_waypoint, base_tracking_waypoint, last_safe,
                        attempt_id=attempt_id, stage="ROUTE_REFRESH", forced_refresh=True)
    agent = {int(item.agent_id): item for item in state.agents}[int(agent_id)]
    planner = service or TravelCostService(state)
    query = planner.query(agent.position, semantic_waypoint, agent)
    reachable_count = 0
    try:
        reachable_count = int(np.count_nonzero(planner.single_source(agent.position, agent).reachable_mask))
    except Exception:
        reachable_count = 0
    if not query.reachable:
        raw = str(query.failure_reason or "query_unreachable")
        reason = "ROUTE_REFRESH_START_CONNECTOR_EMPTY" if raw in {"no_start_connector", "invalid_start"} else "ROUTE_REFRESH_QUERY_UNREACHABLE"
        return None, replace(audit, reachable_cell_count=reachable_count,
                             travel_cost_failure_reason=raw, final_failure_stage="ROUTE_REFRESH", final_failure_reason=reason), reason
    plan = _plan(candidate_id, semantic_waypoint, semantic_waypoint, query,
                 source="SEARCH_COLLISION_FORCED_ROUTE_REFRESH_V2", cell_index=None)
    return plan, replace(audit, reachable_cell_count=reachable_count), None


def _candidate(
    state: PlanningStateView,
    start: np.ndarray,
    endpoint: np.ndarray,
    *,
    tier: int,
    source: str,
    cell_index: int | None,
    failed_unit: np.ndarray,
    last_safe_position: np.ndarray | None,
    semantic: np.ndarray,
) -> LocalConnectorCandidate | None:
    segment = audit_public_segment(state, start, endpoint)
    if not segment.accepted:
        return None
    centers = np.asarray(state.grid.cell_centers, dtype=np.float64).reshape(-1, 3)
    occupied = np.asarray(state.occupancy.occupied_mask, dtype=np.bool_).reshape(-1)
    probability = np.asarray(state.occupancy.occupancy_probability, dtype=np.float64).reshape(-1)
    index = public_cell_index(state, endpoint) if cell_index is None else int(cell_index)
    occupied_centers = centers[occupied]
    clearance = 0.0 if not occupied_centers.size else float(np.min(np.linalg.norm(occupied_centers - endpoint, axis=1)))
    return LocalConnectorCandidate(
        endpoint=_vector3(endpoint), endpoint_cell_index=index, endpoint_tier=int(tier), source=source,
        segment_audit=segment, public_obstacle_clearance=clearance,
        failed_direction_projection=float(np.dot(endpoint - start, failed_unit)),
        occupancy_risk=0.0 if index is None else float(probability[index]),
        last_safe_distance=0.0 if last_safe_position is None else float(np.linalg.norm(endpoint - last_safe_position)),
        semantic_waypoint_distance=float(np.linalg.norm(endpoint - semantic)),
        local_path_length=float(np.linalg.norm(endpoint - start)), unknown_fallback=tier == 2,
    )


def _ranking(candidate: LocalConnectorCandidate):
    return (
        candidate.segment_audit.occupied_sample_count,
        -candidate.segment_audit.known_free_fraction,
        -candidate.public_obstacle_clearance,
        candidate.failed_direction_projection,
        candidate.occupancy_risk,
        candidate.last_safe_distance,
        candidate.semantic_waypoint_distance,
        candidate.local_path_length,
        math.inf if candidate.endpoint_cell_index is None else candidate.endpoint_cell_index,
    )


def plan_local_connector(
    state: PlanningStateView,
    agent_id: int,
    candidate_id: str,
    semantic_waypoint: Iterable[float],
    base_tracking_waypoint: Iterable[float],
    base_planned_path: Iterable[Iterable[float]],
    last_safe: LastCollisionFreeState | None,
    *,
    failed_endpoint_cells: Iterable[int] = (),
    failed_direction: Iterable[float] | None = None,
    attempt_id: int,
):
    """Build a local continuous-space connector without querying from collision position."""

    audit = _audit_base(state, agent_id, semantic_waypoint, base_tracking_waypoint, last_safe,
                        attempt_id=attempt_id, stage="LOCAL_CONNECTOR", forced_refresh=True)
    agent = {int(item.agent_id): item for item in state.agents}[int(agent_id)]
    start = np.asarray(agent.position, dtype=np.float64)
    semantic = np.asarray(_vector3(semantic_waypoint), dtype=np.float64)
    base_tracking = np.asarray(_vector3(base_tracking_waypoint), dtype=np.float64)
    centers = np.asarray(state.grid.cell_centers, dtype=np.float64).reshape(-1, 3)
    valid = np.asarray(state.planning_graph.valid_mask, dtype=np.bool_).reshape(-1)
    free = np.asarray(state.occupancy.free_mask, dtype=np.bool_).reshape(-1)
    occupied = np.asarray(state.occupancy.occupied_mask, dtype=np.bool_).reshape(-1)
    failed = {int(value) for value in failed_endpoint_cells}
    last_safe_position = None if last_safe is None else np.asarray(last_safe.position, dtype=np.float64)
    direction = np.zeros(3, dtype=np.float64) if failed_direction is None else np.asarray(tuple(failed_direction), dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm <= EPSILON:
        direction = np.asarray(agent.velocity, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
    failed_unit = direction / norm if norm > EPSILON else np.zeros(3, dtype=np.float64)
    counters = {"before": 0, "same": 0, "occupied": 0, "bounds": 0, "failed": 0}

    def accept(endpoint: np.ndarray, tier: int, source: str, cell_index: int | None):
        counters["before"] += 1
        if not np.all(np.isfinite(endpoint)) or not _in_bounds(state, endpoint):
            counters["bounds"] += 1; return None
        if float(np.linalg.norm(endpoint - start)) <= EPSILON or float(np.linalg.norm(endpoint - base_tracking)) <= EPSILON:
            counters["same"] += 1; return None
        resolved = public_cell_index(state, endpoint) if cell_index is None else int(cell_index)
        if resolved is not None and resolved in failed:
            counters["failed"] += 1; return None
        value = _candidate(state, start, endpoint, tier=tier, source=source, cell_index=resolved,
                           failed_unit=failed_unit, last_safe_position=last_safe_position, semantic=semantic)
        if value is None:
            counters["occupied"] += 1
        return value

    tiers: list[tuple[int, list[LocalConnectorCandidate]]] = []
    tier0: list[LocalConnectorCandidate] = []
    if last_safe_position is not None:
        value = accept(last_safe_position, 0, "LAST_COLLISION_FREE_ANCHOR", last_safe.cell_index)
        if value is not None: tier0.append(value)
    tiers.append((0, tier0))
    tier1 = [value for index in np.flatnonzero(valid & free)
             if (value := accept(centers[int(index)], 1, "KNOWN_FREE_GRAPH_CELL", int(index))) is not None]
    tiers.append((1, tier1))
    if not tier0 and not tier1:
        tier2 = [value for index in np.flatnonzero(valid & ~occupied)
                 if (value := accept(centers[int(index)], 2, "LOCAL_CONNECTOR_UNKNOWN_FALLBACK", int(index))) is not None]
        tiers.append((2, tier2))
    if not any(values for _, values in tiers):
        if norm > EPSILON:
            directions = (-failed_unit,)
        else:
            directions = tuple(np.asarray(value, dtype=np.float64) for value in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)))
        lower, upper = _bounds(state)
        spacing = np.asarray(state.grid.spacing, dtype=np.float64)
        step_length = float(np.min(spacing[spacing > EPSILON]))
        tier3 = []
        for reverse in directions:
            endpoint = np.clip(start + reverse * step_length, lower, upper)
            value = accept(endpoint, 3, "DETERMINISTIC_REVERSE_DIRECTION", None)
            if value is not None: tier3.append(value)
        tiers.append((3, tier3))
    selected_values = next((values for _, values in tiers if values), [])
    accepted_count = sum(len(values) for _, values in tiers)
    audit = replace(audit, local_candidate_count_before_filter=counters["before"],
                    local_candidate_count_after_filter=accepted_count,
                    rejected_same_position_count=counters["same"],
                    rejected_occupied_segment_count=counters["occupied"],
                    rejected_out_of_bounds_count=counters["bounds"],
                    rejected_failed_endpoint_count=counters["failed"])
    if not selected_values:
        if last_safe is None and not np.any(valid & ~occupied):
            reason = "LOCAL_CONNECTOR_NO_LAST_SAFE_POINT"
        elif not np.any(valid & ~occupied):
            reason = "LOCAL_CONNECTOR_NO_VALID_CELL"
        elif counters["occupied"]:
            reason = "LOCAL_CONNECTOR_NO_PUBLIC_SEGMENT"
        else:
            reason = "LOCAL_CONNECTOR_ALL_ENDPOINTS_FAILED"
        return None, replace(audit, final_failure_stage="LOCAL_CONNECTOR", final_failure_reason=reason), reason
    selected = min(selected_values, key=_ranking)
    base_hash = canonical_path_hash(base_planned_path)
    overlay_hash = canonical_path_hash(selected.segment_audit.sampled_path)
    plan = LocalConnectorPlan(
        semantic_candidate_id=str(candidate_id), semantic_waypoint=_vector3(semantic),
        local_endpoint=selected.endpoint, endpoint_cell_index=selected.endpoint_cell_index,
        endpoint_tier=selected.endpoint_tier, sampled_local_path=selected.segment_audit.sampled_path,
        public_segment_audit=selected.segment_audit, source=selected.source,
        failed_endpoint_history=tuple(sorted(failed)), base_path_hash=base_hash, overlay_path_hash=overlay_hash,
    )
    return plan, replace(audit, selected_tier=selected.endpoint_tier,
                         selected_endpoint=selected.endpoint, segment_audit=selected.segment_audit,
                         base_path_hash=base_hash, overlay_path_hash=overlay_hash), None


__all__ = (
    "EPSILON", "audit_public_segment", "canonical_path_hash", "plan_forced_route_refresh",
    "plan_local_connector", "public_cell_index",
)
