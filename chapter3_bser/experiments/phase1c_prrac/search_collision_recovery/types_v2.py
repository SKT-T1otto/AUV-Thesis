"""Immutable S2-A.1 public recovery and activation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple

from .types import Vector3


ACTIVATION_DIAGNOSTICS_SCHEMA = "bser.phase1c.prrac.search_collision_recovery.activation.v2"


class SearchRecoveryVariantV2(str, Enum):
    S2A1_C0_BASELINE = "S2A1_C0_BASELINE"
    S2A1_C1_FORCED_REFRESH = "S2A1_C1_FORCED_REFRESH"
    S2A1_C2_LOCAL_CONNECTOR = "S2A1_C2_LOCAL_CONNECTOR"


class RecoveryModeV2(str, Enum):
    NORMAL_SEARCH = "NORMAL_SEARCH"
    COLLISION_EDGE_DETECTED = "COLLISION_EDGE_DETECTED"
    PUBLIC_STATE_FORCE_REFRESH = "PUBLIC_STATE_FORCE_REFRESH"
    ROUTE_REFRESH = "ROUTE_REFRESH"
    ROUTE_REFRESH_FAILED = "ROUTE_REFRESH_FAILED"
    LOCAL_CONNECTOR_EGRESS = "LOCAL_CONNECTOR_EGRESS"
    LOCAL_CONNECTOR_FAILED = "LOCAL_CONNECTOR_FAILED"
    GRAPH_RECONNECT = "GRAPH_RECONNECT"
    REJOIN_SEARCH = "REJOIN_SEARCH"
    RECOVERY_FAILED_PASS_THROUGH = "RECOVERY_FAILED_PASS_THROUGH"


@dataclass(frozen=True)
class LastCollisionFreeState:
    position: Vector3
    velocity: Vector3
    step: int
    cell_index: Optional[int]
    tracking_waypoint: Optional[Vector3]


@dataclass(frozen=True)
class PublicSegmentAudit:
    start: Vector3
    endpoint: Vector3
    sampled_path: Tuple[Vector3, ...]
    sample_count: int
    occupied_sample_count: int
    unknown_sample_count: int
    known_free_sample_count: int
    known_free_fraction: float
    accepted: bool
    rejection_reason: Optional[str]


@dataclass(frozen=True)
class LocalConnectorCandidate:
    endpoint: Vector3
    endpoint_cell_index: Optional[int]
    endpoint_tier: int
    source: str
    segment_audit: PublicSegmentAudit
    public_obstacle_clearance: float
    failed_direction_projection: float
    occupancy_risk: float
    last_safe_distance: float
    semantic_waypoint_distance: float
    local_path_length: float
    unknown_fallback: bool = False


@dataclass(frozen=True)
class LocalConnectorPlan:
    semantic_candidate_id: str
    semantic_waypoint: Vector3
    local_endpoint: Vector3
    endpoint_cell_index: Optional[int]
    endpoint_tier: int
    sampled_local_path: Tuple[Vector3, ...]
    public_segment_audit: PublicSegmentAudit
    source: str
    failed_endpoint_history: Tuple[int, ...]
    base_path_hash: str
    overlay_path_hash: str

    @property
    def semantic_search_candidate_id(self) -> str:
        return self.semantic_candidate_id

    @property
    def semantic_search_waypoint(self) -> Vector3:
        return self.semantic_waypoint

    @property
    def navigation_endpoint(self) -> Vector3:
        return self.local_endpoint

    @property
    def path(self) -> Tuple[Vector3, ...]:
        return self.sampled_local_path


@dataclass(frozen=True)
class RecoveryPlanningAudit:
    planning_state_step: int
    planning_state_revision: int
    forced_public_refresh: bool
    current_position: Vector3
    last_collision_free_position: Optional[Vector3]
    semantic_waypoint: Vector3
    base_tracking_waypoint: Vector3
    current_cell_index: Optional[int]
    current_cell_valid: bool
    current_cell_known: bool
    current_cell_free: bool
    current_cell_occupied: bool
    current_cell_occupancy_probability: Optional[float]
    start_connector_count: int
    reachable_cell_count: int
    valid_cell_count: int
    known_free_cell_count: int
    valid_known_free_cell_count: int
    valid_nonoccupied_cell_count: int
    local_candidate_count_before_filter: int
    local_candidate_count_after_filter: int
    rejected_same_position_count: int
    rejected_occupied_segment_count: int
    rejected_out_of_bounds_count: int
    rejected_failed_endpoint_count: int
    nearest_valid_cell_distance: Optional[float]
    nearest_known_free_cell_distance: Optional[float]
    nearest_nonoccupied_cell_distance: Optional[float]
    travel_cost_failure_reason: Optional[str]
    final_failure_stage: Optional[str]
    final_failure_reason: Optional[str]
    agent_id: int = -1
    attempt_id: int = 0
    planning_stage: str = ""
    selected_tier: Optional[int] = None
    selected_endpoint: Optional[Vector3] = None
    segment_audit: Optional[PublicSegmentAudit] = None
    base_path_hash: str = ""
    overlay_path_hash: str = ""
    guidance_changed: bool = False

    def scalar_row(self) -> dict[str, Any]:
        row = dict(self.__dict__)
        segment = row.pop("segment_audit")
        row["segment_audit"] = None if segment is None else dict(segment.__dict__)
        return row


@dataclass(frozen=True)
class ActivationAuditStep:
    base_tracking_waypoint: Optional[Vector3]
    overlay_tracking_waypoint: Optional[Vector3]
    tracking_waypoint_delta_norm: float
    base_final_waypoint: Optional[Vector3]
    overlay_final_waypoint: Optional[Vector3]
    base_path_hash: str
    overlay_path_hash: str
    path_changed: bool
    guidance_changed: bool
    recovery_plan_installed: bool
    recovery_plan_source: Optional[str]
    recovery_endpoint_tier: Optional[int]
    recovery_endpoint_cell_index: Optional[int]
    recovery_effective_intervention: bool
    route_refresh_identical_to_base: bool


__all__ = tuple(name for name in globals() if not name.startswith("_"))
