"""Immutable contracts for the evaluation-only search collision recovery overlay."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


Vector3 = Tuple[float, float, float]


class SearchRecoveryVariant(str, Enum):
    S2A_C0_BASELINE = "S2A_C0_BASELINE"
    S2A_C1_ROUTE_REFRESH = "S2A_C1_ROUTE_REFRESH"
    S2A_C2_EGRESS_ROUTE = "S2A_C2_EGRESS_ROUTE"


class SearchRecoveryMode(str, Enum):
    NORMAL_SEARCH = "NORMAL_SEARCH"
    COLLISION_EDGE_DETECTED = "COLLISION_EDGE_DETECTED"
    ROUTE_REFRESH = "ROUTE_REFRESH"
    EGRESS_ROUTE = "EGRESS_ROUTE"
    REJOIN_SEARCH = "REJOIN_SEARCH"
    RECOVERY_NO_EGRESS = "RECOVERY_NO_EGRESS"


@dataclass(frozen=True)
class RecoveryNavigationPlan:
    semantic_search_candidate_id: str
    semantic_search_waypoint: Vector3
    navigation_endpoint: Vector3
    path: Tuple[Vector3, ...]
    path_cell_indices: Tuple[int, ...]
    planning_cost: float
    physical_travel_time: float
    endpoint_cell_index: Optional[int]
    source: str


@dataclass(frozen=True)
class RecoveryStepSnapshot:
    active_agent_ids: Tuple[int, ...]
    mode_by_agent: dict[int, str]
    attempt_id_by_agent: dict[int, int]
    collision_streak_by_agent: dict[int, int]
    semantic_candidate_id_by_agent: dict[int, str | None]
    semantic_waypoint_by_agent: dict[int, Vector3 | None]
    navigation_endpoint_by_agent: dict[int, Vector3 | None]
    endpoint_cell_index_by_agent: dict[int, int | None]
    trigger_reason_by_agent: dict[int, str | None]
    failure_reason_by_agent: dict[int, str | None]
    route_refresh_attempted_by_agent: dict[int, bool]
    egress_attempted_by_agent: dict[int, bool]


VARIANT_ORDER = tuple(SearchRecoveryVariant)


__all__ = (
    "RecoveryNavigationPlan",
    "RecoveryStepSnapshot",
    "SearchRecoveryMode",
    "SearchRecoveryVariant",
    "VARIANT_ORDER",
    "Vector3",
)
