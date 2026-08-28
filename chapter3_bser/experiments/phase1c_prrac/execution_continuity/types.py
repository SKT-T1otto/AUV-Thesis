"""Immutable contracts for the opt-in PRRAC execution-continuity overlay."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


Vector3 = Tuple[float, float, float]


class ExecutionVariant(str, Enum):
    B0_LEGACY_V2_1 = "B0_LEGACY_V2_1"
    B1_ATOMIC_LAST_VALID = "B1_ATOMIC_LAST_VALID"
    B2_REACHABLE_PROXY = "B2_REACHABLE_PROXY"
    B3_PROXY_SAFE_SUPPRESSION = "B3_PROXY_SAFE_SUPPRESSION"


class NavigationMode(str, Enum):
    EXACT_PUBLIC_TARGET = "EXACT_PUBLIC_TARGET"
    REACHABLE_PUBLIC_PROXY = "REACHABLE_PUBLIC_PROXY"
    LAST_VALID_ROUTE = "LAST_VALID_ROUTE"
    SAFE_HOLD = "SAFE_HOLD"


@dataclass(frozen=True)
class ExecutionNavigationPlanV3:
    schema: str
    variant: ExecutionVariant
    semantic_target: Vector3
    navigation_endpoint: Vector3
    navigation_mode: NavigationMode
    reachable: bool
    path: Tuple[Vector3, ...]
    path_cell_indices: Tuple[int, ...]
    planning_cost: float
    estimated_arrival_time: float
    source: str
    proxy_distance_to_semantic_target: Optional[float]
    preserved_from_previous: bool
    safe_hold: bool
    failure_reason: Optional[str]
    exact_public_target_reachable: bool
    exact_public_target_unreachable: bool
    proxy_attempted: bool


@dataclass(frozen=True)
class ExecutionContinuityDetectionV3:
    events: Tuple[str, ...]
    route_invalid: bool
    retry_due: bool
    validity_evaluated: bool
    validity_deferred: bool
    executor_invalid_reason: str
    query_failure_reason: str
    current_planning_cost: Optional[float]
    planning_cost_relative_change: float
    public_target_shift: float


@dataclass(frozen=True)
class ResidualSuppressionDiagnostics:
    suppressed: bool
    executor_id: int
    raw_norm: float
    applied_norm: float
    reason: str


__all__ = (
    "ExecutionContinuityDetectionV3",
    "ExecutionNavigationPlanV3",
    "ExecutionVariant",
    "NavigationMode",
    "ResidualSuppressionDiagnostics",
    "Vector3",
)
