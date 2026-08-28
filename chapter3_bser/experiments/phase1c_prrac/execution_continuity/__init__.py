"""Opt-in PRRAC execution-continuity ablation runtime."""

from .action_adapter import ExecutionContinuityActionAdapter
from .config import (
    CHECKPOINT_RUNTIME_REVISION,
    EXECUTION_ABLATION_SCHEMA,
    OVERLAY_RUNTIME_REVISION,
    OVERLAY_SCHEMA,
    VARIANT_ORDER,
    overlay_config,
    overlay_enabled,
    parse_execution_variant,
)
from .controller import ExecutionContinuityController
from .diagnostics import (
    ExecutionContinuityDiagnostics,
    aggregate_execution_variant,
    paired_execution_variant_comparisons,
)
from .event_detector import ExecutionContinuityEventDetector
from .planner import assign_reachable_public_proxy, plan_atomic_execution_continuity
from .types import (
    ExecutionContinuityDetectionV3,
    ExecutionNavigationPlanV3,
    ExecutionVariant,
    NavigationMode,
    ResidualSuppressionDiagnostics,
)

__all__ = (
    "CHECKPOINT_RUNTIME_REVISION",
    "EXECUTION_ABLATION_SCHEMA",
    "OVERLAY_RUNTIME_REVISION",
    "OVERLAY_SCHEMA",
    "VARIANT_ORDER",
    "ExecutionContinuityActionAdapter",
    "ExecutionContinuityController",
    "ExecutionContinuityDetectionV3",
    "ExecutionContinuityDiagnostics",
    "ExecutionContinuityEventDetector",
    "ExecutionNavigationPlanV3",
    "ExecutionVariant",
    "NavigationMode",
    "ResidualSuppressionDiagnostics",
    "aggregate_execution_variant",
    "assign_reachable_public_proxy",
    "overlay_config",
    "overlay_enabled",
    "paired_execution_variant_comparisons",
    "parse_execution_variant",
    "plan_atomic_execution_continuity",
)
