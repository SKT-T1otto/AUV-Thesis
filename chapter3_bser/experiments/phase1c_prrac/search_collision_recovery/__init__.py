"""Public evaluation-only Searcher collision recovery API."""

from .aggregation import aggregate_search_collision_recovery, paired_search_collision_recovery_baseline_strata, paired_search_collision_recovery_comparisons, search_collision_recovery_failure_funnel, validate_s2a_baseline_regression
from .config import SEARCH_COLLISION_RECOVERY_SCHEMA, parse_search_recovery_variant, search_collision_recovery_config, search_collision_recovery_config_hash
from .controller import SearchCollisionRecoveryController, build_search_recovery_controller
from .detector import CollisionEdgeDetector
from .diagnostics import baseline_recovery_summary
from .guidance_overlay import apply_search_recovery_guidance
from .planner import plan_route_refresh, select_egress_route
from .types import RecoveryNavigationPlan, RecoveryStepSnapshot, SearchRecoveryMode, SearchRecoveryVariant, VARIANT_ORDER

__all__ = tuple(name for name in globals() if not name.startswith("_"))
