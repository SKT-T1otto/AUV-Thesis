"""Public evaluation-only Searcher collision recovery API."""

from .aggregation import aggregate_search_collision_recovery, paired_search_collision_recovery_baseline_strata, paired_search_collision_recovery_comparisons, search_collision_recovery_failure_funnel, validate_s2a_baseline_regression
from .config import SEARCH_COLLISION_RECOVERY_SCHEMA, SEARCH_COLLISION_RECOVERY_SCHEMA_V2, parse_search_recovery_variant, search_collision_recovery_config, search_collision_recovery_config_hash
from .controller import SearchCollisionRecoveryController, build_search_recovery_controller
from .detector import CollisionEdgeDetector
from .diagnostics import baseline_recovery_summary
from .guidance_overlay import apply_search_recovery_guidance
from .planner import plan_route_refresh, select_egress_route
from .planner_v2 import audit_public_segment, canonical_path_hash, plan_forced_route_refresh, plan_local_connector, public_cell_index
from .types import RecoveryNavigationPlan, RecoveryStepSnapshot, SearchRecoveryMode, SearchRecoveryVariant, VARIANT_ORDER
from .types_v2 import ACTIVATION_DIAGNOSTICS_SCHEMA, ActivationAuditStep, LastCollisionFreeState, LocalConnectorCandidate, LocalConnectorPlan, PublicSegmentAudit, RecoveryModeV2, RecoveryPlanningAudit, SearchRecoveryVariantV2

SEARCH_RECOVERY_VARIANT_ORDER = (*VARIANT_ORDER, *tuple(SearchRecoveryVariantV2))

__all__ = tuple(name for name in globals() if not name.startswith("_"))
