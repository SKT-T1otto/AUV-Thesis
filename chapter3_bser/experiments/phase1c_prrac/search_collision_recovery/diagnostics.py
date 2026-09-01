"""Plain scalar diagnostics for recovery variants, including strict C0 zeros."""

from __future__ import annotations

from typing import Any

from .types import SearchRecoveryMode


def baseline_recovery_summary() -> dict[str, Any]:
    result: dict[str, Any] = {
        "search_recovery_entry_count": 0,
        "search_recovery_active_step_count": 0,
        "search_recovery_active_rate": 0.0,
        "route_refresh_attempt_count": 0,
        "route_refresh_success_count": 0,
        "route_refresh_failure_count": 0,
        "egress_attempt_count": 0,
        "egress_success_count": 0,
        "egress_failure_count": 0,
        "egress_rejoin_count": 0,
        "recovery_collision_count": 0,
        "recovery_max_collision_streak": 0,
        "recovery_duration_sum": 0,
        "recovery_duration_mean": None,
        "recovery_duration_max": 0,
        "recovery_no_egress_count": 0,
        "recovery_failed_endpoint_count": 0,
        "forced_public_refresh_count": 0,
        "route_refresh_identical_to_base_count": 0,
        "local_connector_attempt_count": 0,
        "local_connector_plan_count": 0,
        "local_connector_reached_count": 0,
        "local_connector_collision_count": 0,
        "graph_reconnect_attempt_count": 0,
        "graph_reconnect_success_count": 0,
        "graph_reconnect_failure_count": 0,
        "recovery_state_non_normal_step_count": 0,
        "recovery_plan_active_step_count": 0,
        "recovery_guidance_changed_step_count": 0,
        "recovery_effective_intervention_episode": False,
        "recovery_effective_intervention_count": 0,
        "effective_recovery_active_rate": 0.0,
        "tracking_waypoint_delta_norm_sum": 0.0,
        "tracking_waypoint_delta_norm_mean": None,
        "tracking_waypoint_delta_norm_max": 0.0,
        "path_changed_step_count": 0,
        "recovery_failed_pass_through_count": 0,
        "failure_reason_distribution": {},
        "candidate_tier_distribution": {f"tier{tier}": 0 for tier in range(4)},
    }
    for agent_id in range(3):
        for name in ("search_recovery_entry_count", "route_refresh_attempt_count", "route_refresh_success_count", "egress_attempt_count", "egress_success_count"):
            result[f"{name}_agent_{agent_id}"] = 0
        result[f"last_recovery_mode_agent_{agent_id}"] = SearchRecoveryMode.NORMAL_SEARCH.value
        result[f"last_recovery_failure_reason_agent_{agent_id}"] = None
    for tier in range(4):
        result[f"tier{tier}_count"] = 0
    return result


__all__ = ("baseline_recovery_summary",)
