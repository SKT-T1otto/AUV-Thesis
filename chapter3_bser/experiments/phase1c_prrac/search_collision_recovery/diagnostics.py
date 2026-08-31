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
    }
    for agent_id in range(3):
        for name in ("search_recovery_entry_count", "route_refresh_attempt_count", "route_refresh_success_count", "egress_attempt_count", "egress_success_count"):
            result[f"{name}_agent_{agent_id}"] = 0
        result[f"last_recovery_mode_agent_{agent_id}"] = SearchRecoveryMode.NORMAL_SEARCH.value
        result[f"last_recovery_failure_reason_agent_{agent_id}"] = None
    return result


__all__ = ("baseline_recovery_summary",)
