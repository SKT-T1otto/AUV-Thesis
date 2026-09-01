"""Minimal immutable guidance substitution for active Searcher recovery plans."""

from __future__ import annotations

from dataclasses import replace
from typing import Any


def apply_search_recovery_guidance(base_guidance: Any, planning_state: Any, controller: Any | None):
    if controller is None:
        return base_guidance
    if planning_state.target_found or planning_state.mission_complete:
        return base_guidance
    agents_by_id = {int(item.agent_id): item for item in planning_state.agents}
    changed = False
    assignments = []
    for assignment in base_guidance.agent_assignments:
        agent_id = int(assignment.agent_id)
        state = controller.agents.get(agent_id)
        plan = None if state is None else state.plan
        if agent_id not in (0, 1, 2) or plan is None:
            assignments.append(assignment)
            continue
        tracking = controller.path_tracker.tracking_target(
            agent_id,
            agents_by_id[agent_id].position,
            plan.path,
            plan.navigation_endpoint,
        )
        assignments.append(
            replace(
                assignment,
                assignment_kind=(
                    state.semantic_assignment_kind
                    if hasattr(state, "semantic_assignment_kind") and state.semantic_assignment_kind is not None
                    else assignment.assignment_kind
                ),
                assignment_id=(
                    plan.semantic_search_candidate_id
                    if str(getattr(getattr(controller, "variant", None), "value", "")).startswith("S2A1_")
                    else assignment.assignment_id
                ),
                final_waypoint=(
                    plan.semantic_search_waypoint
                    if str(getattr(getattr(controller, "variant", None), "value", "")).startswith("S2A1_")
                    else assignment.final_waypoint
                ),
                planned_path=plan.path,
                tracking_waypoint=tuple(float(v) for v in tracking),
                hold_state=False,
                reachable=True,
            )
        )
        changed = True
    if not changed:
        return base_guidance
    active_v2 = [
        state for state in controller.agents.values()
        if state.plan is not None and getattr(state, "base_allocation_hash", None) is not None
    ]
    return replace(
        base_guidance,
        agent_assignments=tuple(assignments),
        allocation_version=(active_v2[0].base_allocation_version if active_v2 else base_guidance.allocation_version),
        allocation_hash=(active_v2[0].base_allocation_hash if active_v2 else base_guidance.allocation_hash),
    )


__all__ = ("apply_search_recovery_guidance",)
