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
                planned_path=plan.path,
                tracking_waypoint=tuple(float(v) for v in tracking),
                hold_state=False,
                reachable=True,
            )
        )
        changed = True
    return base_guidance if not changed else replace(base_guidance, agent_assignments=tuple(assignments))


__all__ = ("apply_search_recovery_guidance",)
