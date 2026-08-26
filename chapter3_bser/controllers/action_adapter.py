"""Convert high-level targets to legal fixed feedback actions for E2 only."""

from __future__ import annotations

import numpy as np

from chapter3_bser.controllers.path_tracker import PathTracker
from chapter3_bser.online.types import OnlineAllocation
from core.mapping.planning_state import PlanningStateView


def assignment_to_fixed_actions(
    state: PlanningStateView,
    allocation: OnlineAllocation,
    *,
    position_gain: float = 0.35,
    damping_gain: float = 0.18,
    path_tracker: PathTracker | None = None,
) -> np.ndarray:
    assignments = {
        item.agent_id: (item.path, item.waypoint, True)
        for item in allocation.search_assignments
    }
    executor = allocation.executor_assignment
    assignments[executor.executor_id] = (
        executor.path,
        executor.target_region,
        bool(executor.reachable),
    )
    if path_tracker is not None:
        path_tracker.prune(assignments)
    actions = np.zeros((len(state.agents), 3), dtype=np.float32)
    for agent in state.agents:
        assignment = assignments.get(agent.agent_id)
        if assignment is None or not assignment[2]:
            # An unreachable locked executor target must not be pursued by a
            # straight-line fallback. Hold nominally until a public-path replan
            # becomes feasible.
            target_value = agent.position
        elif path_tracker is None:
            target_value = assignment[1]
        else:
            target_value = path_tracker.tracking_target(
                agent.agent_id,
                agent.position,
                assignment[0],
                assignment[1],
            )
        target = np.asarray(target_value, dtype=np.float64)
        delta = target - np.asarray(agent.position, dtype=np.float64)
        velocity = np.asarray(agent.velocity, dtype=np.float64)
        actions[agent.agent_id] = np.clip(position_gain * delta - damping_gain * velocity, -1.0, 1.0)
    return actions
