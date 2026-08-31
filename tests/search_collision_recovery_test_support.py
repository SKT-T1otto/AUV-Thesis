from __future__ import annotations

from dataclasses import replace

from chapter3_bser.integration.control_context import AgentAssignmentContextV1, BSERControlContextV1, ExecutorAssignmentContextV1


def guidance():
    assignments = tuple(
        AgentAssignmentContextV1(
            agent_id=agent_id,
            role="executor" if agent_id == 3 else "searcher",
            assignment_kind="executor_standby" if agent_id == 3 else "search",
            assignment_id="EXECUTOR" if agent_id == 3 else f"candidate-{agent_id}",
            final_waypoint=(1.5, 1.5, 1.0),
            planned_path=((1.5, 1.5, 1.0),),
            tracking_waypoint=(1.5, 1.5, 1.0),
            hold_position=(0.5, 0.5, 1.0),
            hold_state=agent_id == 3,
            reachable=True,
            execution_request=False,
        )
        for agent_id in range(4)
    )
    executor = ExecutorAssignmentContextV1(3, "STANDBY", (1.5,1.5,1.0), (), (1.5,1.5,1.0), (2.5,2.5,1.0), True, True, False)
    return BSERControlContextV1("bser.control_context.v1", "v1:hash", "hash", 0, "SEARCH", assignments, executor, "BASE")


def state_at(state, step, *, found=False, complete=False):
    return replace(state, step=int(step), target_found=bool(found), mission_complete=bool(complete))
