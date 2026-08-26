from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from chapter3_bser.integration.control_context import (
    AgentAssignmentContextV1,
    BSERControlContextV1,
    ExecutorAssignmentContextV1,
)
from chapter3_bser.integration.guided_env import GuidedEnv
from core.env.task_state import AgentStateView


class _Runtime:
    def __init__(self) -> None:
        self.num_agents = 4
        self.executor_id = 3
        self._nav_targets = torch.zeros((4, 3), dtype=torch.float32)
        self._targets = torch.zeros((4, 3), dtype=torch.float32)
        self._agent_pos = torch.zeros((4, 3), dtype=torch.float32)
        self._agent_vel = torch.zeros((4, 3), dtype=torch.float32)
        self._collision_flags = torch.zeros(4, dtype=torch.bool)
        self.online_unknown_map_active = False

    def _update_nav_targets(self):
        self._nav_targets.copy_(
            torch.tensor(
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [9.0, 8.0, 7.0]]
            )
        )
        self._targets.copy_(self._nav_targets)


class _Env:
    def __init__(self) -> None:
        self.unwrapped = _Runtime()
        self.task = SimpleNamespace(executor_knows_target=True)

    def reset(self, scenario=None):
        self.unwrapped._update_nav_targets()
        return tuple(torch.zeros(28) for _ in range(4))

    def get_task_state(self):
        return self.task

    def get_agent_state(self):
        runtime = self.unwrapped
        triples = lambda tensor: tuple(
            tuple(float(value) for value in row) for row in tensor.tolist()
        )
        return AgentStateView(
            role_order=("searcher", "searcher", "searcher", "executor"),
            positions=triples(runtime._agent_pos),
            velocities=triples(runtime._agent_vel),
            navigation_targets=triples(runtime._nav_targets),
            collision_flags=(False, False, False, False),
        )

    def close(self):
        return None


def _guidance() -> BSERControlContextV1:
    active = (
        (10.0, 0.0, 0.0),
        (20.0, 0.0, 0.0),
        (30.0, 0.0, 0.0),
        (40.0, 0.0, 0.0),
    )
    agents = tuple(
        AgentAssignmentContextV1(
            agent_id=index,
            role="executor" if index == 3 else "searcher",
            assignment_kind="test",
            assignment_id=f"agent-{index}",
            final_waypoint=target,
            planned_path=(),
            tracking_waypoint=target,
            hold_position=(0.0, 0.0, 0.0),
            hold_state=False,
            reachable=True,
            execution_request=index == 3,
        )
        for index, target in enumerate(active)
    )
    return BSERControlContextV1(
        schema_version="bser.control_context.v1",
        allocation_version="test",
        allocation_hash="test",
        step=0,
        mission_phase="EXECUTION",
        agent_assignments=agents,
        executor_assignment=ExecutorAssignmentContextV1(
            executor_id=3,
            source="PUBLIC_HANDOFF",
            target_region=(9.0, 8.0, 7.0),
            planned_path=(),
            tracking_waypoint=active[3],
            hold_position=(0.0, 0.0, 0.0),
            hold_state=False,
            reachable=True,
            execution_request=True,
        ),
        decision_reason="TEST",
    )


class Phase1CPublicTargetSeparationTests(unittest.TestCase):
    def test_public_executor_target_is_separate_from_active_guidance(self) -> None:
        env = GuidedEnv(_Env(), enabled=True)
        try:
            observations = env.reset()
            env.install_guidance(_guidance())
            env.unwrapped._update_nav_targets()

            public = env.get_agent_state()
            self.assertEqual(public.navigation_targets[3], (9.0, 8.0, 7.0))
            self.assertEqual(env.public_executor_navigation_target, (9.0, 8.0, 7.0))
            self.assertEqual(
                tuple(float(value) for value in env.unwrapped._nav_targets[3]),
                (40.0, 0.0, 0.0),
            )
            self.assertEqual(
                public.navigation_targets[:3],
                ((10.0, 0.0, 0.0), (20.0, 0.0, 0.0), (30.0, 0.0, 0.0)),
            )
            self.assertTrue(all(tuple(value.shape) == (28,) for value in observations))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
