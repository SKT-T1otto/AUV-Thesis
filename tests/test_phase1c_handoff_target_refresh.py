from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest import mock

from chapter3_bser.controllers.state_provider import OnlinePlanningStateProvider
from core.env.task_state import AgentStateView
from tests.bser_online_test_utils import state_at


class _Env:
    def __init__(self) -> None:
        self.task = SimpleNamespace(
            step=0,
            target_found=False,
            executor_knows_target=False,
            mission_complete=False,
        )
        self.target = (1.5, 1.5, 1.0)

    def get_task_state(self):
        return self.task

    def get_agent_state(self):
        state = state_at(int(self.task.step), target_found=self.task.target_found)
        targets = [agent.current_navigation_target for agent in state.agents]
        targets[3] = self.target
        return AgentStateView(
            role_order=tuple(agent.role for agent in state.agents),
            positions=tuple(agent.position for agent in state.agents),
            velocities=tuple(agent.velocity for agent in state.agents),
            navigation_targets=tuple(targets),
            collision_flags=(False, False, False, False),
        )


class Phase1CHandoffTargetRefreshTests(unittest.TestCase):
    def test_handoff_forces_full_refresh_only_when_enabled(self) -> None:
        env = _Env()
        initial = state_at(0)
        refreshed = state_at(1, target_found=True)
        with mock.patch(
            "chapter3_bser.controllers.state_provider.extract_planning_state",
            side_effect=(initial, refreshed),
        ) as extract:
            provider = OnlinePlanningStateProvider(
                env, refresh_interval=20, refresh_on_executor_handoff=True
            )
            provider.initialize()
            env.task = SimpleNamespace(
                step=1,
                target_found=True,
                executor_knows_target=True,
                mission_complete=False,
            )
            result = provider.snapshot()
            self.assertIs(result, refreshed)
            self.assertEqual(extract.call_count, 2)
            self.assertTrue(provider.last_snapshot_was_full_refresh)
            self.assertEqual(provider.last_refresh_reason, "EXECUTOR_HANDOFF")
            self.assertEqual(provider.handoff_forced_refresh_count, 1)

        env = _Env()
        with mock.patch(
            "chapter3_bser.controllers.state_provider.extract_planning_state",
            return_value=initial,
        ) as extract:
            provider = OnlinePlanningStateProvider(env, refresh_interval=20)
            provider.initialize()
            env.task = SimpleNamespace(
                step=1,
                target_found=False,
                executor_knows_target=True,
                mission_complete=False,
            )
            provider.snapshot()
            self.assertEqual(extract.call_count, 1)
            self.assertFalse(provider.last_snapshot_was_full_refresh)

    def test_public_target_shift_is_cumulative_from_last_full_refresh(self) -> None:
        env = _Env()
        env.task.executor_knows_target = True
        initial = state_at(0)
        refreshed = replace(state_at(20), agents=state_at(20).agents)
        with mock.patch(
            "chapter3_bser.controllers.state_provider.extract_planning_state",
            side_effect=(initial, refreshed),
        ) as extract:
            provider = OnlinePlanningStateProvider(
                env,
                refresh_interval=0,
                refresh_on_public_target_shift=True,
                public_target_update_distance=0.75,
                public_target_update_min_steps=20,
            )
            provider.initialize()
            env.task.step = 10
            env.target = (1.9, 1.5, 1.0)
            provider.snapshot()
            self.assertFalse(provider.last_snapshot_was_full_refresh)
            env.task.step = 20
            env.target = (2.3, 1.5, 1.0)
            provider.snapshot()
            self.assertEqual(extract.call_count, 2)
            self.assertEqual(provider.last_refresh_reason, "PUBLIC_TARGET_SHIFT")
            self.assertEqual(provider.target_shift_forced_refresh_count, 1)


if __name__ == "__main__":
    unittest.main()
