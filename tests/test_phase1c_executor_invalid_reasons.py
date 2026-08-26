from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest

from chapter3_bser.experiments.phase1c_common.execution_diagnostics import (
    ExecutionEpisodeDiagnostics,
)
from chapter3_bser.events.event_detector import EventDetector
from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.online.allocator import BSEROnlineAllocator
from core.mapping.planning_graph import PlanningConnectorView
from tests.bser_online_test_utils import state_at
from tests.test_phase1c_execution_diagnostics import Env
from tests.test_phase1c_stale_endpoint_deferral import (
    _config,
    _state_with_executor,
)


class Phase1CExecutorInvalidReasonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = state_at(0)
        self.base_allocation = BSEROnlineAllocator().allocate(self.previous)
        self.target = (1.2, 1.2, 1.0)

    def _allocation(self, **changes):
        executor = replace(
            self.base_allocation.executor_assignment,
            target_region=self.target,
            **changes,
        )
        return replace(self.base_allocation, executor_assignment=executor)

    def test_fresh_endpoint_query_unreachable_is_real_invalid(self) -> None:
        current = _state_with_executor(
            state_at(1),
            (2.2, 2.2, 1.0),
            self.target,
            start_connectors=(),
        )
        result = EventDetector(_config()).detect(
            self.previous,
            current,
            assignment=self._allocation(reachable=True),
        )
        self.assertTrue(result.executor_validity_evaluated)
        self.assertFalse(result.executor_validity_deferred)
        self.assertEqual(result.executor_invalid_reason, "QUERY_UNREACHABLE")
        self.assertEqual(result.executor_query_failure_reason, "no_start_connector")
        self.assertIn(BSEREvent.EXECUTOR_INVALID, result.events)

    def test_assignment_unreachable_is_not_hidden_by_stale_snapshot(self) -> None:
        current = _state_with_executor(
            state_at(1), (2.2, 2.2, 1.0), self.target, start_connectors=None
        )
        result = EventDetector(_config()).detect(
            self.previous,
            current,
            assignment=self._allocation(reachable=False),
        )
        self.assertEqual(result.executor_invalid_reason, "ASSIGNMENT_UNREACHABLE")
        self.assertFalse(result.executor_validity_deferred)
        self.assertIn(BSEREvent.EXECUTOR_INVALID, result.events)

    def test_same_target_cost_increase_has_distinct_reason(self) -> None:
        current = _state_with_executor(
            state_at(1),
            self.previous.agents[3].position,
            self.target,
            start_connectors=(PlanningConnectorView(8, 0, 0.0, 0.0),),
        )
        result = EventDetector(_config()).detect(
            self.previous,
            current,
            assignment=self._allocation(reachable=True, planning_cost=0.01),
        )
        self.assertTrue(result.executor_query_reachable)
        self.assertGreater(result.executor_planning_cost_relative_change, 0.15)
        self.assertEqual(result.executor_invalid_reason, "PLANNING_COST_INCREASE")
        self.assertIn(BSEREvent.EXECUTOR_INVALID, result.events)

    def test_episode_diagnostics_count_structured_reasons_and_refreshes(self) -> None:
        env = Env()
        collector = ExecutionEpisodeDiagnostics().reset(
            env, episode_index=0, max_steps=10
        )
        result = SimpleNamespace(
            events=(
                BSEREvent.EXECUTOR_INVALID,
                BSEREvent.EXECUTOR_PUBLIC_TARGET_UPDATED,
            ),
            replanned=True,
            diagnostics=SimpleNamespace(
                affected_agent_ids=(3,), allocation_scope="executor_public_target"
            ),
            event_detection=SimpleNamespace(
                executor_invalid_reason="QUERY_UNREACHABLE",
                executor_validity_evaluated=True,
                executor_validity_deferred=False,
                executor_public_target_shift=1.25,
            ),
        )
        provider = SimpleNamespace(
            full_refresh_count=3,
            handoff_forced_refresh_count=1,
            target_shift_forced_refresh_count=1,
        )
        collector.observe_controller_result(
            env, result, state_provider=provider
        )
        row = collector.finalize(env)
        self.assertEqual(row["executor_invalid_query_unreachable_count"], 1)
        self.assertEqual(row["executor_validity_evaluation_count"], 1)
        self.assertEqual(row["public_target_update_event_count"], 1)
        self.assertEqual(row["public_target_update_accepted_count"], 1)
        self.assertEqual(row["public_target_shift_sum"], 1.25)
        self.assertEqual(row["full_planning_refresh_count"], 3)
        self.assertEqual(row["handoff_forced_refresh_count"], 1)
        self.assertEqual(row["target_shift_forced_refresh_count"], 1)


if __name__ == "__main__":
    unittest.main()
