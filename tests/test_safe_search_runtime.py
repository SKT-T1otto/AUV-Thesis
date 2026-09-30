"""Mechanism tests for opt-in start repair and failed-attempt backoff."""
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.online.config import load_phase1b2_config
from chapter3_bser.experiments.safe_search_v1.planning_refresh import PlanningViews, start_status, topology_key
from chapter3_bser.experiments.safe_search_v1.failure_policy import SearchFailurePolicy
from chapter3_bser.experiments.safe_search_v1.allocators import SafePriorAllocator
from chapter3_bser.experiments.safe_search_v1.runtime import make_runtime
from tests.bser_test_utils import synthetic_state


class StartPreflightTests(unittest.TestCase):
    def test_stale_endpoint_is_different_from_empty_current_connector(self):
        state = synthetic_state()
        self.assertEqual(start_status(state, 0), "ready")
        agent = replace(state.agents[0], position=(state.agents[0].position[0] + .013, *state.agents[0].position[1:]))
        moved = replace(state, agents=(agent, *state.agents[1:]))
        self.assertEqual(start_status(moved, 0), "stale_start_endpoint")
        endpoints = tuple(replace(e, point=agent.position, connectors=())
            if e.point == state.agents[0].position and e.role == agent.role else e
            for e in state.planning_graph.endpoint_connectors)
        failed = replace(moved, planning_graph=replace(state.planning_graph, endpoint_connectors=endpoints))
        self.assertEqual(start_status(failed, 0), "no_start_connector")

    def test_forced_refresh_only_once_per_physical_step(self):
        state = synthetic_state()
        calls = []
        owner = SimpleNamespace(step=4, safe_counts=Counter(), provider=SimpleNamespace(
            snapshot=lambda **kw: calls.append(kw) or state))
        views = PlanningViews(owner, .4)
        self.assertIs(views.refresh(), state)
        self.assertIs(views.refresh(), state)
        self.assertEqual(calls, [{"force": True}])
        owner.step = 5
        views.refresh()
        self.assertEqual(len(calls), 2)

    def test_unrelated_revision_does_not_invalidate_topology_failure(self):
        state = synthetic_state()
        self.assertEqual(topology_key(state), topology_key(replace(state, map_revision=state.map_revision + 1)))

    def test_invalid_start_short_circuits_entire_candidate_pool(self):
        state = synthetic_state()
        moved = replace(state.agents[0], position=(.013 + state.agents[0].position[0], *state.agents[0].position[1:]))
        state = replace(state, agents=(moved, *state.agents[1:]))
        owner = SimpleNamespace(safe_counts=Counter(), safe_options={"failure_policy": True},
                                planning_views=SimpleNamespace(search=lambda value: value))
        allocator = SafePriorAllocator(state.agents[3].position, owner, None)
        with patch("chapter3_bser.experiments.safe_search_v1.allocators.generate_search_candidates", return_value=((), 0, ())) as generate:
            allocator._candidates(state, (0,))
        self.assertEqual(generate.call_args.args[0].searcher_ids, ())
        self.assertEqual(owner.safe_counts["short_circuited_candidate_pools"], 1)


class FailurePolicyTests(unittest.TestCase):
    def test_failures_do_not_rewrite_successful_cooldown(self):
        policy = SearchFailurePolicy(load_phase1b2_config())
        policy.mark_replan(10, BSEREvent.WAYPOINT_STALE)
        policy.mark_attempt(13, BSEREvent.WAYPOINT_STALE)
        self.assertEqual(policy.last_event_step[BSEREvent.WAYPOINT_STALE], 10)
        self.assertEqual(policy.event_cooldown_remaining(BSEREvent.WAYPOINT_STALE, 14), 1)
        policy.recovery_ready = True
        self.assertEqual(policy.event_cooldown_remaining(BSEREvent.WAYPOINT_STALE, 14), 0)

    def test_fresh_snapshot_can_retry_before_failed_backoff_expires(self):
        policy = SearchFailurePolicy(load_phase1b2_config())
        policy.failed(78, "topology-a")
        state = replace(synthetic_state(), step=80)
        self.assertFalse(policy.ready(state, "topology-a"))
        self.assertTrue(policy.ready(state, "topology-a", full_refresh=True))
        self.assertTrue(policy.ready(state, "topology-b"))
        self.assertTrue(policy.ready(replace(state, step=83), "topology-a"))

    def test_execution_attempt_semantics_are_original(self):
        policy = SearchFailurePolicy(load_phase1b2_config())
        policy.search_enabled = False
        policy.mark_attempt(81, BSEREvent.EXECUTOR_INVALID)
        self.assertEqual(policy.last_event_step[BSEREvent.EXECUTOR_INVALID], 81)

    def test_disabled_factory_uses_exact_original_class(self):
        with patch("chapter3_bser.experiments.safe_search_v1.runtime.BasicSearchPriorRuntime") as original:
            result = make_runtime({}, {}, baseline="B0_search_prior", variant="V0", seed=17, episode_id=3)
        self.assertIs(result, original.return_value)
        original.assert_called_once_with({}, {}, seed=17, episode_id=3)


if __name__ == "__main__":
    unittest.main()
