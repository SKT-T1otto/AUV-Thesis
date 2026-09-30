"""Synthetic regressions for searchers omitted from an initial allocation."""
from collections import Counter
from dataclasses import dataclass, replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from chapter3_bser.experiments.safe_search_v1 import runtime as safe
from chapter3_bser.experiments.safe_search_v1.allocators import SafeJointAllocator
from chapter3_bser.experiments.safe_search_v1 import safe_guidance
from chapter3_bser.online import allocator as native_allocator
from chapter3_bser.online.types import SearchAssignment, ExecutorAssignment, OnlineAllocation
from tests.test_safe_search_runtime_review import config


def allocation():
    search = tuple(SearchAssignment(i, f"old_{i}", (3., 3., 1.),
        ((1., 1., 1.), (3., 3., 1.)), 2., (0, 1), 2.) for i in (0, 1))
    executor = ExecutorAssignment(3, (5., 5., 1.), ((5., 5., 1.),), 0., "STANDBY", True, (2,), 0.)
    return OnlineAllocation(search, executor, .2, .2, 2., "INITIALIZE")


@dataclass
class FakeGuidance:
    agent_assignments: tuple


class MissingRecoveryTests(unittest.TestCase):
    def test_native_replan_clears_restored_missing_id_without_next_tick_retry(self):
        current = allocation()
        restored = replace(current, search_assignments=current.search_assignments + (
            replace(current.search_assignments[0], agent_id=2, candidate_id="new_2"),))
        owner = NS(safe_options=dict(safe.VARIANTS["V2"]), safe_pending={2}, safe_counts=Counter(),
                   provider=NS(last_snapshot_was_full_refresh=False), planning_views=NS())
        partial = Mock()
        inner = NS(config=config(), detector=NS(detect=lambda *args, **kwargs: NS(stale_searcher_ids=(), events=())),
                   cache=NS(current=None), current_context=None, current_allocation=current,
                   allocator=NS(allocate_partial=partial))
        def native_step(state, context):
            inner.current_allocation = restored
            return NS(replanned=state.step == 1, allocation=restored,
                      diagnostics=NS(optimizer_invoked=state.step == 1))
        inner.step = native_step
        controller = safe.SearchController(inner, owner)
        with patch.object(safe, "topology_key", return_value="same"), \
             patch.object(safe, "start_status", return_value="ready"):
            controller.step(NS(step=1, target_found=False, searcher_ids=(0, 1, 2)))
            self.assertFalse(owner.safe_pending)
            controller.step(NS(step=2, target_found=False, searcher_ids=(0, 1, 2)))
        partial.assert_not_called()
        self.assertFalse(owner.safe_pending)

    def test_v2_missing_searcher_enters_recovery_without_stale_event(self):
        current = allocation()
        owner = NS(safe_options=dict(safe.VARIANTS["V2"]), safe_pending=set(), safe_counts=Counter(),
                   provider=NS(last_snapshot_was_full_refresh=False), planning_views=NS())
        detector = NS(detect=lambda *args, **kwargs: NS(stale_searcher_ids=(), events=()))
        partial = Mock(return_value=(current, False, "ATOMIC_REJECT_MISSING_SEARCH_ROUTE"))
        inner = NS(config=config(), detector=detector, cache=NS(current=None), current_context=None,
                   current_allocation=current, allocator=NS(allocate_partial=partial),
                   step=lambda *args: NS(replanned=False, diagnostics=NS(optimizer_invoked=False)))
        controller = safe.SearchController(inner, owner)
        state = NS(step=1, target_found=False, searcher_ids=(0, 1, 2))
        with patch.object(safe, "topology_key", return_value="same"), \
             patch.object(safe, "start_status", return_value="ready"):
            controller.step(state)
        partial.assert_called_once_with(state, current, affected_searcher_ids=(2,),
                                        executor_affected=False, trigger_reason="SAFE_SEARCH_RECOVERY")
        self.assertIn(2, owner.safe_pending)

    def test_v4_safety_bridge_marks_unassigned_searcher_pending(self):
        current = allocation()
        state = NS(step=0, target_found=False, searcher_ids=(0, 1, 2), executor_id=3,
                   agents=tuple(NS(agent_id=i, position=(1., 1., 1.)) for i in range(4)))
        owner = NS(safe_options=dict(safe.VARIANTS["V4"]), safe_pending=set(), safe_counts=Counter(),
                   state=state, planning_views=NS(live=lambda: state, clearance=.4))
        context = FakeGuidance(tuple(NS(agent_id=i) for i in range(4)))
        mission = NS(target_found=False, mission_complete=False, searcher_finished_flags=(False, False, False))
        bridge = safe_guidance.SafetyBridge(owner)
        with patch.object(safe_guidance.RMADDPGGuidanceBridge, "compile_guidance", return_value=context), \
             patch.object(safe_guidance, "PublicGeometry", return_value=object()), \
             patch.object(bridge, "_remaining_safe", return_value=True):
            result = bridge.compile_guidance(current, state, mission)
        self.assertEqual(owner.safe_pending, {2})
        self.assertEqual(bridge.last_safety_status[2], "UNASSIGNED")
        self.assertIs(result.agent_assignments[3], context.agent_assignments[3])

    def test_joint_partial_recovery_keeps_newly_selected_missing_searcher(self):
        current = allocation()
        candidate = NS(agent_id=2, candidate_id="recovered_2", waypoint=(7., 7., 1.),
            path_points=((1., 1., 1.), (7., 7., 1.)), path_cell_indices=(0, 3),
            physical_travel_time=4., planning_cost=4.)
        allocator = SafeJointAllocator.__new__(SafeJointAllocator)
        allocator.config = dict(candidate_generation={})
        allocator._partial_search_candidates = lambda state, affected: (candidate,)
        allocator._solve_candidates = lambda candidates, standby, context: NS(
            selected=candidates, standby=standby[0], status="OK")
        state = NS(searcher_ids=(0, 1, 2))
        with patch.object(native_allocator, "build_objective_context", return_value=object()), \
             patch.object(native_allocator, "evaluate_objective", return_value=.3) as objective, \
             patch.object(native_allocator, "expected_detection_probability", return_value=.3), \
             patch.object(native_allocator, "response_diagnostics", return_value=NS(conditional_reachable_response_time=2.)):
            proposed, ok, _ = allocator.allocate_partial(state, current,
                affected_searcher_ids=(2,), executor_affected=False, trigger_reason="SAFE_SEARCH_RECOVERY")
        self.assertTrue(ok)
        self.assertEqual({item.agent_id for item in proposed.search_assignments}, {0, 1, 2})
        self.assertIs(proposed.executor_assignment, current.executor_assignment)
        self.assertIs(proposed.search_assignments[0], current.search_assignments[0])
        self.assertIs(proposed.search_assignments[1], current.search_assignments[1])
        self.assertEqual({item.agent_id for item in objective.call_args.args[0]}, {0, 1, 2})

    def test_existing_agent_partial_replans_delegate_to_native_implementation(self):
        allocator = SafeJointAllocator.__new__(SafeJointAllocator)
        current, state = allocation(), NS(searcher_ids=(0, 1, 2))
        sentinel = (current, False, "native_result")
        with patch.object(native_allocator.BSEROnlineAllocator, "allocate_partial", return_value=sentinel) as original:
            result = allocator.allocate_partial(state, current, affected_searcher_ids=(1,),
                executor_affected=False, trigger_reason="SAME_SCOPE")
        self.assertIs(result, sentinel)
        original.assert_called_once_with(state, current, affected_searcher_ids={1},
                                         executor_affected=False, trigger_reason="SAME_SCOPE")


if __name__ == "__main__":
    unittest.main()
