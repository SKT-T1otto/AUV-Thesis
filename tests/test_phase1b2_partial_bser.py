import unittest
from unittest.mock import patch

from chapter3_bser.objective import build_objective_context, evaluate_objective
from chapter3_bser.online import allocator as allocator_module
from chapter3_bser.online.allocator import BSEROnlineAllocator
from tests.bser_online_test_utils import shifted_belief, state_at


class Phase1B2PartialBSERTest(unittest.TestCase):
    def test_partial_result_uses_joint_greedy_and_original_objective(self):
        allocator = BSEROnlineAllocator()
        current = allocator.allocate(state_at(0))
        state = shifted_belief(1)
        with patch.object(
            allocator_module,
            "solve_joint_greedy",
            wraps=allocator_module.solve_joint_greedy,
        ) as greedy:
            proposed, ok, reason = allocator.allocate_partial(
                state,
                current,
                affected_searcher_ids=(0,),
                executor_affected=False,
                trigger_reason="TEST_PARTIAL_BSER",
            )
        self.assertTrue(ok, reason)
        greedy.assert_called_once()

        candidates = tuple(
            allocator._frozen_search_candidate(item)
            for item in proposed.search_assignments
        )
        standby = allocator._frozen_standby_candidate(proposed)
        context = build_objective_context(
            state,
            candidates,
            (standby,),
            allocator.config,
        )
        expected = evaluate_objective(candidates, standby, context)
        self.assertAlmostEqual(proposed.objective_value, expected, places=12)
        old_by_agent = {item.agent_id: item for item in current.search_assignments}
        new_by_agent = {item.agent_id: item for item in proposed.search_assignments}
        for agent_id in set(old_by_agent) - {0}:
            self.assertIs(new_by_agent[agent_id], old_by_agent[agent_id])


if __name__ == "__main__":
    unittest.main()
