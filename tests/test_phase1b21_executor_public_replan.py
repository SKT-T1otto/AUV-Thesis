import unittest
from dataclasses import replace

from chapter3_bser.online.allocator import BSEROnlineAllocator
from tests.bser_online_test_utils import state_at


class Phase1B21ExecutorPublicReplanTest(unittest.TestCase):
    def test_executor_only_replan_does_not_require_search_candidates(self):
        allocator = BSEROnlineAllocator()
        state = state_at(0)
        current = replace(allocator.allocate(state), search_assignments=())
        public_target = tuple(float(value) for value in state.grid.cell_centers[0])
        proposed, ok, reason = allocator.replan_executor_to_public_target(
            state,
            current,
            public_target,
            trigger_reason="TEST_PUBLIC_TARGET_REPLAN",
        )
        self.assertTrue(ok, reason)
        self.assertEqual(proposed.search_assignments, ())
        self.assertEqual(proposed.executor_assignment.target_region, public_target)
        self.assertTrue(proposed.search_frozen)


if __name__ == "__main__":
    unittest.main()
