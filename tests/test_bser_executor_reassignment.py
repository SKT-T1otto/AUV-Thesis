import unittest

from chapter3_bser.online.controller import OnlineBSERController
from tests.bser_online_test_utils import state_at


class ExecutorReassignmentTest(unittest.TestCase):
    def test_target_found_uses_belief_peak_and_freezes_search(self):
        controller = OnlineBSERController()
        controller.initialize(state_at(0))
        state = state_at(1, target_found=True)
        result = controller.step(state)
        expected = tuple(float(value) for value in state.grid.cell_centers[state.target_belief.peak_index])
        self.assertTrue(result.replanned)
        self.assertTrue(result.allocation.search_frozen)
        self.assertEqual(result.allocation.search_assignments, ())
        self.assertEqual(result.allocation.executor_assignment.target_region, expected)
        self.assertEqual(result.allocation.executor_assignment.source, "target_found_belief_peak")


if __name__ == "__main__": unittest.main()
