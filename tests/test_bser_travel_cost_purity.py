import unittest
import numpy as np

from core.mapping.travel_cost_service import TravelCostService
from tests.bser_test_utils import synthetic_state


class TravelCostPurityTest(unittest.TestCase):
    def test_query_is_deterministic_and_snapshot_unchanged(self):
        state = synthetic_state(); service = TravelCostService(state); before = state.target_belief.probabilities.copy()
        left = service.query(state.agents[0].position, (2.5, 2.5, 1.0), state.agents[0]); right = service.query(state.agents[0].position, (2.5, 2.5, 1.0), state.agents[0])
        self.assertTrue(left.reachable); self.assertEqual(left.estimated_travel_time, right.estimated_travel_time); np.testing.assert_array_equal(left.path_points, right.path_points); np.testing.assert_array_equal(before, state.target_belief.probabilities)


if __name__ == "__main__": unittest.main()
