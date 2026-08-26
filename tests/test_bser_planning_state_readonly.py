import unittest
import numpy as np

from tests.bser_test_utils import synthetic_state


class PlanningStateReadonlyTest(unittest.TestCase):
    def test_arrays_are_defensive_readonly_values(self):
        state = synthetic_state()
        for value in (state.grid.cell_centers, state.target_belief.probabilities, state.occupancy.occupancy_probability, state.occupancy.known_mask):
            self.assertFalse(value.flags.writeable)
            with self.assertRaises(ValueError):
                value.reshape(-1)[0] = 0
        self.assertAlmostEqual(float(np.sum(state.target_belief.probabilities)), 1.0)


if __name__ == "__main__": unittest.main()
