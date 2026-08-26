import unittest
from chapter3_bser.candidate_generator import generate_candidates
from tests.bser_test_utils import synthetic_state
from chapter3_bser.config import load_bser_config


class DeterminismTest(unittest.TestCase):
    def test_candidate_ids_repeat_exactly(self):
        state = synthetic_state(); config = load_bser_config(); left = generate_candidates(state, config, k_search=4, k_standby=4); right = generate_candidates(state, config, k_search=4, k_standby=4)
        self.assertEqual(tuple(x.candidate_id for x in left.search_candidates), tuple(x.candidate_id for x in right.search_candidates)); self.assertEqual(tuple(x.waypoint for x in left.standby_candidates), tuple(x.waypoint for x in right.standby_candidates))


if __name__ == "__main__": unittest.main()
