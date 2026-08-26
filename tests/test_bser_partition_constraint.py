import unittest
from chapter3_bser.greedy_solver import solve_joint_greedy
from chapter3_bser.metrics import partition_feasible
from tests.bser_test_utils import synthetic_instance


class PartitionConstraintTest(unittest.TestCase):
    def test_at_most_one_candidate_per_searcher(self):
        _, _, generated, context = synthetic_instance(); result = solve_joint_greedy(generated.search_candidates, generated.standby_candidates, context)
        self.assertTrue(partition_feasible(result.selected)); self.assertLessEqual(len(result.selected), 3)


if __name__ == "__main__": unittest.main()
