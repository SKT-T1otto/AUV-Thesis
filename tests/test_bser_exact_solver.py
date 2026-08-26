import unittest
from chapter3_bser.exact_solver import exact_combination_count, solve_joint_exact
from tests.bser_test_utils import synthetic_instance


class ExactSolverTest(unittest.TestCase):
    def test_complete_enumeration_count(self):
        _, _, generated, context = synthetic_instance(); result = solve_joint_exact(generated.search_candidates, generated.standby_candidates, context)
        self.assertEqual(result.combination_count, exact_combination_count(generated.search_candidates, generated.standby_candidates)); self.assertEqual(result.status, "OK")


if __name__ == "__main__": unittest.main()
