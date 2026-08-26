import unittest
from chapter3_bser.exact_solver import solve_joint_exact
from chapter3_bser.greedy_solver import solve_joint_greedy
from tests.bser_test_utils import synthetic_instance


class GreedyBoundTest(unittest.TestCase):
    def test_joint_half_bound(self):
        _, _, generated, context = synthetic_instance(); exact = solve_joint_exact(generated.search_candidates, generated.standby_candidates, context); greedy = solve_joint_greedy(generated.search_candidates, generated.standby_candidates, context)
        self.assertGreaterEqual(greedy.objective / exact.objective, 0.5 - 1e-9)


if __name__ == "__main__": unittest.main()
