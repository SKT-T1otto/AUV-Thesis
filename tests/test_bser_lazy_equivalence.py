import unittest
from chapter3_bser.greedy_solver import solve_joint_greedy
from chapter3_bser.lazy_greedy_solver import solve_joint_lazy
from tests.bser_test_utils import synthetic_instance


class LazyEquivalenceTest(unittest.TestCase):
    def test_lazy_matches_standard(self):
        _, _, generated, context = synthetic_instance(); standard = solve_joint_greedy(generated.search_candidates, generated.standby_candidates, context); lazy = solve_joint_lazy(generated.search_candidates, generated.standby_candidates, context)
        self.assertEqual(standard.selected_ids, lazy.selected_ids); self.assertEqual(standard.standby, lazy.standby); self.assertAlmostEqual(standard.objective, lazy.objective)


if __name__ == "__main__": unittest.main()
