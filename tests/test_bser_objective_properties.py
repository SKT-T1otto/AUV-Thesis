"""Current bounded behavior checks; original TestCase bodies retained."""


import unittest
from chapter3_bser.objective import evaluate_objective, marginal_gain
from tests.bser_test_utils import synthetic_instance


class ObjectiveTest(unittest.TestCase):
    def test_empty_zero_and_marginal_identity(self):
        _, _, generated, context = synthetic_instance(); y = generated.standby_candidates[0]; e = generated.search_candidates[0]
        self.assertEqual(evaluate_objective((), y, context), 0.0); self.assertAlmostEqual(marginal_gain((), e, y, context), evaluate_objective((e,), y, context))




import unittest
from chapter3_bser.metrics import validate_small_instance
from tests.bser_test_utils import synthetic_instance


class MonotonicityTest(unittest.TestCase):
    def test_all_feasible_extensions(self):
        _, _, generated, context = synthetic_instance(); self.assertTrue(validate_small_instance(generated.search_candidates, generated.standby_candidates, context)["monotonicity_pass"])




import unittest
from chapter3_bser.metrics import validate_small_instance
from tests.bser_test_utils import synthetic_instance


class SubmodularityTest(unittest.TestCase):
    def test_diminishing_returns(self):
        _, _, generated, context = synthetic_instance(); self.assertTrue(validate_small_instance(generated.search_candidates, generated.standby_candidates, context)["submodularity_pass"])




import unittest
from chapter3_bser.exact_solver import solve_joint_exact
from chapter3_bser.greedy_solver import solve_joint_greedy
from tests.bser_test_utils import synthetic_instance


class GreedyBoundTest(unittest.TestCase):
    def test_joint_half_bound(self):
        _, _, generated, context = synthetic_instance(); exact = solve_joint_exact(generated.search_candidates, generated.standby_candidates, context); greedy = solve_joint_greedy(generated.search_candidates, generated.standby_candidates, context)
        self.assertGreaterEqual(greedy.objective / exact.objective, 0.5 - 1e-9)




import unittest
from chapter3_bser.exact_solver import exact_combination_count, solve_joint_exact
from tests.bser_test_utils import synthetic_instance


class ExactSolverTest(unittest.TestCase):
    def test_complete_enumeration_count(self):
        _, _, generated, context = synthetic_instance(); result = solve_joint_exact(generated.search_candidates, generated.standby_candidates, context)
        self.assertEqual(result.combination_count, exact_combination_count(generated.search_candidates, generated.standby_candidates)); self.assertEqual(result.status, "OK")




import unittest
from chapter3_bser.metrics import validate_small_instance
from tests.bser_test_utils import synthetic_instance


class JointStandbyBoundTest(unittest.TestCase):
    def test_finite_standby_enumeration_bound(self):
        _, _, generated, context = synthetic_instance(); check = validate_small_instance(generated.search_candidates, generated.standby_candidates, context)
        self.assertTrue(check["greedy_bound_pass"]); self.assertGreaterEqual(check["greedy_exact_ratio"], 0.5 - 1e-9)


