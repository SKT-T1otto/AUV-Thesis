import unittest
from chapter3_bser.objective import evaluate_objective, marginal_gain
from tests.bser_test_utils import synthetic_instance


class ObjectiveTest(unittest.TestCase):
    def test_empty_zero_and_marginal_identity(self):
        _, _, generated, context = synthetic_instance(); y = generated.standby_candidates[0]; e = generated.search_candidates[0]
        self.assertEqual(evaluate_objective((), y, context), 0.0); self.assertAlmostEqual(marginal_gain((), e, y, context), evaluate_objective((e,), y, context))


if __name__ == "__main__": unittest.main()
