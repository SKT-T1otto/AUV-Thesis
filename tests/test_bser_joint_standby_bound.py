import unittest
from chapter3_bser.metrics import validate_small_instance
from tests.bser_test_utils import synthetic_instance


class JointStandbyBoundTest(unittest.TestCase):
    def test_finite_standby_enumeration_bound(self):
        _, _, generated, context = synthetic_instance(); check = validate_small_instance(generated.search_candidates, generated.standby_candidates, context)
        self.assertTrue(check["greedy_bound_pass"]); self.assertGreaterEqual(check["greedy_exact_ratio"], 0.5 - 1e-9)


if __name__ == "__main__": unittest.main()
