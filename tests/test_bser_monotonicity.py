import unittest
from chapter3_bser.metrics import validate_small_instance
from tests.bser_test_utils import synthetic_instance


class MonotonicityTest(unittest.TestCase):
    def test_all_feasible_extensions(self):
        _, _, generated, context = synthetic_instance(); self.assertTrue(validate_small_instance(generated.search_candidates, generated.standby_candidates, context)["monotonicity_pass"])


if __name__ == "__main__": unittest.main()
