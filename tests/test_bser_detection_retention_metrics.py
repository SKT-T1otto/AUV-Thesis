import unittest
from chapter3_bser.greedy_solver import solve_joint_greedy
from chapter3_bser.objective import expected_detection_probability
from tests.bser_test_utils import synthetic_instance

class DetectionRetentionTest(unittest.TestCase):
    def test_retention_is_finite_without_threshold_gate(self):
        _,_,generated,context=synthetic_instance(); result=solve_joint_greedy(generated.search_candidates,generated.standby_candidates,context); value=expected_detection_probability(result.selected,context); self.assertGreaterEqual(value,0.0); self.assertLessEqual(value,1.0)
