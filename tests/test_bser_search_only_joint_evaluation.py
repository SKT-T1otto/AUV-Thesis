import unittest
from chapter3_bser.baselines.search_only_allocator import solve_search_only_greedy
from chapter3_bser.objective import evaluate_objective
from tests.bser_test_utils import synthetic_instance

class SearchOnlyJointEvaluationTest(unittest.TestCase):
    def test_selected_allocation_can_be_evaluated_with_joint_objective(self):
        _,_,generated,context=synthetic_instance(); result=solve_search_only_greedy(generated.search_candidates,context); values=[evaluate_objective(result.selected,y,context) for y in generated.standby_candidates]; self.assertTrue(all(value>=0 for value in values)); self.assertGreaterEqual(max(values),0)
