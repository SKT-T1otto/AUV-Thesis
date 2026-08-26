import unittes
from chapter3_bser.objective import evaluate_objective
from chapter3_bser.baselines.search_only_allocator import solve_search_only_greedy
from tests.bser_test_utils import synthetic_instance

class ObjectiveScaleSeparationTest(unittest.TestCase):
    def test_search_only_own_scale_is_not_joint_scale(self):
        _,_,generated,context=synthetic_instance(); result=solve_search_only_greedy(generated.search_candidates,context); standby=generated.standby_candidates[0]; self.assertGreaterEqual(result.objective,evaluate_objective(result.selected,standby,context))
