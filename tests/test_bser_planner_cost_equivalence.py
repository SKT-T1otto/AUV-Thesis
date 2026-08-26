import unittes
from core.mapping.travel_cost_service import TravelCostService
from tests.bser_test_utils import synthetic_state

class CostEquivalenceTest(unittest.TestCase):
    def test_diagonal_cost_matches_authoritative_edge(self):
        state=synthetic_state(); result=TravelCostService(state).query((.5,.5,1),(1.5,1.5,1),state.agents[0]); self.assertAlmostEqual(result.planning_cost,2**.5,places=12)
