import unittes
from core.mapping.travel_cost_service import TravelCostService
from tests.bser_test_utils import synthetic_state

class PhysicalTimeEquivalenceTest(unittest.TestCase):
    def test_executor_physical_time_uses_executor_graph(self):
        state=synthetic_state(); result=TravelCostService(state).query((2.5,2.5,1),(1.5,1.5,1),state.agents[3]); self.assertAlmostEqual(result.physical_travel_time,2**.5/1.15,places=12)
