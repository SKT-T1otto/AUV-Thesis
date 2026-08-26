import unittes
from core.mapping.travel_cost_service import TravelCostService
from tests.bser_test_utils import synthetic_state

class ReachabilityEquivalenceTest(unittest.TestCase):
    def test_all_valid_centers_are_reachable_in_reference_component(self):
        state=synthetic_state(); service=TravelCostService(state); agent=state.agents[0]
        self.assertTrue(all(service.query(agent.position,point,agent).reachable for point in state.grid.cell_centers))
