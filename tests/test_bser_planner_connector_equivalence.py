import unittes
from core.mapping.travel_cost_service import TravelCostService
from tests.bser_test_utils import synthetic_state

class ConnectorEquivalenceTest(unittest.TestCase):
    def test_registered_agent_endpoint_has_zero_connector(self):
        state=synthetic_state(); service=TravelCostService(state); result=service.query(state.agents[0].position,state.agents[0].position,state.agents[0]); self.assertEqual(result.planning_cost,0.0); self.assertEqual(len(result.path_cell_indices),0)
