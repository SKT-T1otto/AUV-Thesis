"""Current bounded behavior checks; original TestCase bodies retained."""


import unittest
from core.mapping.travel_cost_service import TravelCostService
from tests.bser_test_utils import synthetic_state

class ConnectorEquivalenceTest(unittest.TestCase):
    def test_registered_agent_endpoint_has_zero_connector(self):
        state=synthetic_state(); service=TravelCostService(state); result=service.query(state.agents[0].position,state.agents[0].position,state.agents[0]); self.assertEqual(result.planning_cost,0.0); self.assertEqual(len(result.path_cell_indices),0)


import unittest
from core.mapping.travel_cost_service import TravelCostService
from tests.bser_test_utils import synthetic_state

class CostEquivalenceTest(unittest.TestCase):
    def test_diagonal_cost_matches_authoritative_edge(self):
        state=synthetic_state(); result=TravelCostService(state).query((.5,.5,1),(1.5,1.5,1),state.agents[0]); self.assertAlmostEqual(result.planning_cost,2**.5,places=12)


import unittest
from core.mapping.travel_cost_service import TravelCostService
from tests.bser_test_utils import synthetic_state

class CollisionEquivalenceTest(unittest.TestCase):
    def test_returned_cells_are_valid_authoritative_edges(self):
        state=synthetic_state(); result=TravelCostService(state).query((.5,.5,1),(2.5,2.5,1),state.agents[0]); adjacency=state.planning_graph.searcher_adjacency
        self.assertTrue(all(state.planning_graph.valid_mask[i] for i in result.path_cell_indices))
        self.assertTrue(all(any(edge.destination==right for edge in adjacency[left]) for left,right in zip(result.path_cell_indices,result.path_cell_indices[1:])))


import unittest
from core.mapping.travel_cost_service import TravelCostService
from tests.bser_test_utils import synthetic_state

class PhysicalTimeEquivalenceTest(unittest.TestCase):
    def test_executor_physical_time_uses_executor_graph(self):
        state=synthetic_state(); result=TravelCostService(state).query((2.5,2.5,1),(1.5,1.5,1),state.agents[3]); self.assertAlmostEqual(result.physical_travel_time,2**.5/1.15,places=12)


import unittest
from core.mapping.travel_cost_service import TravelCostService
from tests.bser_test_utils import synthetic_state

class ReachabilityEquivalenceTest(unittest.TestCase):
    def test_all_valid_centers_are_reachable_in_reference_component(self):
        state=synthetic_state(); service=TravelCostService(state); agent=state.agents[0]
        self.assertTrue(all(service.query(agent.position,point,agent).reachable for point in state.grid.cell_centers))
