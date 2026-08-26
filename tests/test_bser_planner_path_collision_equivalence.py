import unittest
from core.mapping.travel_cost_service import TravelCostService
from tests.bser_test_utils import synthetic_state

class CollisionEquivalenceTest(unittest.TestCase):
    def test_returned_cells_are_valid_authoritative_edges(self):
        state=synthetic_state(); result=TravelCostService(state).query((.5,.5,1),(2.5,2.5,1),state.agents[0]); adjacency=state.planning_graph.searcher_adjacency
        self.assertTrue(all(state.planning_graph.valid_mask[i] for i in result.path_cell_indices))
        self.assertTrue(all(any(edge.destination==right for edge in adjacency[left]) for left,right in zip(result.path_cell_indices,result.path_cell_indices[1:])))
