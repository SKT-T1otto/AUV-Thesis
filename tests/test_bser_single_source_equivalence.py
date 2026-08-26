import unittest
from core.mapping.travel_cost_service import TravelCostService
from tests.bser_test_utils import synthetic_state

class SingleSourceEquivalenceTest(unittest.TestCase):
    def test_single_source_matches_scalar_queries(self):
        state=synthetic_state(); service=TravelCostService(state); agent=state.agents[0]; tree=service.single_source(agent.position,agent)
        for index,point in enumerate(state.grid.cell_centers): self.assertAlmostEqual(tree.planning_cost_by_cell[index],service.query(agent.position,point,agent).planning_cost,places=12)
