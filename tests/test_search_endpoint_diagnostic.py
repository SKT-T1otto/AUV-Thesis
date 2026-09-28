"""Synthetic mechanism check: continuous start != stale graph endpoint.

This is not a replay of the imported 3090 experiment and does not change the
shared planner or accept an unchecked connector in production.
"""
from dataclasses import replace
import unittest

from core.mapping.planning_graph import EndpointConnectorSet, PlanningConnectorView
from core.mapping.travel_cost_service import TravelCostService
from tests.bser_test_utils import synthetic_state


class SearchEndpointDiagnosticTests(unittest.TestCase):
    def test_small_continuous_motion_loses_connector_until_endpoint_refresh(self):
        state=synthetic_state();agent=state.agents[0];goal=state.grid.cell_centers[-1]
        self.assertTrue(TravelCostService(state).query(agent.position,goal,agent).reachable)
        moved=replace(agent,position=(agent.position[0]+0.013,*agent.position[1:]))
        stale=replace(state,agents=(moved,*state.agents[1:]))
        self.assertEqual(TravelCostService(stale).query(moved.position,goal,moved).failure_reason,"no_start_connector")
        # Fixture's empty, connected graph: a 0.013-long segment back to its
        # original center lies in the known-free initial cell.
        endpoint=next(e for e in state.planning_graph.endpoint_connectors if e.point==agent.position and e.role==agent.role)
        old=endpoint.connectors[0]
        connector=PlanningConnectorView(old.cell_index,old.component_id,0.013,0.013)
        new=EndpointConnectorSet("diagnostic_current_start",agent.role,moved.position,(connector,))
        fresh=replace(stale,planning_graph=replace(stale.planning_graph,
                      endpoint_connectors=(*stale.planning_graph.endpoint_connectors,new)))
        query=TravelCostService(fresh).query(moved.position,goal,moved)
        self.assertTrue(query.reachable)
        self.assertEqual(tuple(query.path_points[0]),moved.position)
        self.assertEqual(stale.planning_graph,state.planning_graph)


if __name__=="__main__":unittest.main()
