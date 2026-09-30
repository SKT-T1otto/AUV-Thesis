"""Closed public geometry and immutable graph filtering contract tests."""
from dataclasses import replace
import unittest

import numpy as np

from chapter3_bser.experiments.safe_search_v1.public_geometry import PublicGeometry, filter_planning_state
from core.mapping.planning_graph import EndpointConnectorSet, PlanningConnectorView, PlanningEdgeView
from core.mapping.planning_state import planning_state_sha256
from core.mapping.travel_cost_service import TravelCostService
from tests.bser_test_utils import synthetic_state


def fixture(occupied=()):
    state = synthetic_state()
    grid = replace(state.grid, origin=(0.0, 0.0, 0.5), spacing=(1.0, 1.0, 1.0))
    mask = np.zeros(9, dtype=np.bool_)
    mask[list(occupied)] = True
    unknown = ~mask
    occupancy = replace(state.occupancy, occupied_mask=mask, unknown_mask=unknown,
                        known_mask=mask.copy(), free_mask=np.zeros(9, dtype=np.bool_),
                        occupancy_probability=np.where(mask, 0.95, 0.5))
    return replace(state, grid=grid, occupancy=occupancy)


class PublicGeometryTests(unittest.TestCase):
    def test_closed_face_edge_corner_and_zero_length_contact(self):
        geometry = PublicGeometry(fixture((4,)))
        cases = [((0.2, 1, 1), (2.8, 1, 1)),
                 ((0.2, 1, 0.5), (2.8, 1, 0.5)),
                 ((0.5, 1.5, 1), (1.5, 0.5, 1)),
                 ((1, 1, 0.5), (1, 1, 0.5))]
        for left, right in cases:
            with self.subTest(left=left, right=right):
                self.assertFalse(geometry.segment_free(left, right))
                self.assertFalse(geometry.segment_free(right, left))
        self.assertTrue(geometry.segment_free((0.2, 0.999, 1), (2.8, 0.999, 1)))

    def test_clearance_is_explicit_closed_box_margin(self):
        state = fixture((4,))
        left, right = (0.2, 0.6, 1), (2.8, 0.6, 1)
        self.assertTrue(PublicGeometry(state, 0.39).segment_free(left, right))
        self.assertFalse(PublicGeometry(state, 0.4).segment_free(left, right))

    def test_narrow_corridor_closes_when_margin_consumes_width(self):
        state = fixture((1, 7))
        left, right = (1.5, 0.1, 1), (1.5, 2.9, 1)
        self.assertTrue(PublicGeometry(state, 0.49).segment_free(left, right))
        self.assertFalse(PublicGeometry(state, 0.5).segment_free(left, right))

    def test_unknown_remains_traversable_and_bounds_are_checked(self):
        geometry = PublicGeometry(fixture())
        self.assertTrue(geometry.segment_free((0, 0, 0.5), (3, 3, 1.5)))
        self.assertFalse(geometry.segment_free((-0.0001, 0, 1), (2, 2, 1)))
        self.assertFalse(geometry.point_free((1, 1, 1.5001)))

    def test_rejects_nonfinite_malformed_points_and_bad_margin(self):
        geometry = PublicGeometry(fixture())
        for bad in ((0, 0), (0, 0, 0, 0), (np.nan, 0, 1), (0, np.inf, 1), None):
            self.assertFalse(geometry.point_free(bad))
            self.assertFalse(geometry.segment_free((1, 1, 1), bad))
        for margin in (-0.1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                PublicGeometry(fixture(), margin)

    def test_empty_invalid_and_single_point_paths(self):
        geometry = PublicGeometry(fixture((4,)))
        self.assertFalse(geometry.path_free(()))
        self.assertFalse(geometry.path_free(None))
        self.assertTrue(geometry.path_free(((0.5, 0.5, 1),)))
        self.assertFalse(geometry.path_free(((0.5, 0.5, 1), (1.5, 1.5, 1))))

    def test_geometry_uses_a_copied_snapshot(self):
        state = fixture((4,))
        geometry = PublicGeometry(state)
        state.occupancy.occupied_mask[:] = False
        self.assertFalse(geometry.point_free((1.5, 1.5, 1)))
        with self.assertRaises(ValueError):
            geometry.occupied_lower.setflags(write=True)

    def test_degenerate_singleton_axis_and_bad_grid(self):
        self.assertTrue(PublicGeometry(synthetic_state()).point_free((0.5, 0.5, 1)))
        state = fixture()
        for grid in (replace(state.grid, spacing=(0, 1, 1)),
                     replace(state.grid, origin=(4, 0, 0)),
                     replace(state.grid, shape=(2, 3, 1))):
            with self.assertRaises(ValueError):
                PublicGeometry(replace(state, grid=grid))


class FilteredGraphTests(unittest.TestCase):
    def test_removes_occupied_nodes_corner_cutting_and_unsafe_endpoints(self):
        state = fixture((4,))
        filtered = filter_planning_state(state)
        graph = filtered.planning_graph
        self.assertFalse(graph.valid_mask[4])
        self.assertEqual(graph.component_labels[4], -1)
        self.assertNotIn(3, [edge.destination for edge in graph.searcher_adjacency[1]])
        self.assertIn(0, [edge.destination for edge in graph.searcher_adjacency[1]])
        nav = next(endpoint for endpoint in graph.endpoint_connectors if endpoint.endpoint_id == "navigation_target_0")
        self.assertFalse(nav.point_valid)
        self.assertEqual(nav.connectors, ())
        result = TravelCostService(filtered).query(state.agents[0].position, nav.point, state.agents[0])
        self.assertEqual(result.failure_reason, "invalid_goal")

    def test_surviving_costs_physical_times_and_role_heuristics_preserved(self):
        state = fixture((4,))
        graph = state.planning_graph
        rows = list(graph.searcher_adjacency)
        rows[0] = tuple(replace(edge, planning_cost=9.0, physical_travel_time=2.0)
                        if edge.destination == 1 else edge for edge in rows[0])
        state = replace(state, planning_graph=replace(graph, searcher_adjacency=tuple(rows)))
        filtered = filter_planning_state(state).planning_graph
        edge = next(edge for edge in filtered.searcher_adjacency[0] if edge.destination == 1)
        self.assertEqual((edge.planning_cost, edge.physical_travel_time), (9.0, 2.0))
        executor_edge = next(edge for edge in filtered.executor_adjacency[0] if edge.destination == 1)
        self.assertEqual(executor_edge, next(edge for edge in graph.executor_adjacency[0] if edge.destination == 1))
        self.assertNotEqual(edge.planning_cost, executor_edge.planning_cost)
        for name in ("searcher_horizontal_speed_divisor", "executor_horizontal_speed_divisor",
                     "searcher_vertical_cost_lookup", "executor_vertical_cost_lookup"):
            self.assertEqual(getattr(filtered, name), getattr(graph, name))

    def test_existing_astar_routes_around_closed_occupied_cell(self):
        state = fixture((4,))
        filtered = filter_planning_state(state)
        result = TravelCostService(filtered).query(state.agents[0].position, (2.5, 2.5, 1), state.agents[0])
        self.assertTrue(result.reachable)
        self.assertTrue(PublicGeometry(state).path_free(result.path_points))

    def test_true_disconnection_relabels_components_without_fake_connector(self):
        state = fixture((3, 4, 5))
        filtered = filter_planning_state(state)
        labels = filtered.planning_graph.component_labels
        self.assertNotEqual(labels[0], labels[6])
        endpoint = next(endpoint for endpoint in filtered.planning_graph.endpoint_connectors if endpoint.endpoint_id == "agent_1")
        self.assertEqual(endpoint.connectors[0].component_id, labels[6])
        result = TravelCostService(filtered).query(state.agents[0].position, (2.5, 0.5, 1), state.agents[0])
        self.assertEqual(result.failure_reason, "disconnected_endpoint_components")

    def test_legal_endpoint_without_safe_connector_remains_disconnected(self):
        state = fixture((4,))
        point = (0.6, 0.5, 1)
        endpoint = EndpointConnectorSet("current", "searcher", point, (PlanningConnectorView(4, 0, 4.2, 2.1),))
        state = replace(state, planning_graph=replace(state.planning_graph, endpoint_connectors=(endpoint,)))
        filtered = filter_planning_state(state)
        self.assertTrue(filtered.planning_graph.endpoint_connectors[0].point_valid)
        self.assertEqual(filtered.planning_graph.endpoint_connectors[0].connectors, ())
        result = TravelCostService(filtered).query(point, (0.5, 0.5, 1), state.agents[0])
        self.assertEqual(result.failure_reason, "no_start_connector")

    def test_missing_moved_endpoint_is_not_silently_snapped(self):
        state = filter_planning_state(fixture())
        agent = replace(state.agents[0], position=(0.513, 0.5, 1))
        result = TravelCostService(state).query(agent.position, (2.5, 2.5, 1), agent)
        self.assertEqual(result.failure_reason, "no_start_connector")

    def test_hash_repeatability_input_nonmutation_and_strong_immutability(self):
        state = fixture((4,))
        before = planning_state_sha256(state)
        first = filter_planning_state(state)
        second = filter_planning_state(state)
        self.assertEqual(planning_state_sha256(state), before)
        self.assertEqual(first.planning_graph.graph_sha256, second.planning_graph.graph_sha256)
        self.assertNotEqual(first.planning_graph.graph_sha256, state.planning_graph.graph_sha256)
        self.assertNotEqual(first.planning_graph.graph_sha256,
                            filter_planning_state(state, 0.4).planning_graph.graph_sha256)
        for value in (first.planning_graph.valid_mask, first.planning_graph.component_labels,
                      first.planning_graph.cell_centers):
            self.assertFalse(value.flags.writeable)
            with self.assertRaises(ValueError):
                value.setflags(write=True)

    def test_role_specific_topology_still_checked_by_original_astar(self):
        state = fixture()
        graph = replace(state.planning_graph, executor_adjacency=tuple(() for _ in range(9)))
        filtered = filter_planning_state(replace(state, planning_graph=graph))
        service = TravelCostService(filtered)
        self.assertTrue(service.query((0.5, 0.5, 1), (2.5, 2.5, 1), state.agents[0]).reachable)
        self.assertFalse(service.query((0.5, 0.5, 1), (2.5, 2.5, 1), state.agents[3]).reachable)

    def test_cached_geometry_keeps_current_costs_and_refreshes_new_occupancy(self):
        initial = fixture()
        first = filter_planning_state(initial)
        self.assertTrue(first.planning_graph.valid_mask[4])
        rows = list(initial.planning_graph.searcher_adjacency)
        rows[0] = tuple(replace(edge, planning_cost=15.0, physical_travel_time=6.0) for edge in rows[0])
        changed_costs = replace(initial, planning_graph=replace(initial.planning_graph, searcher_adjacency=tuple(rows)))
        second = filter_planning_state(changed_costs)
        self.assertTrue(all(edge.planning_cost == 15.0 and edge.physical_travel_time == 6.0
                            for edge in second.planning_graph.searcher_adjacency[0]))
        self.assertFalse(filter_planning_state(fixture((4,))).planning_graph.valid_mask[4])
        self.assertTrue(filter_planning_state(initial).planning_graph.valid_mask[4])

    def test_unsafe_continuous_start_is_invalid_not_snapped(self):
        state = fixture((4,))
        point = (1.0, 1.0, 1.0)
        endpoint = EndpointConnectorSet("current", "searcher", point, (PlanningConnectorView(0, 0, 1.0, 1.0),))
        state = replace(state, planning_graph=replace(state.planning_graph, endpoint_connectors=(endpoint,)))
        filtered = filter_planning_state(state)
        self.assertFalse(filtered.planning_graph.endpoint_connectors[0].point_valid)
        result = TravelCostService(filtered).query(point, (0.5, 0.5, 1), state.agents[0])
        self.assertEqual(result.failure_reason, "invalid_start")

    def test_bad_edge_index_or_cost_fails_closed(self):
        state = fixture()
        for edge in (PlanningEdgeView(99, 1.0, 1.0), PlanningEdgeView(1, np.nan, 1.0)):
            rows = list(state.planning_graph.searcher_adjacency)
            rows[0] = (edge,)
            with self.assertRaises(ValueError):
                filter_planning_state(replace(state, planning_graph=replace(state.planning_graph, searcher_adjacency=tuple(rows))))


if __name__ == "__main__":
    unittest.main()
