import unittest
from dataclasses import replace

from chapter3_bser.online.allocator import BSEROnlineAllocator
from chapter3_bser.online.route_impact import RouteImpactEvaluator
from tests.bser_online_test_utils import discovered_obstacle, state_at


class ObstacleRouteImpactTest(unittest.TestCase):
    def test_path_intersection_marks_only_declared_route(self):
        previous = state_at(0)
        allocation = BSEROnlineAllocator().allocate(previous)
        first = replace(allocation.search_assignments[0], path_cell_indices=(4,), planning_cost=1.0)
        others = tuple(replace(item, path_cell_indices=(0,), planning_cost=1.0) for item in allocation.search_assignments[1:])
        executor = replace(allocation.executor_assignment, path_cell_indices=(0,), planning_cost=1.0)
        result = RouteImpactEvaluator().evaluate(previous, discovered_obstacle(), replace(allocation, search_assignments=(first,)+others, executor_assignment=executor))
        self.assertTrue(result.route_impacted)
        self.assertIn(first.agent_id, result.affected_searcher_ids)


if __name__ == "__main__": unittest.main()
