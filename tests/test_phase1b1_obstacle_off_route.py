import unittest
from dataclasses import replace

from chapter3_bser.online.allocator import BSEROnlineAllocator
from chapter3_bser.online.route_impact import RouteImpactEvaluator
from tests.bser_online_test_utils import discovered_obstacle_at, state_at


class ObstacleOffRouteTest(unittest.TestCase):
    def test_new_cell_away_from_all_declared_paths_is_off_route(self):
        previous = state_at(0)
        allocation = BSEROnlineAllocator().allocate(previous)
        searches = tuple(replace(item, path_cell_indices=(0,)) for item in allocation.search_assignments)
        executor = replace(allocation.executor_assignment, path_cell_indices=(0,))
        result = RouteImpactEvaluator().evaluate(previous, discovered_obstacle_at(8), replace(allocation, search_assignments=searches, executor_assignment=executor))
        self.assertFalse(result.route_impacted)


if __name__ == "__main__": unittest.main()
