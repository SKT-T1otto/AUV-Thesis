import unittest
from dataclasses import replace

from chapter3_bser.online.allocator import BSEROnlineAllocator
from chapter3_bser.online.waypoint_manager import WaypointManager
from tests.bser_online_test_utils import state_at


class MinimumSwitchDistanceTest(unittest.TestCase):
    def test_subunit_waypoint_change_is_suppressed(self):
        allocation = BSEROnlineAllocator().allocate(state_at(0))
        first = allocation.search_assignments[0]
        moved = replace(first, candidate_id=first.candidate_id+"x", waypoint=(first.waypoint[0]+0.5,first.waypoint[1],first.waypoint[2]))
        proposed = replace(allocation, search_assignments=(moved,)+allocation.search_assignments[1:])
        stable = WaypointManager(1.0).stabilize(allocation, proposed, affected_agent_ids=(first.agent_id,), step=20)
        self.assertEqual(stable.search_assignments[0], first)


if __name__ == "__main__": unittest.main()
