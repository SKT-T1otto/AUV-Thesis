import unittest

from chapter3_bser.online.allocator import BSEROnlineAllocator
from chapter3_bser.online.waypoint_manager import WaypointManager
from tests.bser_online_test_utils import shifted_belief, state_at


class WaypointUpdateTest(unittest.TestCase):
    def test_only_changed_high_level_waypoints_are_reported(self):
        allocator = BSEROnlineAllocator()
        first = allocator.allocate(state_at(0))
        same = WaypointManager().updates(first, first, reason="same", step=1)
        self.assertEqual(same, ())
        second = allocator.allocate(shifted_belief())
        changed = WaypointManager().updates(first, second, reason="belief", step=1)
        self.assertLessEqual(len(changed), 3)


if __name__ == "__main__": unittest.main()
