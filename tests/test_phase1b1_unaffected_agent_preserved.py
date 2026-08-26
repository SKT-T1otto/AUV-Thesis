import unittest

from chapter3_bser.online.allocator import BSEROnlineAllocator
from chapter3_bser.online.waypoint_manager import WaypointManager
from tests.bser_online_test_utils import shifted_belief, state_at


class UnaffectedAgentPreservedTest(unittest.TestCase):
    def test_stability_filter_restores_unaffected_assignments(self):
        allocator = BSEROnlineAllocator()
        old = allocator.allocate(state_at(0))
        proposed = allocator.allocate(shifted_belief())
        stable = WaypointManager(1.0).stabilize(old, proposed, affected_agent_ids=(0,), step=20)
        self.assertEqual(stable.search_assignments[1:], old.search_assignments[1:])


if __name__ == "__main__": unittest.main()
