import unittest

from chapter3_bser.online.config import load_phase1b2_config
from chapter3_bser.online.controller import OnlineBSERController
from tests.bser_online_test_utils import mission_context, state_at


class Phase1B2WaypointConsistencyTest(unittest.TestCase):
    def test_detector_controller_and_allocation_share_canonical_waypoints(self):
        controller = OnlineBSERController(load_phase1b2_config())
        initial = state_at(0)
        controller.initialize(initial, mission_context(initial))
        current = state_at(1)
        result = controller.step(current, mission_context(current))

        canonical = controller.waypoints.current_assignment
        self.assertIs(canonical, controller.current_allocation)
        self.assertEqual(
            dict(result.event_detection.assignment_waypoints),
            controller.waypoints.waypoint_by_agent(),
        )
        self.assertEqual(result.allocation, canonical)


if __name__ == "__main__":
    unittest.main()
