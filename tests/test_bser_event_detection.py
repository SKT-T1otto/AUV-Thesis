import unittest

from chapter3_bser.events.event_detector import EventDetector
from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.online.config import load_phase1b_config
from tests.bser_online_test_utils import state_at


class EventDetectionTest(unittest.TestCase):
    def test_event_enum_and_detection_are_deterministic(self):
        self.assertEqual({event.value for event in BSEREvent}, {
            "BELIEF_SHIFT", "OBSTACLE_DISCOVERED", "TARGET_FOUND",
            "EXECUTOR_TARGET_RECEIVED", "EXECUTOR_PUBLIC_TARGET_UPDATED",
            "TARGET_LOST", "EXECUTOR_INVALID", "WAYPOINT_STALE", "PERIODIC_REFRESH",
        })
        detector = EventDetector(load_phase1b_config())
        previous = state_at(0)
        current = state_at(100)
        first = detector.detect(previous, current)
        second = detector.detect(previous, current)
        self.assertEqual(first, second)
        self.assertIn(BSEREvent.PERIODIC_REFRESH, first.events)
        self.assertNotIn(BSEREvent.EXECUTOR_PUBLIC_TARGET_UPDATED, first.events)
        lost = detector.detect(state_at(0, target_found=True), state_at(1))
        self.assertIn(BSEREvent.TARGET_LOST, lost.events)
        # Public-target movement/refresh behavior is exercised by
        # test_phase1c_dynamic_public_target_update on the real controller.


if __name__ == "__main__": unittest.main()
