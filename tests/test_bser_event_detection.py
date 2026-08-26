import unittest

from chapter3_bser.events.event_detector import EventDetector
from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.online.config import load_phase1b_config
from tests.bser_online_test_utils import state_at


class EventDetectionTest(unittest.TestCase):
    def test_event_enum_and_detection_are_deterministic(self):
        self.assertEqual(len(BSEREvent), 8)
        detector = EventDetector(load_phase1b_config())
        previous = state_at(0)
        current = state_at(100)
        first = detector.detect(previous, current)
        second = detector.detect(previous, current)
        self.assertEqual(first, second)
        self.assertIn(BSEREvent.PERIODIC_REFRESH, first.events)


if __name__ == "__main__": unittest.main()
