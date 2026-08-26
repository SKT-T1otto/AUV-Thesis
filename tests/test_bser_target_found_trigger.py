import unittest

from chapter3_bser.events.event_detector import EventDetector
from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.online.config import load_phase1b_config
from tests.bser_online_test_utils import state_at


class TargetFoundTriggerTest(unittest.TestCase):
    def test_false_to_true_is_immediate_event(self):
        detection = EventDetector(load_phase1b_config()).detect(state_at(0), state_at(1, target_found=True))
        self.assertIn(BSEREvent.TARGET_FOUND, detection.events)


if __name__ == "__main__": unittest.main()
