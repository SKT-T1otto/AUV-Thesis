import unittest

from chapter3_bser.events.event_detector import EventDetector
from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.online.config import load_phase1b_config
from tests.bser_online_test_utils import shifted_belief, state_at


class BeliefTriggerTest(unittest.TestCase):
    def test_configured_belief_shift_triggers(self):
        detection = EventDetector(load_phase1b_config()).detect(state_at(0), shifted_belief())
        self.assertGreater(detection.belief_shift_score, 0.15)
        self.assertIn(BSEREvent.BELIEF_SHIFT, detection.events)
        self.assertGreater(detection.belief_distance, 0.0)


if __name__ == "__main__": unittest.main()
