import unittest

from chapter3_bser.events.event_detector import EventDetector
from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.online.config import load_phase1b1_config
from tests.bser_online_test_utils import mission_context, state_at


class ExecutorTargetReceivedTest(unittest.TestCase):
    def test_false_to_true_public_transition_is_distinct_event(self):
        previous = state_at(1, target_found=True)
        current = state_at(2, target_found=True)
        result = EventDetector(load_phase1b1_config()).detect(
            previous, current, mission_context(previous), mission_context(current, executor_knows_target=True)
        )
        self.assertIn(BSEREvent.EXECUTOR_TARGET_RECEIVED, result.events)
        self.assertNotIn(BSEREvent.TARGET_FOUND, result.events)


if __name__ == "__main__": unittest.main()
