import unittest
from dataclasses import replace

from chapter3_bser.events.event_detector import EventDetector
from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.online.allocator import BSEROnlineAllocator
from chapter3_bser.online.config import load_phase1b2_config
from tests.bser_online_test_utils import shifted_belief, state_at


class Phase1B2ExecutorInvalidTest(unittest.TestCase):
    def setUp(self):
        self.config = load_phase1b2_config()
        self.detector = EventDetector(self.config)
        self.initial = state_at(0)
        self.allocation = BSEROnlineAllocator().allocate(self.initial)

    def test_belief_peak_change_does_not_invalidate_current_executor_task(self):
        result = self.detector.detect(
            self.initial,
            shifted_belief(1),
            assignment=self.allocation,
        )
        self.assertNotIn(BSEREvent.EXECUTOR_INVALID, result.events)

    def test_invalid_current_assignment_triggers(self):
        invalid_executor = replace(
            self.allocation.executor_assignment,
            reachable=False,
            failure_reason="TEST_ROUTE_INVALID",
        )
        invalid = replace(self.allocation, executor_assignment=invalid_executor)
        result = self.detector.detect(
            self.initial,
            state_at(1),
            assignment=invalid,
        )
        self.assertIn(BSEREvent.EXECUTOR_INVALID, result.events)


if __name__ == "__main__":
    unittest.main()
