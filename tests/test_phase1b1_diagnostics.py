import unittest

from chapter3_bser.online.config import load_phase1b1_config
from chapter3_bser.online.controller import OnlineBSERController
from tests.bser_online_test_utils import mission_context, state_at


class DiagnosticsTest(unittest.TestCase):
    def test_rejected_target_found_attempt_is_fully_recorded(self):
        controller = OnlineBSERController(load_phase1b1_config())
        initial = state_at(0)
        controller.initialize(initial, mission_context(initial))
        found = state_at(1, target_found=True)
        result = controller.step(found, mission_context(found))
        self.assertIsNotNone(result.diagnostics)
        self.assertFalse(result.diagnostics.accepted)
        self.assertEqual(result.diagnostics.reject_reason, "WAITING_FOR_PUBLIC_HANDOFF")
        self.assertIn("TARGET_FOUND", result.diagnostics.detected_events)


if __name__ == "__main__": unittest.main()
