import unittest

from chapter3_bser.online.config import load_phase1b1_config
from chapter3_bser.online.controller import OnlineBSERController
from tests.bser_online_test_utils import mission_context, state_at


class TargetFoundPreservesRouteTest(unittest.TestCase):
    def test_waits_for_public_handoff(self):
        controller = OnlineBSERController(load_phase1b1_config())
        initial = state_at(0)
        original = controller.initialize(initial, mission_context(initial)).allocation
        found = state_at(1, target_found=True)
        result = controller.step(found, mission_context(found))
        self.assertFalse(result.replanned)
        self.assertEqual(result.decision_reason, "WAITING_FOR_PUBLIC_HANDOFF")
        self.assertEqual(result.allocation.executor_assignment, original.executor_assignment)
        self.assertEqual(result.allocation.search_assignments, original.search_assignments)


if __name__ == "__main__": unittest.main()
