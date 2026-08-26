import unittest

from chapter3_bser.online.config import load_phase1b1_config
from chapter3_bser.online.controller import OnlineBSERController
from tests.bser_online_test_utils import mission_context, state_at


class PublicHandoffPriorityTest(unittest.TestCase):
    def test_public_navigation_target_has_first_priority(self):
        controller = OnlineBSERController(load_phase1b1_config())
        initial = state_at(0)
        controller.initialize(initial, mission_context(initial))
        found = state_at(1, target_found=True)
        controller.step(found, mission_context(found))
        received = state_at(2, target_found=True)
        result = controller.step(received, mission_context(received, executor_knows_target=True))
        self.assertTrue(result.replanned)
        self.assertEqual(result.allocation.executor_assignment.source, "PUBLIC_HANDOFF_TARGET")
        self.assertEqual(result.allocation.search_assignments, controller.current_allocation.search_assignments)


if __name__ == "__main__": unittest.main()
