import unittest
from dataclasses import replace

from chapter3_bser.online.config import load_phase1b2_config
from chapter3_bser.online.controller import OnlineBSERController
from tests.bser_online_test_utils import mission_context, shifted_belief, state_at


class Phase1B21ExecutionTargetLockTest(unittest.TestCase):
    def test_belief_shift_cannot_replace_public_execution_target(self):
        controller = OnlineBSERController(load_phase1b2_config())
        initial = state_at(0)
        controller.initialize(initial, mission_context(initial))

        found = state_at(1, target_found=True)
        controller.step(
            found,
            mission_context(found, target_found=True, executor_knows_target=False),
        )

        received = state_at(2, target_found=True)
        received_context = mission_context(
            received,
            target_found=True,
            executor_knows_target=True,
        )
        handoff = controller.step(received, received_context)
        locked = tuple(received_context.executor_navigation_target)
        self.assertEqual(controller.execution_target, locked)
        self.assertEqual(handoff.allocation.executor_assignment.target_region, locked)

        shifted = replace(shifted_belief(3), target_found=True)
        after_shift = controller.step(
            shifted,
            mission_context(
                shifted,
                target_found=True,
                executor_knows_target=True,
            ),
        )
        self.assertEqual(after_shift.allocation.executor_assignment.target_region, locked)
        self.assertTrue(after_shift.allocation.search_frozen)


if __name__ == "__main__":
    unittest.main()
