import unittest

from chapter3_bser.experiments.phase1c_prrac.execution_continuity import (
    ExecutionVariant, NavigationMode, plan_atomic_execution_continuity,
)
from tests.execution_continuity_test_support import FakeTravelCostService, previous_plan, state


class SafeHoldTests(unittest.TestCase):
    def test_no_route_enters_explicit_safe_hold_at_current_position(self):
        current = state(executor_position=(0.5, 0.0, 0.0))
        plan = plan_atomic_execution_continuity(
            current, (9.0, 0.0, 0.0), previous_plan(),
            ExecutionVariant.B1_ATOMIC_LAST_VALID,
            service=FakeTravelCostService(),
        )
        self.assertEqual(plan.navigation_mode, NavigationMode.SAFE_HOLD)
        self.assertEqual(plan.navigation_endpoint, (0.5, 0.0, 0.0))
        self.assertTrue(plan.safe_hold)
        self.assertFalse(plan.reachable)
        self.assertEqual(plan.path, ())


if __name__ == "__main__":
    unittest.main()
