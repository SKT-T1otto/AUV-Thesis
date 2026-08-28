import unittest

from chapter3_bser.experiments.phase1c_prrac.execution_continuity import (
    ExecutionVariant, NavigationMode, plan_atomic_execution_continuity,
)
from tests.execution_continuity_test_support import FakeTravelCostService, previous_plan, state


class AtomicLastValidTests(unittest.TestCase):
    def test_exact_reachable_commits_exact(self):
        target = (3.0, 0.0, 0.0)
        plan = plan_atomic_execution_continuity(
            state(), target, previous_plan(), ExecutionVariant.B1_ATOMIC_LAST_VALID,
            service=FakeTravelCostService((target,)),
        )
        self.assertEqual(plan.navigation_mode, NavigationMode.EXACT_PUBLIC_TARGET)
        self.assertEqual(plan.semantic_target, target)
        self.assertTrue(plan.reachable)

    def test_failed_exact_refreshes_previous_endpoint_from_current_position(self):
        old = previous_plan((1.0, 0.0, 0.0))
        service = FakeTravelCostService((old.navigation_endpoint,))
        plan = plan_atomic_execution_continuity(
            state(executor_position=(0.5, 0.0, 0.0)), (9.0, 0.0, 0.0), old,
            ExecutionVariant.B1_ATOMIC_LAST_VALID, service=service,
        )
        self.assertEqual(plan.navigation_mode, NavigationMode.LAST_VALID_ROUTE)
        self.assertTrue(plan.preserved_from_previous)
        self.assertIn(((0.5, 0.0, 0.0), old.navigation_endpoint), service.queries)


if __name__ == "__main__":
    unittest.main()
