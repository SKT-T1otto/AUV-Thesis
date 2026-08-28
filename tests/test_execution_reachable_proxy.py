import unittest

from chapter3_bser.experiments.phase1c_prrac.execution_continuity import (
    ExecutionVariant, NavigationMode, assign_reachable_public_proxy,
    plan_atomic_execution_continuity,
)
from tests.execution_continuity_test_support import FakeTravelCostService, state


class ReachableProxyTests(unittest.TestCase):
    def test_proxy_is_deterministic_and_preserves_semantic_target(self):
        semantic = (4.0, 0.0, 0.0)
        service = FakeTravelCostService(((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
        first = assign_reachable_public_proxy(state(), semantic, service=service)
        second = assign_reachable_public_proxy(
            state(), semantic,
            service=FakeTravelCostService(((1.0, 0.0, 0.0), (2.0, 0.0, 0.0))),
        )
        self.assertEqual(first[0], (2.0, 0.0, 0.0))
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[2], second[2])

        plan = plan_atomic_execution_continuity(
            state(), semantic, None, ExecutionVariant.B2_REACHABLE_PROXY,
            service=FakeTravelCostService(((2.0, 0.0, 0.0),)),
        )
        self.assertEqual(plan.navigation_mode, NavigationMode.REACHABLE_PUBLIC_PROXY)
        self.assertEqual(plan.semantic_target, semantic)
        self.assertEqual(plan.navigation_endpoint, (2.0, 0.0, 0.0))
        self.assertNotEqual(plan.semantic_target, plan.navigation_endpoint)


if __name__ == "__main__":
    unittest.main()
