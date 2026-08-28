import unittest
from types import SimpleNamespace

from chapter3_bser.experiments.phase1c_prrac.execution_continuity import (
    ExecutionContinuityEventDetector, ExecutionVariant, NavigationMode,
    ExecutionContinuityController, plan_atomic_execution_continuity,
)
from tests.execution_continuity_test_support import FakeTravelCostService, state


class ExecutionContinuityEventDetectorTests(unittest.TestCase):
    def test_proxy_validity_uses_proxy_endpoint_not_semantic_target(self):
        semantic = (9.0, 0.0, 0.0)
        proxy = (2.0, 0.0, 0.0)
        plan = plan_atomic_execution_continuity(
            state(), semantic, None, ExecutionVariant.B2_REACHABLE_PROXY,
            service=FakeTravelCostService((proxy,)),
        )
        self.assertEqual(plan.navigation_mode, NavigationMode.REACHABLE_PUBLIC_PROXY)
        service = FakeTravelCostService((proxy,))
        detected = ExecutionContinuityEventDetector().detect(
            state(), plan, semantic, retry_due=False, service=service
        )
        self.assertFalse(detected.route_invalid)
        self.assertIn("PROXY_ACTIVE", detected.events)
        self.assertNotIn("ASSIGNMENT_UNREACHABLE", detected.events)
        self.assertEqual(service.queries[-1][1], proxy)

    def test_safe_hold_is_retry_pending_not_assignment_unreachable(self):
        plan = plan_atomic_execution_continuity(
            state(), (9.0, 0.0, 0.0), None,
            ExecutionVariant.B2_REACHABLE_PROXY,
            service=FakeTravelCostService(),
        )
        detected = ExecutionContinuityEventDetector().detect(
            state(), plan, plan.semantic_target, retry_due=False,
            service=FakeTravelCostService(),
        )
        self.assertEqual(detected.events, ("SAFE_HOLD_RETRY_PENDING",))
        self.assertFalse(detected.validity_evaluated)

    def test_retry_is_aligned_to_existing_refresh_cadence_and_shift_threshold(self):
        legacy = SimpleNamespace(
            detector=SimpleNamespace(dynamic_public_target_enabled=True)
        )
        controller = ExecutionContinuityController(
            legacy,
            variant=ExecutionVariant.B2_REACHABLE_PROXY,
            config={
                "execution_runtime": {
                    "public_target_update_distance": 0.75,
                    "public_target_update_min_steps": 20,
                    "defer_stale_endpoint_invalid": True,
                },
                "execution_continuity": {
                    "state_refresh_interval": 20,
                    "executor_cost_increase_threshold": 0.15,
                },
            },
        )
        controller.current_plan = plan_atomic_execution_continuity(
            state(), (4.0, 0.0, 0.0), None,
            ExecutionVariant.B2_REACHABLE_PROXY,
            service=FakeTravelCostService(((2.0, 0.0, 0.0),)),
        )
        controller.last_plan_attempt_step = 7
        controller.last_semantic_update_step = 0
        self.assertFalse(controller._retry_due(19))
        self.assertTrue(controller._retry_due(20))
        self.assertFalse(controller._semantic_shift_due((4.8, 0.0, 0.0), 19))
        self.assertTrue(controller._semantic_shift_due((4.8, 0.0, 0.0), 20))


if __name__ == "__main__":
    unittest.main()
