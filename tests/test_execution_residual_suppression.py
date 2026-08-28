import unittest

import torch

from chapter3_bser.experiments.phase1c_prrac.execution_continuity import (
    ExecutionContinuityActionAdapter, ExecutionVariant, NavigationMode,
    plan_atomic_execution_continuity,
)
from tests.execution_continuity_test_support import FakeTravelCostService, state


class ResidualSuppressionTests(unittest.TestCase):
    def setUp(self):
        self.actions = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        self.hold = plan_atomic_execution_continuity(
            state(), (9.0, 0.0, 0.0), None,
            ExecutionVariant.B3_PROXY_SAFE_SUPPRESSION,
            service=FakeTravelCostService(),
        )

    def test_b3_execution_safe_hold_zeros_only_executor_copy(self):
        applied, diagnostic = ExecutionContinuityActionAdapter().apply(
            self.actions, plan=self.hold,
            variant=ExecutionVariant.B3_PROXY_SAFE_SUPPRESSION,
            mission_phase="EXECUTION",
        )
        self.assertTrue(torch.equal(applied[:3], self.actions[:3]))
        self.assertTrue(torch.equal(applied[3], torch.zeros(3)))
        self.assertTrue(torch.equal(self.actions, torch.arange(12).reshape(4, 3)))
        self.assertTrue(diagnostic.suppressed)
        self.assertEqual(diagnostic.applied_norm, 0.0)

    def test_b1_b2_and_active_routes_never_suppress(self):
        for variant in (ExecutionVariant.B1_ATOMIC_LAST_VALID, ExecutionVariant.B2_REACHABLE_PROXY):
            applied, diagnostic = ExecutionContinuityActionAdapter().apply(
                self.actions, plan=self.hold, variant=variant, mission_phase="EXECUTION"
            )
            self.assertTrue(torch.equal(applied, self.actions))
            self.assertFalse(diagnostic.suppressed)


if __name__ == "__main__":
    unittest.main()
