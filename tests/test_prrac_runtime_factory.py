import unittest
from types import SimpleNamespace
from unittest import mock

from chapter3_bser.experiments.phase1c_prrac.execution_continuity import (
    ExecutionContinuityController,
    ExecutionVariant,
)
from chapter3_bser.experiments.phase1c_prrac.runtime_factory import (
    NATIVE_B1_RUNTIME_REVISION,
    build_prrac_online_controller,
)
from chapter3_bser.online.config import load_phase1b2_config
from chapter3_bser.online.controller import OnlineBSERController


class PRRACRuntimeFactoryTests(unittest.TestCase):
    def test_legacy_overlay_and_native_are_strict(self):
        phase = load_phase1b2_config()
        legacy = build_prrac_online_controller(phase, {})
        self.assertIsInstance(legacy, OnlineBSERController)
        overlay = build_prrac_online_controller(
            phase,
            {"execution_variant": "B1_ATOMIC_LAST_VALID", "runtime_integration_mode": "overlay"},
        )
        native = build_prrac_online_controller(
            phase,
            {
                "execution_runtime_revision": NATIVE_B1_RUNTIME_REVISION,
                "execution_variant": "B1_ATOMIC_LAST_VALID",
                "runtime_integration_mode": "native",
            },
        )
        self.assertIsInstance(overlay, ExecutionContinuityController)
        self.assertIsInstance(native, ExecutionContinuityController)
        self.assertEqual(native.variant, ExecutionVariant.B1_ATOMIC_LAST_VALID)
        self.assertEqual(native.prrac_runtime_contract.runtime_integration_mode, "native")
        evaluation = build_prrac_online_controller(
            phase,
            {
                "execution_runtime_revision": NATIVE_B1_RUNTIME_REVISION,
                "execution_variant": "B1_ATOMIC_LAST_VALID",
                "runtime_integration_mode": "native",
            },
        )
        self.assertEqual(type(native), type(evaluation))
        self.assertEqual(native.config, evaluation.config)

    def test_b1_delegates_search_without_overlaying_allocation(self):
        initialized = object()
        stepped = SimpleNamespace(allocation=object())
        legacy = SimpleNamespace(
            detector=SimpleNamespace(dynamic_public_target_enabled=True),
            initialize=mock.Mock(return_value=initialized),
            step=mock.Mock(return_value=stepped),
        )
        controller = ExecutionContinuityController(
            legacy, variant="B1_ATOMIC_LAST_VALID", config={}
        )
        context = SimpleNamespace(
            target_found=False, executor_knows_target=False,
            mission_complete=False, executor_navigation_target=None,
        )
        state = SimpleNamespace(step=0)
        self.assertIs(controller.initialize(state, context), initialized)
        self.assertIs(controller.step(state, context), stepped)
        self.assertIsNone(controller.current_plan)

    def test_native_rejects_proxy_variants(self):
        with self.assertRaisesRegex(ValueError, "only B1"):
            build_prrac_online_controller(
                load_phase1b2_config(),
                {
                    "execution_runtime_revision": NATIVE_B1_RUNTIME_REVISION,
                    "execution_variant": "B2_REACHABLE_PROXY",
                    "runtime_integration_mode": "native",
                },
            )


if __name__ == "__main__":
    unittest.main()
