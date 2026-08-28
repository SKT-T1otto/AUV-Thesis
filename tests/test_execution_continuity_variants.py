import unittest

from chapter3_bser.experiments.phase1c_prrac.execution_continuity import (
    ExecutionVariant,
    VARIANT_ORDER,
    overlay_enabled,
    parse_execution_variant,
)


class ExecutionContinuityVariantTests(unittest.TestCase):
    def test_registry_is_exactly_the_four_frozen_variants(self):
        self.assertEqual(tuple(ExecutionVariant), VARIANT_ORDER)
        self.assertEqual([item.name for item in VARIANT_ORDER], [
            "B0_LEGACY_V2_1", "B1_ATOMIC_LAST_VALID",
            "B2_REACHABLE_PROXY", "B3_PROXY_SAFE_SUPPRESSION",
        ])
        self.assertFalse(overlay_enabled(ExecutionVariant.B0_LEGACY_V2_1))
        self.assertTrue(overlay_enabled(ExecutionVariant.B1_ATOMIC_LAST_VALID))

    def test_arbitrary_combinations_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_execution_variant("B2+SUPPRESSION_OFF")


if __name__ == "__main__":
    unittest.main()
