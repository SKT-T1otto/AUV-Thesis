from __future__ import annotations

import unittest

from chapter3_bser.experiments.phase1c_prrac.search_collision_recovery import SearchRecoveryVariant, build_search_recovery_controller, parse_search_recovery_variant


class VariantTests(unittest.TestCase):
    def test_registry_and_c0_no_state(self):
        self.assertIsNone(build_search_recovery_controller(SearchRecoveryVariant.S2A_C0_BASELINE))
        self.assertEqual(len(tuple(SearchRecoveryVariant)),3)
        with self.assertRaises(ValueError): parse_search_recovery_variant("C3")


if __name__ == "__main__": unittest.main()
