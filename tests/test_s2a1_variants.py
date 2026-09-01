from __future__ import annotations

import json
import unittest
from pathlib import Path

from chapter3_bser.experiments.phase1c_prrac.search_collision_recovery import (
    SEARCH_COLLISION_RECOVERY_SCHEMA,
    SEARCH_COLLISION_RECOVERY_SCHEMA_V2,
    SearchRecoveryVariant,
    SearchRecoveryVariantV2,
    build_search_recovery_controller,
    parse_search_recovery_variant,
    search_collision_recovery_config,
    search_collision_recovery_config_hash,
)


class S2A1VariantTests(unittest.TestCase):
    def test_v1_contract_and_hash_are_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        source = json.loads((root / "configs/chapter3/bser_phase1c_prrac_s2a_collision_ablation.json").read_text(encoding="utf-8"))
        resolved = search_collision_recovery_config(source)
        self.assertEqual(resolved["schema"], SEARCH_COLLISION_RECOVERY_SCHEMA)
        self.assertEqual(len(search_collision_recovery_config_hash(resolved)), 64)
        self.assertEqual(tuple(SearchRecoveryVariant), (
            SearchRecoveryVariant.S2A_C0_BASELINE,
            SearchRecoveryVariant.S2A_C1_ROUTE_REFRESH,
            SearchRecoveryVariant.S2A_C2_EGRESS_ROUTE,
        ))

    def test_v2_variants_and_c0_no_controller(self):
        root = Path(__file__).resolve().parents[1]
        source = json.loads((root / "configs/chapter3/bser_phase1c_prrac_s2a1_local_connector_ablation.json").read_text(encoding="utf-8"))
        resolved = search_collision_recovery_config(source)
        self.assertEqual(resolved["schema"], SEARCH_COLLISION_RECOVERY_SCHEMA_V2)
        self.assertIs(parse_search_recovery_variant("S2A1_C2_LOCAL_CONNECTOR"), SearchRecoveryVariantV2.S2A1_C2_LOCAL_CONNECTOR)
        self.assertIsNone(build_search_recovery_controller(SearchRecoveryVariantV2.S2A1_C0_BASELINE))

    def test_v1_and_v2_hashes_are_resume_isolated(self):
        root = Path(__file__).resolve().parents[1]
        values = []
        for name in ("bser_phase1c_prrac_s2a_collision_ablation.json", "bser_phase1c_prrac_s2a1_local_connector_ablation.json"):
            config = json.loads((root / "configs/chapter3" / name).read_text(encoding="utf-8"))
            values.append(search_collision_recovery_config_hash(search_collision_recovery_config(config)))
        self.assertNotEqual(*values)


if __name__ == "__main__":
    unittest.main()
