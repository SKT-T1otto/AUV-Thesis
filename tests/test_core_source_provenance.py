import hashlib
import json
import unittest
from pathlib import Path

from core.registry.experiment_registry import ACTIVE_CH3_FINAL_EXPERIMENT_MODES


ROOT = Path(__file__).resolve().parents[1]
REVIEWED_POST_PHASE0B2_EVOLUTIONS = {
    "core/registry/experiment_registry.py": {
        "historical_sha256": (
            "769dad9c900af98bc0cb067632d2343db573fcd36c52bd176cba6966351f2b61"
        ),
        "current_sha256": (
            "8c735bdbe3e6bff0a56a8e4c120f9e65c236987721ad4ba7ec7b74e01d41a87c"
        ),
    }
}
HISTORICAL_ACTIVE_CH3_FINAL_EXPERIMENT_MODES = (
    "ch3_pheromone_prior",
    "ch3_pheromone_rmaddpg",
    "ch3_pse_rmaddpg",
    "ch3_pse_no_belief",
    "ch3_pse_no_exec_cost",
    "ch3_pse_no_standby",
    "ch3_pse_no_residual",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CoreSourceProvenanceTests(unittest.TestCase):
    def test_all_27_authority_records_map_to_current_core_files(self):
        path = ROOT / "docs/provenance/ch3_to_core_migration_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["authority_record_count"], 27)
        self.assertFalse(manifest["semantic_changes"])
        self.assertEqual(len(manifest["records"]), 27)
        reviewed = set()
        for record in manifest["records"]:
            self.assertFalse(record["semantic_changes"])
            relative = str(record["new_core_path"])
            target = ROOT / relative
            self.assertTrue(target.is_file(), relative)

            historical_sha = str(record["new_core_sha256"])
            current_sha = _sha256(target)
            evolution = REVIEWED_POST_PHASE0B2_EVOLUTIONS.get(relative)
            if evolution is None:
                self.assertEqual(
                    current_sha,
                    historical_sha,
                    f"unexpected post-Phase-0B-2 drift: {relative}",
                )
                continue

            reviewed.add(relative)
            self.assertEqual(
                historical_sha,
                evolution["historical_sha256"],
                f"historical provenance was rewritten: {relative}",
            )
            self.assertEqual(
                current_sha,
                evolution["current_sha256"],
                f"unreviewed drift after permitted evolution: {relative}",
            )

        self.assertEqual(reviewed, set(REVIEWED_POST_PHASE0B2_EVOLUTIONS))

    def test_reviewed_registry_evolution_preserves_historical_active_modes(self):
        self.assertEqual(
            ACTIVE_CH3_FINAL_EXPERIMENT_MODES,
            HISTORICAL_ACTIVE_CH3_FINAL_EXPERIMENT_MODES,
        )


if __name__ == "__main__":
    unittest.main()
