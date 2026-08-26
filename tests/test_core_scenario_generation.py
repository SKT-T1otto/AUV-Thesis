import json
import unittest
from pathlib import Path

from core.scenarios.generator import E0_SEEDS, build_e0_manifests


ROOT = Path(__file__).resolve().parents[1]


class CoreScenarioGenerationTests(unittest.TestCase):
    def test_core_regenerates_frozen_manifests_exactly(self):
        generated = build_e0_manifests()
        self.assertEqual(len(generated), 4)
        for profile, manifest in generated.items():
            frozen = json.loads((ROOT / "configs/scenarios/e0_equivalence" / f"{profile}.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest, frozen)
            self.assertEqual(tuple(item["scenario_seed"] for item in manifest["scenarios"]), E0_SEEDS)
            self.assertTrue(all("scenario_sha256" in item for item in manifest["scenarios"]))


if __name__ == "__main__":
    unittest.main()
