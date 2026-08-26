import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from core.scenarios.generator import E0_SEEDS, build_e0_manifests
from core.scenarios.registry import CANONICAL_THESIS_PROFILE, PROFILES


class ScenarioReproducibilityTests(unittest.TestCase):
    def test_fixed_manifests_reproduce(self):
        first = build_e0_manifests()
        second = build_e0_manifests()
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(PROFILES))
        self.assertEqual(CANONICAL_THESIS_PROFILE, "M20_MOVING_UNKNOWN_MULTI")
        for profile, generated in first.items():
            disk = json.loads((ROOT / "configs/scenarios/e0_equivalence" / f"{profile}.json").read_text(encoding="utf-8"))
            self.assertEqual(generated, disk)
            self.assertEqual(tuple(item["scenario_seed"] for item in generated["scenarios"]), E0_SEEDS)


if __name__ == "__main__":
    unittest.main()
