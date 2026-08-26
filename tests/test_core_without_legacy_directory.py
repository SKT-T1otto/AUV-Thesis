import unittest
from pathlib import Path

from core.env import MissionCoreEnv


ROOT = Path(__file__).resolve().parents[1]


class CoreWithoutHistoricalDirectoryTests(unittest.TestCase):
    def test_historical_area_contains_no_python_and_is_not_required(self):
        historical = ROOT / "legacy_adapters"
        python_files = [] if not historical.exists() else list(historical.rglob("*.py"))
        self.assertEqual(python_files, [])
        self.assertEqual(MissionCoreEnv.implementation_source, "core")


if __name__ == "__main__":
    unittest.main()
