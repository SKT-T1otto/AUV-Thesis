import json
import tempfile
import unittest
from pathlib import Path

from tools.run_core_training_smoke import run_smoke


class CoreTrainingSmokeTests(unittest.TestCase):
    def test_two_by_ten_training_closure(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary = run_smoke(Path(temporary))
            disk = json.loads((Path(temporary) / "training_smoke_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary, disk)
        self.assertEqual(summary["status"], "PASS")
        self.assertTrue(summary["critic_update_completed"])
        self.assertTrue(summary["actor_update_completed"])
        self.assertTrue(summary["checkpoint_roundtrip_passed"])
        self.assertTrue(summary["post_load_step_completed"])
        self.assertFalse(summary["checkpoint_persisted_in_repository"])


if __name__ == "__main__":
    unittest.main()
