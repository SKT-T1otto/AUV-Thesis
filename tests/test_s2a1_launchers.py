from __future__ import annotations

import json
import unittest
from pathlib import Path

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator


class S2A1LauncherTests(unittest.TestCase):
    def test_config_and_launchers(self):
        root = Path(__file__).resolve().parents[1]
        config_path = root / "configs/chapter3/bser_phase1c_prrac_s2a1_local_connector_ablation.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["evaluation_episodes"], 100)
        self.assertEqual(config["search_collision_recovery"]["schema"], "bser.phase1c.prrac.search_collision_recovery.v2")
        self.assertEqual(len(evaluator._load_config(config_path)["search_collision_recovery_config_hash"]), 64)
        paths = (root / "scripts/run_phase1c_prrac_s2a1_local_connector_ablation.ps1",
                 root / "scripts/run_phase1c_prrac_s2a1_local_connector_ablation.bat",
                 root / "scripts/linux/run_phase1c_prrac_s2a1_local_connector_ablation.sh")
        for path in paths:
            text = path.read_text(encoding="utf-8")
            self.assertIn("S2A1_C2_LOCAL_CONNECTOR", text)
            self.assertIn("scenario-id-file", text)
            self.assertNotIn("train_phase", text)
        data = paths[-1].read_bytes()
        self.assertNotIn(b"\r\n", data)
        self.assertIn(b"set -euo pipefail", data)
        self.assertIn(b'conda activate "${CRK_CONDA_ENV:-AUV}"', data)


if __name__ == "__main__":
    unittest.main()
