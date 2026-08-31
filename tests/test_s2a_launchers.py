from __future__ import annotations

import json
import unittest
from pathlib import Path

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator


class LauncherTests(unittest.TestCase):
    def test_config_and_launchers_are_frozen(self):
        root=Path(__file__).resolve().parents[1]
        config=json.loads((root/"configs/chapter3/bser_phase1c_prrac_s2a_collision_ablation.json").read_text(encoding="utf-8"))
        self.assertEqual(config["evaluation_episodes"],100); self.assertEqual(config["modes"],["full_prrac"]); self.assertEqual(len(config["search_recovery_variants"]),3); self.assertEqual(config["checkpoints"],[])
        for path in (root/"scripts/run_phase1c_prrac_s2a_collision_ablation.ps1",root/"scripts/run_phase1c_prrac_s2a_collision_ablation.bat",root/"scripts/linux/run_phase1c_prrac_s2a_collision_ablation.sh"):
            text=path.read_text(encoding="utf-8"); self.assertIn("S2A_C0_BASELINE",text); self.assertNotIn("train_phase",text)

    def test_linux_launcher_is_lf_and_safe(self):
        path=Path(__file__).resolve().parents[1]/"scripts/linux/run_phase1c_prrac_s2a_collision_ablation.sh"; data=path.read_bytes()
        self.assertNotIn(b"\r\n",data); text=data.decode(); self.assertIn("set -euo pipefail",text); self.assertIn("MPLBACKEND=Agg",text); self.assertIn("OMP_NUM_THREADS=1",text); self.assertIn("MKL_NUM_THREADS=1",text)

    def test_evaluator_parses_s2a_schema_and_hash(self):
        path=Path(__file__).resolve().parents[1]/"configs/chapter3/bser_phase1c_prrac_s2a_collision_ablation.json"; config=evaluator._load_config(path)
        self.assertEqual(config["schema"],"bser.phase1c.prrac.evaluation_report.v2"); self.assertEqual(len(config["search_collision_recovery_config_hash"]),64)


if __name__ == "__main__": unittest.main()
