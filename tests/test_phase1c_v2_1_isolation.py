from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from chapter3_bser.experiments.phase1c_bser_rmaddpg_v2 import (
    CHECKPOINT_SCHEMA,
    IMPLEMENTATION_VERSION,
)
from chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.train_phase1c_v2 import (
    DYNAMIC_EXECUTION_RUNTIME_REVISION,
    LEGACY_EXECUTION_RUNTIME_REVISION,
    _load_checkpoint,
    _load_config,
)


ROOT = Path(__file__).resolve().parents[1]
OLD_CONFIG = ROOT / "configs/chapter3/bser_phase1c_v2_train.json"
NEW_CONFIG = ROOT / "configs/chapter3/bser_phase1c_v2_1_train.json"


class Phase1CV21IsolationTests(unittest.TestCase):
    def test_v2_and_v2_1_share_training_contract_but_not_runtime_namespace(self) -> None:
        old_raw = json.loads(OLD_CONFIG.read_text(encoding="utf-8"))
        new_raw = json.loads(NEW_CONFIG.read_text(encoding="utf-8"))
        for key in (
            "implementation_version",
            "observation_dim",
            "action_dim",
            "critic_dim",
            "rl",
            "reward",
            "replay",
            "profile",
            "base_candidate",
        ):
            self.assertEqual(new_raw[key], old_raw[key], key)
        self.assertEqual(
            new_raw["execution_runtime_revision"],
            DYNAMIC_EXECUTION_RUNTIME_REVISION,
        )
        self.assertIn("phase1c_bser_rmaddpg_v2_1", new_raw["output_dir"])

        old = _load_config(OLD_CONFIG)
        new = _load_config(NEW_CONFIG)
        self.assertEqual(
            old["execution_runtime_revision"], LEGACY_EXECUTION_RUNTIME_REVISION
        )
        self.assertFalse(old["execution_runtime"]["dynamic_public_target_enabled"])
        self.assertFalse(old["execution_runtime"]["defer_stale_endpoint_invalid"])
        self.assertFalse(old["execution_runtime"]["refresh_on_executor_handoff"])
        self.assertTrue(new["execution_runtime"]["dynamic_public_target_enabled"])
        self.assertTrue(new["execution_runtime"]["defer_stale_endpoint_invalid"])

    def test_legacy_or_revisionless_checkpoint_is_rejected_by_v2_1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(
                {
                    "schema": CHECKPOINT_SCHEMA,
                    "metadata": {"implementation_version": IMPLEMENTATION_VERSION},
                },
                path,
            )
            with self.assertRaisesRegex(
                ValueError,
                "legacy checkpoints cannot be resumed under dynamic-public-intercept",
            ):
                _load_checkpoint(path, None, None, _load_config(NEW_CONFIG))

    def test_launchers_are_thin_and_select_new_config(self) -> None:
        paths = (
            ROOT / "scripts/run_phase1c_v2_1_train.ps1",
            ROOT / "scripts/run_phase1c_v2_1_train.bat",
            ROOT / "scripts/linux/run_phase1c_v2_1_train.sh",
        )
        self.assertTrue(all(path.is_file() for path in paths))
        powershell = paths[0].read_text(encoding="utf-8")
        batch = paths[1].read_text(encoding="utf-8")
        shell = paths[2].read_text(encoding="utf-8")
        self.assertIn("run_phase1c_v2_train.ps1", powershell)
        self.assertIn("bser_phase1c_v2_1_train.json", powershell)
        self.assertIn("run_phase1c_v2_1_train.ps1", batch)
        self.assertIn("train_phase1c_v2", shell)
        self.assertIn("bser_phase1c_v2_1_train.json", shell)
        self.assertNotIn("def run_training", powershell + batch + shell)


if __name__ == "__main__":
    unittest.main()
