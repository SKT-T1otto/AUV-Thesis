import json
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]
FROZEN = (
    "chapter3_bser/experiments/phase1c_bser_rmaddpg/train_phase1c.py",
    "configs/chapter3/bser_phase1c_train.json",
    "scripts/run_phase1c_train.ps1",
    "scripts/run_phase1c_train.bat",
    "core/env/uav_env.py",
    "core/env/mission_env.py",
    "core/replay/ch3_buffer.py",
)

class Phase1CV2OverlayTests(unittest.TestCase):
    def test_overlay_manifest_contains_no_frozen_or_core_files(self) -> None:
        manifest = json.loads(
            (ROOT / "docs2/phase1c_v2_design/overlay_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        paths = set(manifest["files"])
        self.assertTrue(paths.isdisjoint(FROZEN))
        self.assertFalse(any(path.startswith("core/") for path in paths))
        self.assertNotIn("core/registry/experiment_registry.py", paths)

