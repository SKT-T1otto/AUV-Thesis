from __future__ import annotations

import json
from pathlib import Path
import subprocess
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


class Phase1CV2IsolationTests(unittest.TestCase):
    def test_v2_uses_independent_schema_config_scripts_and_output(self) -> None:
        config = json.loads(
            (ROOT / "configs/chapter3/bser_phase1c_v2_train.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["method"], "ch3_bser_rmaddpg_phase1c")
        self.assertEqual(config["implementation_version"], "bser.phase1c.execution_v2")
        self.assertTrue(config["output_dir"].startswith(
            "outputs/chapter3/phase1c_bser_rmaddpg_v2/"
        ))
        self.assertEqual(config["observation_dim"], 28)
        self.assertEqual(config["action_dim"], 3)
        self.assertEqual(config["critic_dim"], 124)
        self.assertEqual(config["reward"]["schema"], "bser.phase1c.execution_reward.v2")
        self.assertEqual(config["replay"]["schema"], "bser.phase1c.phase_aware_replay.v1")
        self.assertTrue((ROOT / "scripts/run_phase1c_v2_train.ps1").is_file())
        self.assertTrue((ROOT / "scripts/run_phase1c_v2_train.bat").is_file())

    def test_v2_source_does_not_import_v1_trainer_or_mutate_python_path(self) -> None:
        package = ROOT / "chapter3_bser/experiments/phase1c_bser_rmaddpg_v2"
        combined = "\n".join(
            path.read_text(encoding="utf-8") for path in package.glob("*.py")
        )
        self.assertNotIn("phase1c_bser_rmaddpg.train_phase1c", combined)
        self.assertNotIn("sys.path", combined)
        self.assertIn("bser.phase1c.training_state.v1", combined)

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

    def test_frozen_paths_have_no_worktree_diff_when_git_metadata_is_available(self) -> None:
        if not (ROOT / ".git").exists():
            self.skipTest("archive checkout has no .git metadata")
        result = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", *FROZEN],
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            "one or more frozen v1/core paths differ from HEAD; inspect git diff before running v2",
        )


if __name__ == "__main__":
    unittest.main()
