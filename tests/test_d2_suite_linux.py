"""Linux launch plumbing only: no formal training, evaluation or checkpoint load."""
from contextlib import redirect_stdout, redirect_stderr
import io
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from chapter3_bser.experiments.d2_suite_v1 import linux
from chapter3_bser.experiments.d2_suite_v1.plan import ROOT, prepare, read, sha, write
from tests.test_d2_suite import inputs


class LinuxLaunchTests(unittest.TestCase):
    def test_native_preflight_and_previews_preserve_plan_and_start_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            root = directory / "collision_terminal/run"
            prepare(root, **inputs(directory))
            before = {p.relative_to(root): sha(p) for p in root.rglob("*") if p.is_file()}
            with patch("chapter3_bser.experiments.hgr.train.Trainer.run", side_effect=AssertionError("training")), \
                 patch("chapter3_bser.experiments.baselines.common.train.BaselineTrainer.run", side_effect=AssertionError("training")):
                train = linux.launch(root, stage="train")
                evaluate = linux.launch(root, stage="evaluate")
            self.assertFalse(train["executed"])
            self.assertEqual(len(train["jobs"]), 3)
            self.assertEqual(len(evaluate["jobs"]), 4)
            self.assertEqual(train["preflight"]["runtime"]["training_device"], "cpu")
            self.assertEqual(before, {p.relative_to(root): sha(p) for p in root.rglob("*") if p.is_file()})

    def test_relocated_plan_cannot_execute_but_can_be_summarized(self):
        from chapter3_bser.experiments.d2_suite_v1.summarize import summarize
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            root = directory / "collision_terminal/run"
            prepare(root, **inputs(directory))
            moved = directory / "copied"
            shutil.copytree(root, moved)
            with self.assertRaisesRegex(ValueError, "another location"):
                linux.check(moved)
            # Offline summary of a copied partial run stays valid and has no rates.
            summarize(moved, directory / "report")
            self.assertTrue((directory / "report/report.md").exists())

    def test_bad_stage_d2_training_and_unknown_seed_cannot_launch(self):
        with self.assertRaises(ValueError):
            linux.launch("unused", stage="all")
        with self.assertRaisesRegex(ValueError, "no trainable"):
            linux.launch("unused", stage="train", arms=["D2"], confirmed=True)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            root = directory / "collision_terminal/run"
            prepare(root, **inputs(directory))
            with self.assertRaisesRegex(ValueError, "unknown arm/seed"):
                linux.launch(root, stage="train", seeds=[999], confirmed=True)
            self.assertFalse((root / "launch_logs").exists())

    def test_lock_conflict_release_on_failure_and_foreign_ownership(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / ".d2_linux.lock"
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                with linux.run_lock(root):
                    owner = path.read_bytes()
                    with self.assertRaises(FileExistsError):
                        with linux.run_lock(root):
                            self.fail("second writer acquired lock")
                    self.assertEqual(path.read_bytes(), owner)
                    raise RuntimeError("synthetic")
            self.assertFalse(path.exists())
            with linux.run_lock(root):
                path.write_text("foreign-owner", encoding="utf-8")
            self.assertEqual(path.read_text(), "foreign-owner")

    def test_explicit_launch_logs_both_streams_and_uses_only_selected_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def fake_execute(root, **options):
                if options["execute"]:
                    print("training stdout")
                    print("training stderr", file=sys.stderr)
                return dict(executed=options["execute"], jobs=[])
            with patch.object(linux, "check", return_value={}), \
                 patch.object(linux, "execute", side_effect=fake_execute) as run, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = linux.launch(root, stage="train", arms=["D2_B2"], seeds=[17], confirmed=True)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args.kwargs, dict(stage="train", arms=["D2_B2"], seeds=[17], execute=True))
            folder = Path(result["launch_directory"])
            self.assertEqual(read(folder / "launch.json")["status"], "complete")
            self.assertIn("training stdout", (folder / "console.log").read_text())
            self.assertIn("training stderr", (folder / "console.log").read_text())
            self.assertFalse((root / ".d2_linux.lock").exists())

    def test_failed_launch_retains_traceback_and_propagates_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(linux, "check", return_value={}), \
                 patch.object(linux, "execute", side_effect=[{}, RuntimeError("synthetic failure")]), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    linux.launch(root, stage="evaluate", confirmed=True)
            folders = list((root / "launch_logs").iterdir())
            self.assertEqual(len(folders), 1)
            status = read(folders[0] / "launch.json")
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["error_type"], "RuntimeError")
            self.assertIn("Traceback", (folders[0] / "console.log").read_text())
            self.assertFalse((root / ".d2_linux.lock").exists())

    def test_cli_evaluation_and_check_dispatch(self):
        with patch.object(linux, "launch", return_value={}) as run, redirect_stdout(io.StringIO()):
            linux.main(["evaluate", "--root", "sample", "--arms", "D2", "--execute"])
        run.assert_called_once_with(stage="evaluate", root="sample", arms=["D2"], seeds=None, confirmed=True)
        with patch.object(linux, "check", return_value={}) as check, redirect_stdout(io.StringIO()):
            linux.main(["check", "--root", "sample"])
        check.assert_called_once_with(root="sample")

    def test_cli_rejects_training_d2_with_nonzero_exit(self):
        result = subprocess.run([sys.executable, "-B", "-m", linux.__name__, "train", "--root", "unused",
                                 "--arms", "D2", "--execute"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)

    def test_new_provenance_scope_review_and_unused_profile_are_protected(self):
        from chapter3_bser.experiments.d2_suite_v1 import linux_provenance as gate
        from tests.test_safe_search_provenance import checked_copy
        with checked_copy() as root:
            gate.old.framework_sources(root)
            for name in gate.ADDED | gate.CHANGED | {gate.REVIEW}:
                path = root / name
                before = path.read_bytes()
                try:
                    path.write_bytes(before + b"\n")
                    with self.subTest(path=name), self.assertRaises(ValueError):
                        gate.old.framework_sources(root)
                finally:
                    path.write_bytes(before)
            path = root / gate.MANIFEST
            value = read(path)
            value["profiles"]["git_lf"]["changes"][sorted(gate.CHANGED)[0]]["before"] = "0" * 64
            value.pop("sha256")
            value["sha256"] = gate.old.digest(value)
            write(path, value)
            with self.assertRaises(ValueError):
                gate.old.framework_sources(root)


if __name__ == "__main__":
    unittest.main()
