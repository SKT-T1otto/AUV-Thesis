"""Readiness checks and bounded synthetic full workflow, never formal evidence."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from chapter3_bser.experiments.d2_suite_v1 import evaluation_cli as cli
from chapter3_bser.experiments.d2_suite_v1.plan import ROOT, prepare, read, sha, write
from chapter3_bser.experiments.d2_suite_v1.run import execute, save_receipt
from tests.test_d2_suite import inputs


@contextmanager
def prepared():
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        root = directory / "collision_terminal/run"
        plan = prepare(root, **inputs(directory))
        yield root, plan


class EvaluationEntryTests(unittest.TestCase):
    def test_previews_select_reference_or_learning_jobs_without_writing(self):
        with prepared() as (root, plan):
            before = {p.relative_to(root): sha(p) for p in root.rglob("*") if p.is_file()}
            d2 = cli.evaluate(root, scope="d2")
            trained = cli.evaluate(root, scope="trained")
            self.assertEqual([j["job"] for j in d2["jobs"]], ["D2"])
            self.assertTrue(d2["ready"])
            self.assertFalse(d2["executed"])
            self.assertEqual(trained["missing_training_jobs"], [j["name"] for j in plan["jobs"][1:]])
            self.assertFalse(trained["ready"])
            subset = cli.evaluate(root, scope="trained", arms=["D2_B3"], seeds=[17])
            self.assertEqual(subset["missing_training_jobs"], ["D2_B3_seed17"])
            self.assertEqual(before, {p.relative_to(root): sha(p) for p in root.rglob("*") if p.is_file()})

    def test_missing_training_fails_before_any_evaluation_or_launch_log(self):
        with prepared() as (root, plan), \
             patch("chapter3_bser.experiments.d2_suite_v1.evaluate.evaluate", side_effect=AssertionError("started")):
            with self.assertRaisesRegex(ValueError, "D2_B2_seed17.*D2_B3_seed17.*D2_HGR_seed17"):
                cli.evaluate(root, scope="trained", confirmed=True)
            self.assertFalse((root / "jobs").exists())
            self.assertFalse((root / "launch_logs").exists())

    def test_training_receipt_verifies_checkpoint_bytes_before_evaluation(self):
        with prepared() as (root, plan):
            job = plan["jobs"][1]
            folder = root / job["train"]
            for name in ("summary.json", "episodes.json", "config.json"):
                write(folder / name, {})
            checkpoint = folder / "synthetic.pt"
            checkpoint.write_bytes(b"receipt plumbing only; never loaded")
            save_receipt(root, plan, job, "train", checkpoint)
            self.assertTrue(cli.evaluate(root, scope="trained", arms=["D2_B2"])["ready"])
            checkpoint.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                cli.evaluate(root, scope="trained", arms=["D2_B2"], confirmed=True)
            self.assertFalse((root / job["evaluation"]).exists())

    def test_scope_and_seed_validation(self):
        for kwargs in (dict(scope="d2", seeds=[17]), dict(scope="d2", arms=["D2_B2"]),
                       dict(scope="trained", arms=["D2"]), dict(scope="other")):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                cli.evaluate("unused", **kwargs)
        with prepared() as (root, _), self.assertRaisesRegex(ValueError, "unknown arm/seed"):
            cli.evaluate(root, scope="trained", seeds=[999])
        result = subprocess.run([sys.executable, "-B", "-m", cli.__name__, "d2", "--root", "unused",
                                 "--arms", "D2_B2"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)

    def test_cli_dispatch_and_default_preview(self):
        with patch.object(cli, "evaluate", return_value={}) as call, redirect_stdout(io.StringIO()):
            cli.main(["d2", "--root", "example"])
        call.assert_called_once_with(scope="d2", root="example", confirmed=False)
        with patch.object(cli, "evaluate", return_value={}) as call, redirect_stdout(io.StringIO()):
            cli.main(["trained", "--root", "example", "--arms", "D2_HGR", "--seeds", "17", "--execute"])
        call.assert_called_once_with(scope="trained", root="example", confirmed=True, arms=["D2_HGR"], seeds=[17])

    def test_native_bounded_reference_train_evaluate_summary_and_skip(self):
        import torch
        from chapter3_bser.experiments.hgr.runtime import MissionRuntime
        from chapter3_bser.experiments.baselines.common.runtime import BaselineMissionRuntime
        from chapter3_bser.experiments.d2_suite_v1.summarize import summarize
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        def guarded(method):
            def call(runtime, *args, **kwargs):
                if runtime.step >= 3:
                    raise AssertionError("synthetic collision exceeded three steps")
                physics = runtime.env.unwrapped
                physics.obstacles = [dict(center=physics._agent_pos[0].tolist(), size=[.001]*3)]
                physics._build_obstacle_tensors()
                return method(runtime, *args, **kwargs)
            return call
        try:
            with prepared() as (root, plan), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with patch.object(MissionRuntime, "advance", guarded(MissionRuntime.advance)), \
                     patch.object(BaselineMissionRuntime, "advance", guarded(BaselineMissionRuntime.advance)):
                    reference = cli.evaluate(root, scope="d2", confirmed=True)
                    self.assertEqual(len(reference["jobs"]), 1)
                    self.assertFalse(any((root / j["train"]).exists() for j in plan["jobs"]))
                    execute(root, stage="train", execute=True)
                    evaluated = cli.evaluate(root, scope="trained", confirmed=True)
                self.assertEqual(len(evaluated["jobs"]), 3)
                self.assertTrue(all(j["status"] == "complete" for j in evaluated["jobs"]))
                report = summarize(root, root / "report")
                self.assertTrue(report["suite_complete"])
                self.assertEqual(len(read(root / "report/summary.json")["paired_comparisons"]), 6)
                with patch("chapter3_bser.experiments.d2_suite_v1.evaluate.evaluate", side_effect=AssertionError("reran")):
                    repeated = cli.evaluate(root, scope="trained", confirmed=True)
                self.assertTrue(all(j["status"] == "verified_complete" for j in repeated["jobs"]))
        finally:
            torch.set_num_threads(threads)

    def test_new_source_scope_and_unused_profile_reject_corruption(self):
        from chapter3_bser.experiments.d2_suite_v1 import evaluation_provenance as gate
        from tests.test_safe_search_provenance import checked_copy
        with checked_copy() as root:
            gate.old.framework_sources(root)
            for name in gate.ADDED | gate.CHANGED | {gate.REVIEW}:
                path = root / name
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"\n")
                    with self.subTest(path=name), self.assertRaises(ValueError):
                        gate.old.framework_sources(root)
                finally:
                    path.write_bytes(original)
            value = read(root / gate.MANIFEST)
            value["profiles"]["git_lf"]["changes"][sorted(gate.CHANGED)[0]]["before"] = "0" * 64
            value.pop("sha256")
            value["sha256"] = gate.old.digest(value)
            write(root / gate.MANIFEST, value)
            with self.assertRaises(ValueError):
                gate.old.framework_sources(root)


if __name__ == "__main__":
    unittest.main()
