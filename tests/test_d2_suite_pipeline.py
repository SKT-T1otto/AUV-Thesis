"""Stage-order, failure barriers, no-checkpoint D2 and bounded native smoke."""
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chapter3_bser.experiments.d2_suite_v1 import pipeline as p
from chapter3_bser.experiments.d2_suite_v1.plan import prepare, read, sha, write
from tests.test_d2_suite import inputs
from tests.test_d2_suite_linux_evaluation import prepared


class PipelineTests(unittest.TestCase):
    def test_default_nine_training_ten_evaluation_preview_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            args = inputs(directory)
            config = read(args["config"])
            config["training_seeds"] = [2729, 3729, 4729]
            write(args["config"], config)
            root = directory / "collision_terminal/run"
            prepare(root, **args)
            before = {x.relative_to(root): sha(x) for x in root.rglob("*") if x.is_file()}
            result = p.run(root)
            self.assertFalse(result["executed"])
            self.assertFalse(result["d2_checkpoint_required"])
            self.assertEqual(len(result["phases"]["train"]["jobs"]), 9)
            self.assertEqual(len(result["phases"]["evaluate"]["jobs"]), 10)
            self.assertEqual(before, {x.relative_to(root): sha(x) for x in root.rglob("*") if x.is_file()})

    def test_entire_phase_order_lock_and_new_reports_on_repeat(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = []
            def execute(root, *, stage, execute):
                if execute:
                    self.assertTrue((root / ".d2_linux.lock").exists())
                    events.append(stage)
                return dict(executed=execute, jobs=[])
            def verify(root, stage):
                events.append("verify_" + stage)
            def summarize(root, report):
                self.assertTrue((root / ".d2_linux.lock").exists())
                events.append("summary")
                write(report / "summary.json", dict(suite_complete=True))
                return dict(suite_complete=True)
            with patch.object(p, "check", return_value={}), patch.object(p, "load_plan", return_value={}), \
                 patch.object(p, "execute", side_effect=execute), patch.object(p, "require_complete", side_effect=verify), \
                 patch.object(p, "summarize", side_effect=summarize), redirect_stdout(io.StringIO()):
                first = p.run(root, confirmed=True)
                second = p.run(root, confirmed=True)
            self.assertEqual(events, ["train", "verify_train", "evaluate", "verify_evaluate", "summary"] * 2)
            self.assertNotEqual(first["report_directory"], second["report_directory"])
            self.assertTrue(Path(first["report_directory"], "summary.json").exists())
            self.assertFalse((root / ".d2_linux.lock").exists())

    def test_training_or_evaluation_error_stops_later_phases_and_records_failure(self):
        for failing in ("train", "evaluate"):
            with self.subTest(phase=failing), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                events = []
                def execute(root, *, stage, execute):
                    if execute:
                        events.append(stage)
                        if stage == failing:
                            raise RuntimeError("synthetic " + stage + " failure")
                    return dict(executed=execute, jobs=[])
                with patch.object(p, "check", return_value={}), patch.object(p, "execute", side_effect=execute), \
                     patch.object(p, "require_complete"), patch.object(p, "summarize") as summary, \
                     redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "synthetic"):
                        p.run(root, confirmed=True)
                self.assertEqual(events, ["train"] if failing == "train" else ["train", "evaluate"])
                summary.assert_not_called()
                directory = next((root / "launch_logs").iterdir())
                status = read(directory / "pipeline.json")
                self.assertEqual((status["status"], status["stage"]), ("failed", failing))
                self.assertIn("Traceback", (directory / "console.log").read_text())
                self.assertFalse((root / ".d2_linux.lock").exists())

    def test_missing_training_receipts_block_even_d2_before_evaluation(self):
        with prepared() as (root, _):
            stages = []
            def execute(root, *, stage, execute):
                if execute:
                    stages.append(stage)
                return dict(executed=execute, jobs=[])
            with patch.object(p, "execute", side_effect=execute), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "train phase is incomplete"):
                    p.run(root, confirmed=True)
            self.assertEqual(stages, ["train"])
            self.assertFalse((root / "jobs/D2/evaluation").exists())

    def test_partial_summary_cannot_mark_pipeline_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(p, "check", return_value={}), patch.object(p, "execute", return_value={}), \
                 patch.object(p, "require_complete"), patch.object(p, "summarize", return_value=dict(suite_complete=False)), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "partial summary"):
                    p.run(root, confirmed=True)
            status = read(next((root / "launch_logs").iterdir()) / "pipeline.json")
            self.assertEqual((status["status"], status["stage"]), ("failed", "summarize"))

    def test_cli_requires_explicit_execution(self):
        with patch.object(p, "run", return_value={}) as run, redirect_stdout(io.StringIO()):
            p.main(["--root", "example"])
        run.assert_called_once_with(root="example", confirmed=False)

    def test_native_smoke_all_training_precedes_d2_and_learning_evaluation(self):
        import torch
        from chapter3_bser.experiments.hgr.runtime import MissionRuntime
        from chapter3_bser.experiments.baselines.common.runtime import BaselineMissionRuntime
        from chapter3_bser.experiments.d2_suite_v1 import evaluate as evaluation
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        evaluated = []
        native_evaluate = evaluation.evaluate
        def evaluate(root, plan, job, checkpoint):
            p.require_complete(root, "train")
            if job["arm"] == "D2":
                self.assertIsNone(checkpoint)
            else:
                self.assertTrue(Path(checkpoint).is_file())
            evaluated.append(job["arm"])
            return native_evaluate(root, plan, job, checkpoint)
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
            with prepared() as (root, _), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with patch.object(MissionRuntime, "advance", guarded(MissionRuntime.advance)), \
                     patch.object(BaselineMissionRuntime, "advance", guarded(BaselineMissionRuntime.advance)), \
                     patch.object(evaluation, "evaluate", side_effect=evaluate):
                    result = p.run(root, confirmed=True)
                self.assertEqual(evaluated, ["D2", "D2_B2", "D2_B3", "D2_HGR"])
                self.assertTrue(result["suite_complete"])
                self.assertTrue(read(Path(result["report_directory"]) / "summary.json")["suite_complete"])
                self.assertEqual(read(Path(result["launch_directory"]) / "pipeline.json")["status"], "complete")
                with patch.object(evaluation, "evaluate", side_effect=AssertionError("reran")):
                    repeat = p.run(root, confirmed=True)
                for stage in ("train", "evaluate"):
                    self.assertTrue(all(j["status"] == "verified_complete" for j in repeat["phases"][stage]["jobs"]))
        finally:
            torch.set_num_threads(threads)

    def test_new_source_scope_and_unused_profile_are_protected(self):
        from chapter3_bser.experiments.d2_suite_v1 import pipeline_provenance as gate
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
