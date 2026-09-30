"""Bounded orchestration checks; synthetic collisions, never formal evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chapter3_bser.experiments.d2_suite_v1.plan import (
    ARMS, DEFAULT, ROOT, digest, prepare, read, settings, sha, validate_pair, write,
)
from chapter3_bser.experiments.d2_suite_v1.run import execute, receipt
from chapter3_bser.experiments.d2_suite_v1.summarize import metrics, paired, summarize


def inputs(directory):
    cfg = settings()
    cfg.update(training_seeds=[17], training_environment_steps=1, checkpoint_interval=1,
               evaluation_episodes=1, generated_train_count=1)
    # Reduce HGR cycle counts, not task horizon/reward/action contracts.
    hgr = read(ROOT / cfg["templates"]["D2_HGR"])
    hgr.update(main_prefix_batch_size=1, suffix_training_episodes_per_cycle=1,
               pilot_prefix_episodes_per_cycle=1, correction_draws_per_cycle=1)
    hgr["predictor"]["updates"] = 1
    write(directory / "hgr.json", hgr)
    cfg["templates"]["D2"] = cfg["templates"]["D2_HGR"] = str(directory / "hgr.json")
    write(directory / "suite.json", cfg)
    scene = read(ROOT / "tests/fixtures/hgr/handoff_manifest.json")["scenarios"][0]
    scene.update(max_steps=400, protocol="ch3_unknown_map_v1", scenario_id="suite_smoke_train", scenario_seed=117,
                 target_position=[18.,18.,6.], target_initial_position=[18.,18.,6.],
                 obstacles=[dict(center=[14.,14.,3.], size=[2.,2.,2.])])
    train = dict(scenarios=[scene])
    evaluation = copy.deepcopy(train)
    evaluation["scenarios"][0].update(scenario_id="suite_smoke_eval", scenario_seed=217,
                  scenario_split="validation", scenario_role="validation", flow_phase_x=0.1)
    write(directory / "train.json", train)
    write(directory / "eval.json", evaluation)
    return dict(config=directory / "suite.json", train_manifest=directory / "train.json",
                eval_manifest=directory / "eval.json")


class SuiteTests(unittest.TestCase):
    def test_swapped_method_templates_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            args = inputs(directory)
            value = read(args["config"])
            value["templates"]["D2_B2"] = value["templates"]["D2_B3"]
            write(args["config"], value)
            output = directory / "collision_terminal/swapped"
            with self.assertRaises(ValueError):
                prepare(output, **args)
            self.assertFalse(output.exists())

    def test_baseline_manifest_accepts_only_canonical_and_historical_protocols(self):
        from types import SimpleNamespace
        from chapter3_bser.experiments.baselines.common.train import BaselineTrainer
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            args = inputs(directory)
            config = read(ROOT / settings()["templates"]["D2_B2"])
            manifest = read(args["train_manifest"])
            for protocol in ("ch3_unknown_map_v1", "CH3_UNKNOWN_MAP_V1", "unrelated_protocol"):
                manifest["scenarios"][0]["protocol"] = protocol
                write(args["train_manifest"], manifest)
                config.update(scenario_manifest=str(args["train_manifest"]),
                              scenario_manifest_sha256=sha(args["train_manifest"]))
                instance = SimpleNamespace(config=config)
                if protocol == "unrelated_protocol":
                    with self.assertRaises(ValueError):
                        BaselineTrainer._load_scenarios(instance)
                else:
                    self.assertEqual(BaselineTrainer._load_scenarios(instance), manifest["scenarios"])

    def test_new_source_scope_and_unused_profile_are_protected(self):
        from chapter3_bser.experiments.d2_suite_v1 import provenance as gate
        from tests.test_safe_search_provenance import checked_copy
        with checked_copy() as root:
            gate.old.framework_sources(root)
            p = root / gate.MANIFEST
            original = p.read_bytes()
            value = read(p)
            value["profiles"]["git_lf"]["changes"][sorted(gate.CHANGED)[0]]["before"] = "0"*64
            value.pop("sha256")
            value["sha256"] = gate.old.digest(value)
            write(p, value)
            with self.assertRaises(ValueError):
                gate.old.framework_sources(root)
            p.write_bytes(original)
            for name in gate.ADDED | gate.CHANGED | {gate.REVIEW}:
                target = root / name
                original = target.read_bytes()
                target.write_bytes(original+b"\n")
                with self.subTest(path=name), self.assertRaises(ValueError):
                    gate.old.framework_sources(root)
                target.write_bytes(original)

    def test_explicit_scene_generation_freezes_disjoint_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            args = inputs(directory)
            root = directory / "collision_terminal/generated"
            plan = prepare(root, config=args["config"], generate_scenes=True)
            train, evaluation = read(root / "inputs/train.json"), read(root / "inputs/evaluation.json")
            validate_pair(train, evaluation, 1)
            self.assertEqual(plan["scene_origin"]["kind"], "generated")
            self.assertFalse(execute(root)["executed"])
            self.assertFalse((root / "jobs").exists())

    def test_preparation_preview_partial_summary_and_mutation_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            root = directory / "collision_terminal/run"
            args = inputs(directory)
            with patch("chapter3_bser.experiments.hgr.train.Trainer", side_effect=AssertionError("training started")):
                plan = prepare(root, **args)
                preview = execute(root)
            self.assertEqual(len(plan["jobs"]), 4)
            self.assertEqual(len(preview["jobs"]), 7)
            self.assertFalse(preview["executed"])
            report = summarize(root, directory / "report")
            self.assertFalse(report["suite_complete"])
            self.assertTrue(all(g["found_rate"] is None for g in report["groups"]))
            with self.assertRaises(FileExistsError):
                prepare(root, **args)
            p = root / "inputs/evaluation.json"
            p.write_bytes(p.read_bytes()+b" ")
            with self.assertRaisesRegex(ValueError, "prepared input changed"):
                execute(root)

    def test_scene_overlap_and_split_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args = inputs(Path(temp))
            train, original = read(args["train_manifest"]), read(args["eval_manifest"])
            for field in ("scenario_id", "scenario_seed", "scenario_role", "max_steps"):
                bad = copy.deepcopy(original)
                bad["scenarios"][0][field] = (399 if field == "max_steps" else train["scenarios"][0][field])
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_pair(train, bad, 1)
            bad = copy.deepcopy(original)
            bad["scenarios"][0]["flow_phase_x"] = train["scenarios"][0]["flow_phase_x"]
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_pair(train, bad, 1)

    def test_pairing_failures_and_seed_statistics(self):
        rows = [dict(scenario_id=str(i), scenario_seed=i, environment_innovation_seed=20+i,
                     found=i==0, found_step=10 if i==0 else None, pre_found_collision=i==1,
                     found_penalized_steps=10 if i==0 else 400, pre_found_steps=20,
                     pre_found_stagnant_steps=2, pre_found_moving_steps=18,
                     collision_episode=i==1, success=False) for i in range(2)]
        self.assertEqual(metrics(rows)["found_penalized_steps"], 205)
        self.assertEqual(metrics(rows)["found_steps_conditional"], 10)
        other = copy.deepcopy(rows)
        other[1].update(found=True, found_step=30, found_penalized_steps=30, pre_found_collision=False)
        result = paired(rows, other)
        self.assertEqual(result["right_only_found"], 1)
        self.assertEqual(result["found_rate_delta"], .5)
        self.assertEqual(result["found_penalized_steps_delta"], -185)
        with self.assertRaises(ValueError):
            paired(rows, list(reversed(other)))

    def test_real_four_arm_collision_smoke_and_verified_skip(self):
        import torch
        torch.set_num_threads(1)
        from chapter3_bser.experiments.hgr.runtime import MissionRuntime
        from chapter3_bser.experiments.baselines.common.runtime import BaselineMissionRuntime
        native, baseline = MissionRuntime.advance, BaselineMissionRuntime.advance
        def guarded(method):
            def call(runtime, *args, **kwargs):
                if runtime.step >= 3:
                    raise AssertionError("synthetic collision smoke exceeded three steps")
                physics = runtime.env.unwrapped
                physics.obstacles = [dict(center=physics._agent_pos[0].tolist(), size=[.001]*3)]
                physics._build_obstacle_tensors()
                return method(runtime, *args, **kwargs)
            return call
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            root = directory / "collision_terminal/run"
            plan = prepare(root, **inputs(directory))
            with patch.object(MissionRuntime, "advance", guarded(native)), patch.object(BaselineMissionRuntime, "advance", guarded(baseline)):
                result = execute(root, execute=True)
            self.assertTrue(all(j["status"] == "complete" for j in result["jobs"]))
            report = summarize(root, directory / "report")
            self.assertTrue(report["suite_complete"])
            self.assertEqual(len(read(directory / "report/summary.json")["paired_comparisons"]), 6)
            self.assertTrue(all(g["pre_found_collision_rate"] == 1 for g in report["groups"]))
            with patch("chapter3_bser.experiments.d2_suite_v1.evaluate.evaluate", side_effect=AssertionError("reran")):
                repeated = execute(root, execute=True)
            self.assertTrue(all(j["status"] == "verified_complete" for j in repeated["jobs"]))
            job = plan["jobs"][0]
            path = root / job["evaluation"] / "episodes.json"
            path.write_bytes(path.read_bytes()+b" ")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                receipt(root, plan, job, "evaluate")


if __name__ == "__main__":
    unittest.main()
