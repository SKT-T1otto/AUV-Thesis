"""Frozen-population and fail-closed development orchestration checks; no simulator."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chapter3_bser.experiments.safe_search_v1 import run_development as runner


def fixtures():
    scenes = [dict(scenario_id=f"scene_{i}", scenario_seed=runner.SEED + i,
                   scenario_split="validation", scenario_profile="M20_MOVING_UNKNOWN_MULTI",
                   max_steps=400) for i in range(100)]
    records = [dict(source_episode_index=i, scenario_id=scenes[i]["scenario_id"],
                    scenario_seed=runner.SEED + i, environment_innovation_seed=runner.SEED + i,
                    scene_content_sha256=runner.digest(scenes[i])) for i in runner.INDICES]
    return (dict(scenarios=scenes), dict(profile="M20_MOVING_UNKNOWN_MULTI", max_steps=400),
            dict(schema="ch3.safe_search.experiment_plan.v1", splits=dict(development=records)))


def parent_identity():
    manifest, config, plan = fixtures()
    return dict(schema=runner.IDENTITY_SCHEMA, selected=runner.validate_development_inputs(manifest, config, plan),
                variants=list(runner.VARIANTS), planned_episode_runs=100, max_physical_steps=40000,
                seed=runner.SEED, task_horizon=400, baseline=runner.BASELINE, full_episodes=True,
                formal_thesis_evaluation=False, training=False, checkpoint_loaded=False,
                sources_before=dict(inventory=dict(sha256="source")),
                sources_after=dict(inventory=dict(sha256="source")),
                child_input_sha256={"config": "config_hash", "manifest": "manifest_hash"},
                search_coverage_enabled=True, experiment_complete=True,
                source_and_input_verification_passed=True)


def write_child(output, expected, parent):
    output.mkdir(parents=True)
    identity = dict(schema="ch3.safe_search.paired_identity.v1", baseline=runner.BASELINE,
                    variants=list(runner.VARIANTS), seed=runner.SEED, steps=400, task_horizon=400,
                    full_episodes=True, training=False, checkpoint_loaded=False,
                    formal_thesis_evaluation=False, search_coverage_enabled=True, selected=[expected],
                    input_sha256=parent["child_input_sha256"], sources_before=parent["sources_before"],
                    sources_after=parent["sources_before"], source_and_input_verification_passed=True)
    rows = []
    for variant in runner.VARIANTS:
        row = dict(variant=variant, baseline=runner.BASELINE, terminal=True,
                   full_episode_completed=True, full_episode_requested=True,
                   physical_steps=400, stop_reason="timeout", complete_episode_row={},
                   found_within_budget=False, pre_found_collision=False,
                   pre_found_exposure_steps=400, searcher_motion_stall_proxy_agent_steps=0,
                   searcher_hold_agent_steps=0, wall_seconds=0.1,
                   **{k: expected[k] for k in ("original_episode_index", "scenario_id", "scenario_seed",
                                              "environment_innovation_seed")})
        arm = output / f"episode_{expected['original_episode_index']:04d}" / variant
        arm.mkdir(parents=True)
        runner.write_json(arm / "summary.json", row)
        rows.append(row)
    summary = runner.summarize(rows, list(runner.VARIANTS), 1, True)
    summary["source_and_input_verification_passed"] = True
    for name, value in (("identity", identity), ("episodes", rows), ("summary", summary)):
        runner.write_json(output / f"{name}.json", value)


class FrozenPopulationTests(unittest.TestCase):
    def test_fixed_indices_retain_manifest_identity_and_innovation_seed(self):
        manifest, config, plan = fixtures()
        selected = runner.validate_development_inputs(manifest, config, plan)
        self.assertEqual(len(selected), 20)
        self.assertEqual(selected[-1]["original_episode_index"], 97)
        self.assertEqual(selected[-1]["environment_innovation_seed"], 12826)
        self.assertEqual(selected[5]["scenario_id"], "scene_28")

    def test_subset_reindexed_reordered_and_changed_content_rejected(self):
        for mutation in ("subset", "reordered", "changed_seed", "changed_content", "plan_indices"):
            manifest, config, plan = fixtures()
            if mutation == "subset":
                manifest["scenarios"] = manifest["scenarios"][:20]
            elif mutation == "reordered":
                manifest["scenarios"][0], manifest["scenarios"][1] = manifest["scenarios"][1], manifest["scenarios"][0]
            elif mutation == "changed_seed":
                manifest["scenarios"][28]["scenario_seed"] += 1
            elif mutation == "changed_content":
                manifest["scenarios"][28]["obstacles"] = []
            else:
                plan["splits"]["development"][0]["source_episode_index"] = 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                runner.validate_development_inputs(manifest, config, plan)

    def test_command_preserves_index_full_horizon_and_instrumentation(self):
        command = runner.child_command(Path("m.json"), Path("c.json"), Path("out"), 97)
        self.assertEqual(command[command.index("--episode-indices") + 1], "97")
        self.assertEqual(command[command.index("--seed") + 1], "12729")
        self.assertIn("--full-episodes", command)
        self.assertIn("--search-coverage", command)
        self.assertNotIn("--verify-v0", command)

    def test_cli_requires_deliberate_execution(self):
        with self.assertRaises(SystemExit):
            runner.parser().parse_args(["--manifest", "m", "--output-dir", "o"])
        args = runner.parser().parse_args(["--manifest", "m", "--output-dir", "o", "--execute"])
        self.assertEqual(args.workers, 4)


class CompletenessTests(unittest.TestCase):
    def test_full_population_accepted_and_missing_scene_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, parent = Path(temporary), parent_identity()
            runner.write_json(root / "identity.json", parent)
            for item in parent["selected"]:
                write_child(root / f"scene_{item['original_episode_index']:04d}", item, parent)
            identity, rows = runner.collect_completed(root, plan=fixtures()[2])
            self.assertEqual(len(rows), 100)
            self.assertEqual(identity, parent)
            (root / "scene_0097" / "episodes.json").unlink()
            with self.assertRaises(FileNotFoundError):
                runner.collect_completed(root, plan=fixtures()[2])

    def test_root_program_failure_or_unverified_completion_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, parent = Path(temporary), parent_identity()
            parent["experiment_complete"] = False
            runner.write_json(root / "identity.json", parent)
            with self.assertRaisesRegex(ValueError, "not completed"):
                runner.collect_completed(root)

    def test_root_scenario_hash_cannot_be_replaced_even_with_matching_children(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, parent = Path(temporary), parent_identity()
            parent["selected"][0]["scenario_sha256"] = "changed_scene"
            runner.write_json(root / "identity.json", parent)
            with self.assertRaisesRegex(ValueError, "frozen development plan"):
                runner.collect_completed(root, plan=fixtures()[2])
            runner.write_json(root / "failure.json", dict(message="program error"))
            with self.assertRaisesRegex(ValueError, "program failure"):
                runner.collect_completed(root)

    def test_mixed_source_prefix_missing_duplicate_or_tampered_arm_rejected(self):
        for mutation in ("source", "input", "prefix", "duplicate", "missing", "arm_artifact", "failure"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root, parent = Path(temporary), parent_identity()
                expected = parent["selected"][0]
                child = root / "scene_0000"
                write_child(child, expected, parent)
                if mutation in ("source", "input"):
                    identity = runner.read_json(child / "identity.json")
                    identity["sources_after" if mutation == "source" else "input_sha256"] = {"changed": True}
                    runner.write_json(child / "identity.json", identity)
                elif mutation == "failure":
                    runner.write_json(child / "failure.json", dict(message="program error"))
                elif mutation == "arm_artifact":
                    runner.write_json(child / "episode_0000/V4/summary.json", {})
                else:
                    rows = runner.read_json(child / "episodes.json")
                    if mutation == "prefix":
                        rows[0]["terminal"] = False
                        rows[0]["stop_reason"] = "budget_cutoff"
                    elif mutation == "duplicate":
                        rows[-1] = copy.deepcopy(rows[0])
                    else:
                        rows.pop()
                    runner.write_json(child / "episodes.json", rows)
                with self.assertRaises(ValueError):
                    runner.validate_child(child, expected, parent)


class OrchestrationTests(unittest.TestCase):
    def run_fake(self, temporary, fail=False):
        root = Path(temporary)
        manifest, config, plan = fixtures()
        for name, value in (("manifest", manifest), ("config", config), ("plan", plan)):
            runner.write_json(root / f"{name}.json", value)
        calls = []
        def fake_child(command, log):
            index = int(command[command.index("--episode-indices") + 1])
            calls.append(index)
            Path(log).write_text("fake independent process", encoding="utf-8")
            if fail:
                return dict(returncode=7)
            output = Path(command[command.index("--output-dir") + 1])
            parent = runner.read_json(output.parent / "identity.json")
            expected = next(r for r in parent["selected"] if r["original_episode_index"] == index)
            write_child(output, expected, parent)
            return dict(returncode=0)
        with (patch.object(runner, "PLAN", root / "plan.json"),
              patch.object(runner, "_run_child", fake_child),
              patch("chapter3_bser.experiments.safe_search_v1.provenance.framework_sources", return_value={"source": "fixed"}),
              patch("chapter3_bser.experiments.safe_search_v1.provenance.verify_sources")):
            if fail:
                with self.assertRaisesRegex(RuntimeError, "no complete-stage verdict"):
                    runner.run_development(root / "manifest.json", root / "out",
                                           config_path=root / "config.json", workers=1)
            else:
                summary = runner.run_development(root / "manifest.json", root / "out",
                                                config_path=root / "config.json", workers=4)
                self.assertTrue(summary["experiment_complete"])
                self.assertEqual(len(runner.collect_completed(root / "out")[1]), 100)
        return calls

    def test_independent_children_yield_one_complete_100_arm_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls = self.run_fake(temporary)
            self.assertEqual(sorted(calls), list(runner.INDICES))

    def test_failed_child_stops_new_scheduling_and_retains_planned_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls = self.run_fake(temporary, fail=True)
            self.assertEqual(calls, [0])
            failure = runner.read_json(Path(temporary) / "out/failure.json")
            self.assertEqual(failure["planned_episode_runs"], 100)
            self.assertEqual(len(failure["unstarted_original_indices"]), 19)
            self.assertFalse((Path(temporary) / "out/summary.json").exists())


if __name__ == "__main__":
    unittest.main()
