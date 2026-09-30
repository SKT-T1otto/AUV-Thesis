"""V5 fixed-population orchestration/accounting tests without a simulator."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chapter3_bser.experiments.safe_search_v1 import run_ac_development as runner
from chapter3_bser.experiments.safe_search_v1.run_paired import file_hash, read_json, summarize, write_json
from tests.test_safe_search_development import fixtures


def inputs():
    manifest, config, plan = fixtures()
    selected = runner.validate_development_inputs(manifest, config, plan)
    old_source = dict(inventory=dict(sha256="old_source", files={}), evolution_sha256="old_evolution", checkout_profile="old")
    new_source = dict(inventory=dict(sha256="new_source", files={}), evolution_sha256="new_evolution", checkout_profile="new")
    reference = dict(selected=selected, input_sha256={"config": "c", "manifest": "m"},
        identity_sha256="identity", episodes_sha256="episodes", source_inventory_sha256="old_source",
        completed_episode_runs=100, reference_complete_verified=True)
    parent = dict(schema=runner.IDENTITY_SCHEMA, baseline=runner.BASELINE, variants=["V5"],
        options=copy.deepcopy(runner.OPTIONS), selected=selected, seed=runner.SEED, task_horizon=400,
        planned_episode_runs=20, max_physical_steps=8000, training=False, checkpoint_loaded=False,
        formal_thesis_evaluation=False, search_coverage_enabled=True,
        child_input_sha256={"config": "c", "manifest": "m"}, sources_before=new_source,
        sources_after=new_source, source_changed_from_reference=True, reference=reference,
        source_and_input_verification_passed=True, experiment_complete=True)
    ac_plan = dict(schema="ch3.safe_search.ac_plan.v1", variant="V5", options=copy.deepcopy(runner.OPTIONS),
        development_indices=list(runner.INDICES), seed=runner.SEED, max_steps=400,
        reference_identity_sha256="identity", reference_episodes_sha256="episodes",
        reference_source_inventory_sha256="old_source")
    return manifest, config, plan, ac_plan, parent, old_source


def write_child(output, selected, parent):
    output.mkdir(parents=True)
    child_identity = dict(schema="ch3.safe_search.paired_identity.v1", baseline=runner.BASELINE,
        variants=["V5"], seed=runner.SEED, steps=400, task_horizon=400, full_episodes=True,
        training=False, checkpoint_loaded=False, formal_thesis_evaluation=False, selected=[selected],
        search_coverage_enabled=True, input_sha256=parent["child_input_sha256"],
        sources_before=parent["sources_before"], sources_after=parent["sources_before"],
        source_and_input_verification_passed=True)
    row = dict(variant="V5", baseline=runner.BASELINE, terminal=True, full_episode_completed=True,
        full_episode_requested=True, physical_steps=2, found_step=1, found_within_budget=True,
        stop_reason="success", pre_found_collision=False, pre_found_exposure_steps=1,
        searcher_motion_stall_proxy_agent_steps=0, searcher_hold_agent_steps=0, wall_seconds=.1,
        search_coverage=dict(available=True, pre_found_exposure_steps=1),
        episode_result=dict(termination_reason="success", terminal_step=2, success=True,
                            first_collision_step=None, task_protocol="collision_terminal_v1"),
        **{key: selected[key] for key in ("original_episode_index", "scenario_id", "scenario_seed", "environment_innovation_seed")})
    row["complete_episode_row"] = dict(found=True, found_step=1, success=True, actual_length=2, episode_length=2,
        terminal_step=2, termination_reason="success", scenario_id=selected["scenario_id"],
        scenario_seed=selected["scenario_seed"], environment_innovation_seed=selected["environment_innovation_seed"],
        evaluation_episode_index=selected["original_episode_index"], optimizer_update_count=0,
        training_update=False, first_collision_step=None)
    arm = output / f"episode_{selected['original_episode_index']:04d}" / "V5"
    arm.mkdir(parents=True)
    write_json(arm / "summary.json", row)
    records = [dict(step_before=step-1, step_after=step,
        before=dict(step=step-1, found=step>1), after=dict(step=step, found=True), search_transition=step==1,
        search_coverage=dict(step=step, pre_found_exposure_steps=int(step==1))) for step in (1, 2)]
    import json
    (arm / "step_trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    summary = summarize([row], ["V5"], 1, True)
    summary["source_and_input_verification_passed"] = True
    for name, value in (("identity", child_identity), ("episodes", [row]), ("summary", summary)):
        write_json(output / f"{name}.json", value)
    return row, file_hash(arm / "step_trace.jsonl")


class ACRequestTests(unittest.TestCase):
    def test_plan_requires_only_ac_and_exact_reference(self):
        _, _, _, plan, parent, old_source = inputs()
        reference_identity = dict(sources_before=old_source)
        runner.validate_ac_plan(plan, reference_identity, "identity", "episodes")
        for kind in ("failure_policy", "integer_bool", "index", "source", "hash", "variant"):
            changed = copy.deepcopy(plan)
            if kind == "failure_policy": changed["options"]["failure_policy"] = True
            elif kind == "integer_bool": changed["options"]["path_safety"] = 1
            elif kind == "index": changed["development_indices"][0] = 1
            elif kind == "source": changed["reference_source_inventory_sha256"] = "new_source"
            elif kind == "hash": changed["reference_episodes_sha256"] = "other"
            else: changed["variant"] = "V4"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                runner.validate_ac_plan(changed, reference_identity, "identity", "episodes")

    def test_child_command_is_single_v5_original_seed_full_horizon(self):
        command = runner.child_command(Path("manifest"), Path("config"), Path("output"), 97)
        self.assertEqual(command[command.index("--variants")+1], "V5")
        self.assertEqual(command[command.index("--episode-indices")+1], "97")
        self.assertEqual(command[command.index("--seed")+1], "12729")
        self.assertEqual(command[command.index("--steps")+1], "400")
        self.assertIn("--full-episodes", command)
        self.assertIn("--search-coverage", command)

    def test_cli_requires_reference_and_explicit_execution(self):
        with self.assertRaises(SystemExit):
            runner.parser().parse_args(["--manifest", "m", "--output-dir", "o", "--reference-dir", "r"])
        args = runner.parser().parse_args(["--manifest", "m", "--output-dir", "o", "--reference-dir", "r", "--execute"])
        self.assertEqual(args.workers, 4)


class ACCompletenessTests(unittest.TestCase):
    def test_changed_source_is_explicit_and_complete20_trace_collection_works(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, plan, ac_plan, parent, _ = inputs()
            write_json(root / "plan.json", plan)
            write_json(root / "ac.json", ac_plan)
            parent["ac_plan_sha256"] = file_hash(root / "ac.json")
            rows, parent["trace_sha256"] = [], {}
            for selected in parent["selected"]:
                index = selected["original_episode_index"]
                row, trace = write_child(root / f"scene_{index:04d}", selected, parent)
                rows.append(row)
                parent["trace_sha256"][str(index)] = trace
            write_json(root / "identity.json", parent)
            write_json(root / "episodes.json", rows)
            with patch.object(runner, "PLAN", root / "plan.json"), patch.object(runner, "AC_PLAN", root / "ac.json"):
                identity, collected = runner.collect_completed_ac(root)
                self.assertEqual(collected, rows)
                self.assertTrue(identity["source_changed_from_reference"])
                self.assertEqual(len(collected), 20)
                parent["source_changed_from_reference"] = False
                write_json(root / "identity.json", parent)
                with self.assertRaisesRegex(ValueError, "mislabelled"):
                    runner.collect_completed_ac(root)

    def test_truncated_or_wrong_found_trace_cannot_be_complete(self):
        for mutation in ("truncate", "found", "coverage", "wrong_source", "terminal", "collision"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, _, _, _, parent, _ = inputs()
                selected = parent["selected"][0]
                row, _ = write_child(root / "child", selected, parent)
                trace = root / "child/episode_0000/V5/step_trace.jsonl"
                if mutation in ("truncate", "found", "coverage"):
                    import json
                    lines = [json.loads(line) for line in trace.read_text().splitlines()]
                    if mutation == "truncate": lines.pop()
                    elif mutation == "found": lines[0]["after"]["found"] = False
                    else: lines[0]["search_coverage"]["pre_found_exposure_steps"] = 0
                    trace.write_text("".join(json.dumps(r)+"\n" for r in lines), encoding="utf-8")
                elif mutation == "wrong_source":
                    identity = read_json(root / "child/identity.json")
                    identity["sources_after"] = {"unexpected": True}
                    write_json(root / "child/identity.json", identity)
                else:
                    if mutation == "terminal": row["terminal"] = False
                    else: row["episode_result"]["first_collision_step"] = 1
                    write_json(root / "child/episodes.json", [row])
                    write_json(root / "child/episode_0000/V5/summary.json", row)
                with self.assertRaises(ValueError):
                    runner.validate_child_ac(root / "child", selected, parent)

    def test_reference_root_cannot_differ_from_reconstructed_children(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "episodes.json", [])
            with patch.object(runner, "collect_completed", return_value=({}, [{"unexpected": True}])):
                with self.assertRaisesRegex(ValueError, "reference root episodes"):
                    runner.read_reference(root)


class ACOrchestrationTests(unittest.TestCase):
    def run_fake(self, temporary, *, fail=False):
        root = Path(temporary)
        manifest, config, plan, ac_plan, template, old_source = inputs()
        for name, value in (("manifest", manifest), ("config", config), ("plan", plan)):
            write_json(root / f"{name}.json", value)
        reference_dir = root / "reference"
        reference_dir.mkdir()
        old_identity = dict(sources_before=old_source, selected=template["selected"],
            child_input_sha256={str(root / "config.json"):file_hash(root / "config.json"),
                                str(root / "manifest.json"):file_hash(root / "manifest.json")})
        write_json(reference_dir / "identity.json", old_identity)
        write_json(reference_dir / "episodes.json", [])
        reference = dict(run_directory=str(reference_dir), selected=old_identity["selected"],
            input_sha256=old_identity["child_input_sha256"], completed_episode_runs=100, reference_complete_verified=True,
            identity_sha256=file_hash(reference_dir / "identity.json"), episodes_sha256=file_hash(reference_dir / "episodes.json"),
            source_inventory_sha256="old_source", source_evolution_sha256="old_evolution", source_checkout_profile="old")
        ac_plan.update(reference_identity_sha256=reference["identity_sha256"], reference_episodes_sha256=reference["episodes_sha256"])
        write_json(root / "ac.json", ac_plan)
        calls = []
        def fake_child(command, log):
            index = int(command[command.index("--episode-indices")+1])
            calls.append(index)
            Path(log).write_text("fake independent AC process", encoding="utf-8")
            if fail: return dict(returncode=7)
            output = Path(command[command.index("--output-dir")+1])
            parent = read_json(output.parent / "identity.json")
            selected = next(s for s in parent["selected"] if s["original_episode_index"] == index)
            write_child(output, selected, parent)
            return dict(returncode=0)
        with (patch.object(runner, "PLAN", root / "plan.json"), patch.object(runner, "AC_PLAN", root / "ac.json"),
              patch.object(runner, "read_reference", return_value=(old_identity, [], reference)),
              patch.object(runner, "_run_child", fake_child),
              patch("chapter3_bser.experiments.safe_search_v1.provenance.framework_sources", return_value=template["sources_before"]),
              patch("chapter3_bser.experiments.safe_search_v1.provenance.verify_sources")):
            if fail:
                with self.assertRaisesRegex(RuntimeError, "no complete-stage verdict"):
                    runner.run_ac_development(root / "manifest.json", root / "out", reference_dir=reference_dir,
                                              config_path=root / "config.json", workers=1)
            else:
                result = runner.run_ac_development(root / "manifest.json", root / "out", reference_dir=reference_dir,
                                                   config_path=root / "config.json", workers=4)
                self.assertTrue(result["experiment_complete"])
                self.assertEqual(len(runner.collect_completed_ac(root / "out")[1]), 20)
                self.assertEqual(result["variants"]["V5"]["found_rate"], 1.0)
                self.assertNotIn("paired_vs_v0", result)
        return calls, root / "out"

    def test_only20_v5_children_run_and_old_source_stays_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls, output = self.run_fake(temporary)
            self.assertEqual(sorted(calls), list(runner.INDICES))
            identity = read_json(output / "identity.json")
            self.assertEqual(identity["planned_episode_runs"], 20)
            self.assertEqual(identity["max_physical_steps"], 8000)
            self.assertTrue(identity["source_changed_from_reference"])

    def test_failure_stops_new_scheduling_and_preserves_20_arm_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls, output = self.run_fake(temporary, fail=True)
            self.assertEqual(calls, [0])
            failure = read_json(output / "failure.json")
            self.assertEqual(failure["planned_episode_runs"], 20)
            self.assertEqual(len(failure["unstarted_original_indices"]), 19)
            self.assertFalse((output / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
