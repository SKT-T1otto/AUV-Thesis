"""Fail-closed outcome/CLI/source checks for the manual experiment interface."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chapter3_bser.experiments.safe_search_v2 import run_windows as run
from chapter3_bser.experiments.safe_search_v2.options import ARMS, SETTINGS, parse_arms
from chapter3_bser.experiments.safe_search_v2 import provenance
from tests.test_safe_search_provenance import checked_copy


def scene(index=0):
    return dict(original_episode_index=index, scenario_id="scene"+str(index), scenario_seed=10+index,
                environment_innovation_seed=12729+index)


def row(index=0, arm="R0", found=None, collision=None):
    steps = collision if collision is not None else 400
    exposure = found if found is not None else steps
    reason = "obstacle_collision" if collision is not None else "timeout"
    result = dict(scene(index), baseline="B0_search_prior", arm=arm, parent_variant="V5",
        physical_steps=steps, found_step=found, found_within_budget=found is not None,
        terminal=True, full_episode_completed=True, full_episode_requested=True,
        pre_found_exposure_steps=exposure, pre_found_collision=collision is not None and (found is None or collision <= found),
        searcher_motion_stall_proxy_agent_steps=0, searcher_hold_agent_steps=0,
        effective_search_steps=1, stop_reason=reason,
        episode_result=dict(termination_reason=reason, terminal_step=steps, task_protocol="collision_terminal_v1",
                            success=False, first_collision_step=collision),
        controller={k: 0 for k in run.ZERO_FIELDS},
        search_coverage=dict(available=True, pre_found_exposure_steps=exposure))
    if arm != "R0":
        result["controller"]["found_search_v2"] = dict(options=ARMS[arm], settings=SETTINGS.record())
    return result


class RunnerTests(unittest.TestCase):
    def test_cli_defaults_to_plan_without_simulator_execution(self):
        plan = dict(planned_episode_runs=200, sources={}, selected=[])
        with patch.object(run, "request", return_value=plan), patch.object(run, "run_jobs") as jobs, patch.object(run, "run_episode") as episode:
            run.main(["--output-dir", "runs/unused"])
        jobs.assert_not_called()
        episode.assert_not_called()

    def test_arms_require_reference_and_do_not_reinterpret_v_names(self):
        self.assertEqual(parse_arms("R0,R4"), ["R0","R4"])
        for bad in ("R4", "R0,R0", "V5,R4", "R0,", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_arms(bad)

    def test_actual_timeouts_and_collisions_accepted_but_cutoffs_rejected(self):
        for found, collision in ((None,None), (100,None), (None,37), (20,37)):
            value = row(found=found, collision=collision)
            run.validate_terminal(value, scene(), "B0", "R0")
        for key, value in (("terminal",False), ("stop_reason","budget_cutoff"),
                           ("physical_steps",20), ("pre_found_collision",True), ("found_step",401)):
            broken = row()
            broken[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                run.validate_terminal(broken, scene(), "B0", "R0")

    def test_missing_coverage_wrong_arm_and_nonzero_actions_rejected(self):
        for bad in ("coverage", "switches", "residual"):
            value = row(arm="R4")
            if bad == "coverage": value["search_coverage"] = None
            elif bad == "switches": value["controller"]["found_search_v2"]["options"] = ARMS["R1"]
            else: value["controller"]["residual_action_max_abs"] = .1
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                run.validate_terminal(value, scene(), "B0", "R4")

    def test_penalized_time_keeps_failures_and_incomplete_rates_are_null(self):
        plan = dict(baselines=["B0"], arms=["R0","R4"], planned_episode_runs=40)
        rows = [row(i, arm, found=100 if i < 10 else None) for arm in plan["arms"] for i in range(20)]
        summary = run.summarize(rows, plan)
        arm = summary["groups"]["B0_R4"]
        self.assertEqual(arm["found_rate"], .5)
        self.assertEqual(arm["penalized_found_steps_mean_400"], 250)
        self.assertEqual(arm["found_steps_mean_conditional"], 100)
        self.assertTrue(summary["experiment_complete"])
        self.assertEqual(len(summary["paired_vs_R0"]["B0_R4_vs_R0"]), 20)
        incomplete = run.summarize(rows[:-1], plan)
        self.assertIsNone(incomplete["groups"]["B0_R4"]["found_rate"])
        self.assertFalse(incomplete["experiment_complete"])

    def test_atomic_json_writer_publishes_valid_value(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"progress.json"
            run.write_json(path, dict(completed=1))
            run.write_json(path, dict(completed=2))
            self.assertEqual(run.read_json(path), dict(completed=2))
            self.assertFalse(path.with_name("progress.json.partial").exists())

    def test_protected_output_rejected_even_for_resume(self):
        args = run.parser().parse_args(["--output-dir", str(run.ROOT/"outputs/retained"), "--resume"])
        with self.assertRaisesRegex(ValueError, "under runs"):
            run.run_jobs(args, {})

    def test_scheduler_and_resume_reuse_only_verified_completed_rows(self):
        plan = dict(baselines=["B0"], arms=["R0","R4"], planned_episode_runs=40,
                    selected=[scene(i) for i in range(20)], sources={}, input_sha256={})
        def completed(directory, identity, baseline, arm, selected):
            return row(selected["original_episode_index"], arm, found=100)
        with tempfile.TemporaryDirectory() as temporary, patch.object(run, "ROOT", Path(temporary)), \
             patch.object(run, "verify_sources"), patch.object(run, "read_completed", side_effect=completed) as read, \
             patch.object(run.subprocess, "Popen") as spawn, patch("builtins.print"):
            spawn.return_value.poll.return_value = 0
            args = run.parser().parse_args(["--output-dir", str(Path(temporary)/"runs/dev"), "--execute"])
            summary = run.run_jobs(args, plan)
            self.assertTrue(summary["experiment_complete"])
            self.assertEqual(spawn.call_count, 40)
            before = (args.output_dir/"episodes.json").read_bytes()
            spawn.reset_mock()
            args.resume = True
            run.run_jobs(args, plan)
            spawn.assert_not_called()
            self.assertEqual((args.output_dir/"episodes.json").read_bytes(), before)
            broken = run.read_json(args.output_dir/"episodes.json")
            broken[0]["found_step"] = 99
            run.write_json(args.output_dir/"episodes.json", broken)
            with self.assertRaisesRegex(ValueError, "differs from original"):
                run.run_jobs(args, plan)

    def test_program_failure_never_becomes_a_task_timeout(self):
        plan = dict(baselines=["B0"], arms=["R0"], planned_episode_runs=20,
                    selected=[scene(i) for i in range(20)], sources={}, input_sha256={})
        with tempfile.TemporaryDirectory() as temporary, patch.object(run, "ROOT", Path(temporary)), \
             patch.object(run, "verify_sources"), patch.object(run.subprocess, "Popen") as spawn, patch("builtins.print"):
            spawn.return_value.poll.return_value = 1
            args = run.parser().parse_args(["--output-dir", str(Path(temporary)/"runs/dev"), "--workers", "1", "--execute"])
            with self.assertRaises(RuntimeError):
                run.run_jobs(args, plan)
            self.assertEqual(run.read_json(args.output_dir/"episodes.json"), [])
            self.assertFalse(run.read_json(args.output_dir/"summary.json")["experiment_complete"])
            self.assertEqual(len(list(args.output_dir.glob("interruption_*.json"))), 1)

    def test_final_source_failure_cannot_publish_complete_summary(self):
        plan = dict(baselines=["B0"], arms=["R0"], planned_episode_runs=20,
                    selected=[scene(i) for i in range(20)], sources={}, input_sha256={})
        calls = []
        def verify(_):
            calls.append(1)
            if len(calls) == 41:  # twenty launches, twenty completions, final check
                raise ValueError("source changed")
        with tempfile.TemporaryDirectory() as temporary, patch.object(run, "ROOT", Path(temporary)), \
             patch.object(run, "verify_sources", side_effect=verify), \
             patch.object(run, "read_completed", side_effect=lambda d,p,b,a,s: row(s["original_episode_index"],a)), \
             patch.object(run.subprocess, "Popen") as spawn, patch("builtins.print"):
            spawn.return_value.poll.return_value = 0
            args = run.parser().parse_args(["--output-dir", str(Path(temporary)/"runs/dev"), "--execute"])
            with self.assertRaisesRegex(ValueError, "source changed"):
                run.run_jobs(args, plan)
            summary = run.read_json(args.output_dir/"summary.json")
            self.assertFalse(summary["experiment_complete"])
            self.assertFalse(summary["source_and_input_verification_passed"])
            self.assertEqual(len(run.read_json(args.output_dir/"episodes.json")), 20)


class EvolutionTests(unittest.TestCase):
    def test_exact_additive_profile_preserves_all_old_runtime_files(self):
        current = provenance.framework_sources()
        parent = provenance.old.read_json(provenance.old.ROOT/provenance.old.MANIFEST)
        before = parent["profiles"][current["checkout_profile"]]["files"]
        # The R transition's end state is historical; current D2 source bytes
        # must also pass the complete successor chain above.
        sealed = provenance.old.read_json(provenance.old.ROOT/provenance.MANIFEST)
        after = sealed["profiles"][current["checkout_profile"]]["files"]
        for path, expected in before.items():
            if path != provenance.HOOK:
                self.assertEqual(after[path], expected, path)
        self.assertEqual(current["historical_record_count"], 27)

    def test_new_sources_and_review_document_are_not_exempted(self):
        with checked_copy() as root:
            for name in ("chapter3_bser/experiments/safe_search_v2/motion.py", provenance.REVIEW):
                target = root/name
                before = target.read_bytes()
                try:
                    target.write_bytes(before+b"\n")
                    with self.assertRaises(ValueError):
                        provenance.old.framework_sources(root)
                finally:
                    target.write_bytes(before)

    def test_recomputed_manifest_cannot_change_scope_or_before_binding(self):
        with checked_copy() as root:
            original = provenance.old.read_json(root/provenance.MANIFEST)
            for mode in ("scope", "before", "parent", "unused_profile"):
                changed = copy.deepcopy(original)
                if mode == "scope": changed["added_paths"].append("core/unreviewed.py")
                elif mode == "before": changed["profiles"]["windows_existing"]["changes"][provenance.HOOK]["before"] = "0"*64
                elif mode == "parent": changed["parent_manifest_sha256"] = "0"*64
                else: changed["profiles"]["git_lf"]["sha256"] = "0"*64
                changed.pop("sha256")
                changed["sha256"] = provenance.old.digest(changed)
                (root/provenance.MANIFEST).write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    provenance.old.framework_sources(root)


if __name__ == "__main__":
    unittest.main()
