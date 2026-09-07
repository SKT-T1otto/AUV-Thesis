"""Synthetic fixtures only; no real checkpoint or 30-episode diagnostic run."""

import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.experiments.phase1c_prrac import run_searcher_residual_trace as runner
from chapter3_bser.experiments.phase1c_prrac.search_continuity.diagnostics import SearchContinuityDiagnostics
from chapter3_bser.experiments.phase1c_prrac.searcher_residual_trace import SearcherResidualTrace, SCHEMA
from scripts import analyze_searcher_residual_trace as analysis


def fake_worker(work):
    sid, mode = work
    return dict(episode=dict(scenario_id=sid, evaluation_mode=mode),
                action=[dict(scenario_id=sid, mode=mode, step=step, agent_id=agent) for step in (1, 0) for agent in (2, 0, 1)],
                mission=[dict(scenario_id=sid, mode=mode, step=step) for step in (1, 0)])


def action_row(step, agent=0, offset=0.):
    return dict(step=step, agent_id=agent, stage=0, position_x=step+offset, position_y=agent, position_z=0.,
                navigation_target_x=10., navigation_target_y=0., navigation_target_z=0.,
                final_action_0=offset, final_action_1=0., final_action_2=0., waypoint_cursor=0,
                collision_event=False, collision_streak=0, c2_active=False, c2_state_or_tier="NORMAL_SEARCH",
                waypoint_switch_event=False, negative_alignment=True, residual_prior_cosine=-.5,
                raw_residual_norm=.8, applied_residual_norm=.8, residual_contribution_ratio=None, route_active=True)


class TraceSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root/"fixture.pt"
        self.checkpoint.write_bytes(b"not a trained checkpoint; validation fixture only")
        self.cases = self.root/"cases.csv"
        self.source = self.root/"source"
        config = json.loads((evaluator.ROOT/"configs/chapter3/bser_phase1c_prrac_s2a1_local_connector_ablation.json").read_text())
        config["search_recovery_variants"] = [runner.C2]
        config["search_collision_recovery"]["variants"] = [runner.C2]
        self.manifest = dict(scenario_seed=config["scenario_seed"], evaluation_episodes=100,
            scenarios=[dict(scenario_id=f"s{i:03}", scenario_seed=1000+i, max_steps=400, physical_fixture=[i, i+1]) for i in range(100)])
        self.manifest["manifest_sha256"] = evaluator._hash(self.manifest)
        for mode in runner.MODES:
            folder = self.source/mode
            current = dict(config, modes=[mode])
            current["resolved_config_hash"] = evaluator._hash(current)
            evaluator._write_json(folder/"resolved_evaluation_config.json", current)
            evaluator._write_json(folder/"evaluation_manifest.json", self.manifest)
            resolved = evaluator._load_config(folder/"resolved_evaluation_config.json")
            rows = [dict(scenario_id=f"s{i:03}", scenario_seed=1000+i, checkpoint=str(self.checkpoint),
                         manifest_sha256=self.manifest["manifest_sha256"], max_steps=400, evaluation_mode=mode,
                         execution_variant="B1_ATOMIC_LAST_VALID", search_recovery_variant=runner.C2,
                         runtime_integration_mode="native", explore=False, training_update=False,
                         checkpoint_config_hash="fixture-config", checkpoint_episode=100,
                         checkpoint_runtime_revision=config["checkpoint_runtime_revision"], evaluation_runtime_revision=config["evaluation_runtime_revision"],
                         execution_overlay_config_hash=evaluator._hash(resolved["execution_continuity"]), search_collision_recovery_config_hash=resolved["search_collision_recovery_config_hash"],
                         found=True, success=i < 5 if mode == "full_prrac" else 5 <= i < 15) for i in range(100)]
            evaluator._write_csv(folder/"episode_evaluation.csv", rows)
        self.config = self.source/"searcher_residual_off"/"resolved_evaluation_config.json"
        evaluator._write_csv(self.cases, [dict(scenario_id=f"s{i:03}", transition_type=runner.HELP if i < 5 else runner.HURT) for i in range(15)])

    def validate(self, limit=None):
        return runner.validate_sources(self.source, self.cases, self.config, self.checkpoint, limit)

    def test_formal_count_seeds_and_original_indices_without_generation(self):
        with patch.object(evaluator, "build_scenario_manifests", side_effect=AssertionError("must not regenerate")):
            histories, selected = self.validate()
        self.assertEqual(len(selected), 15)
        self.assertEqual(2*len(selected), 30)
        for entry in selected:
            self.assertEqual(entry["scenario"], self.manifest["scenarios"][entry["episode_index"]])
        self.assertEqual(sum(e["transition_type"] == runner.HELP for e in selected), 5)
        self.assertEqual(sum(e["transition_type"] == runner.HURT for e in selected), 10)
        self.assertEqual(histories[runner.MODES[0]]["manifest"], histories[runner.MODES[1]]["manifest"])

    def test_bad_case_counts_rejected_even_in_smoke(self):
        evaluator._write_csv(self.cases, evaluator._read_csv(self.cases)[:-1])
        for limit in (None, 1):
            with self.assertRaisesRegex(ValueError, "5 help.*10 hurt"):
                self.validate(limit)

    def test_case_outside_original_manifest_rejected(self):
        rows = evaluator._read_csv(self.cases)
        rows[0]["scenario_id"] = "not-in-manifest"
        evaluator._write_csv(self.cases, rows)
        with self.assertRaisesRegex(ValueError, "not in original"):
            self.validate()

    def test_different_manifests_rejected(self):
        path = self.source/"full_prrac"/"evaluation_manifest.json"
        manifest = runner.read_json(path)
        manifest["scenarios"][0]["physical_fixture"] = [999]
        manifest["manifest_sha256"] = evaluator._hash({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        evaluator._write_json(path, manifest)
        rows = evaluator._read_csv(path.parent/"episode_evaluation.csv")
        for row in rows:
            row["manifest_sha256"] = manifest["manifest_sha256"]
        evaluator._write_csv(path.parent/"episode_evaluation.csv", rows)
        with self.assertRaisesRegex(ValueError, "manifests differ"):
            self.validate()

    def test_smoke_only_slices_after_full_validation(self):
        _, selected = self.validate(1)
        self.assertEqual([e["episode_index"] for e in selected], [0, 5])
        self.assertTrue(all(e["scenario"]["max_steps"] == 400 for e in selected))

    def test_seed_and_behavior_mismatch_fail(self):
        rows_path = self.source/"full_prrac"/"episode_evaluation.csv"
        rows = evaluator._read_csv(rows_path)
        rows[0]["scenario_seed"] = -1
        evaluator._write_csv(rows_path, rows)
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            self.validate()

    def test_nonempty_output_refused_before_checkpoint_load(self):
        output = self.root/"existing"
        output.mkdir()
        (output/"retained").write_text("keep")
        with patch.object(evaluator, "load_prrac_checkpoint", side_effect=AssertionError("must not load")):
            with self.assertRaises(FileExistsError):
                runner.run(SimpleNamespace(output_dir=output, source_root=self.source))

    def test_spawn_parent_sorting_stable(self):
        jobs = [(sid, mode) for sid in ("b", "a") for mode in reversed(runner.MODES)]
        serial = runner.collect_jobs(jobs, 1, worker=fake_worker)
        parallel = runner.collect_jobs(jobs, 4, worker=fake_worker)
        self.assertEqual(serial, parallel)
        runner.write_results(self.root/"serial", serial)
        runner.write_results(self.root/"parallel", list(reversed(parallel)))
        for mode in runner.MODES:
            for name in ("episode_evaluation.csv", "searcher_action_trace.csv", "mission_step_trace.csv"):
                self.assertEqual((self.root/"serial"/mode/name).read_bytes(), (self.root/"parallel"/mode/name).read_bytes())

    def test_runner_builds_exact_30_jobs_without_simulation(self):
        config = evaluator._load_config(self.config)
        payload = dict(schema=evaluator.CHECKPOINT_SCHEMA, completed_episode=100,
            metadata=dict(config_hash="fixture-config", execution_runtime_revision=config["checkpoint_runtime_revision"],
                          architecture={}, loss={}, reward={}), prrac_training_state=dict(gamma=.95, tau=.005))
        learner = SimpleNamespace(policy_snapshot=lambda: ({}, {}, {}, {}), search_value_config={})
        captured = []
        source_rows = {mode: runner.indexed(evaluator._read_csv(self.source/mode/"episode_evaluation.csv")) for mode in runner.MODES}
        def collect(jobs, workers, on_result):
            captured.extend(jobs)
            for work in jobs:
                info = work["job"]["checkpoint_info"]
                sid = work["job"]["scenario"]["scenario_id"]
                on_result(dict(episode=source_rows[info["evaluation_mode"]][sid], action=[], mission=[]))
        args = SimpleNamespace(output_dir=self.root/"run", source_root=self.source, cases_csv=self.cases,
                               config=self.config, checkpoint=self.checkpoint, max_per_transition=None, workers=1)
        with patch.object(evaluator, "load_prrac_checkpoint", return_value=(learner, payload)), patch.object(runner, "collect_jobs", side_effect=collect):
            self.assertEqual(runner.run(args), 0)
        self.assertEqual(len(captured), 30)
        for work in captured:
            job = work["job"]
            self.assertEqual(job["scenario"], self.manifest["scenarios"][job["episode_index"]])
            self.assertEqual(job["config"]["max_steps"], 400)
            self.assertEqual(job["failure_trace"]["enabled"], config["failure_trace"]["enabled"])
        self.assertEqual(runner.read_json(args.output_dir/"trace_manifest.json")["status"], "completed")


class TraceMetricTests(unittest.TestCase):
    def diagnostic(self, zero=False, enabled=True):
        state = SimpleNamespace(agents=[SimpleNamespace(agent_id=a, position=(0., 0., 0.)) for a in range(4)], step=0,
                                occupancy=SimpleNamespace(known_mask=np.ones(2)), target_belief=SimpleNamespace(entropy=.5, peak_probability=.5))
        after = copy.deepcopy(state)
        after.step = 1
        assignment = SimpleNamespace(assignment_kind="SEARCH", reachable=True, hold_state=False, assignment_id="a",
                                     final_waypoint=(1., 0., 0.), tracking_waypoint=(0., 0., 0.) if zero else (1., 0., 0.))
        guidance = SimpleNamespace(assignment_for=lambda agent: assignment)
        raw = torch.ones(4, 3)*.8
        applied = evaluator._apply_residual_mode(raw, "searcher_residual_off", 0)
        outputs = [SimpleNamespace(residual_mix=torch.zeros(1, 3) if zero else torch.ones(1, 3), alignment_cosine=torch.tensor([-.5])) for _ in range(4)]
        diagnostics, rows = SearchContinuityDiagnostics(), []
        diagnostics.begin_episode(state)
        before_raw, before_applied = raw.clone(), applied.clone()
        diagnostics.observe_transition(stage_before=0, stage_after=0, installed_guidance=guidance,
            planning_state_before=state, planning_state_after=after, collision_flags=[False]*4,
            raw_actions=raw, applied_actions=applied, actor_outputs=outputs, residual_contribution_ratios=.25,
            **({"trace_rows": rows} if enabled else {}))
        self.assertTrue(torch.equal(raw, before_raw))
        self.assertTrue(torch.equal(applied, before_applied))
        return diagnostics.summary(found=False, max_steps=400), rows

    def test_raw_applied_and_authoritative_aggregate_no_change(self):
        summary, rows = self.diagnostic()
        self.assertEqual(summary, self.diagnostic(enabled=False)[0])
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(np.allclose(row["raw_residual"], .8) for row in rows))
        self.assertTrue(all(row["applied_residual"] == [0., 0., 0.] for row in rows))
        self.assertTrue(all(row["negative_alignment"] is True for row in rows))
        self.assertTrue(all(row["residual_contribution_ratio"] is None for row in rows))
        self.assertEqual(summary["searcher_residual_contribution_ratio_mean_pre_found"], .25)

    def test_zero_norm_cosine_is_NA(self):
        summary, rows = self.diagnostic(zero=True)
        self.assertTrue(all(r["residual_prior_cosine"] is None and r["negative_alignment"] is None for r in rows))
        self.assertIsNone(summary["searcher_residual_negative_alignment_rate_pre_found"])

    def test_distance_first_divergence_and_non_counterfactual_label(self):
        left = [action_row(t) for t in range(4)]
        right = [action_row(t, offset=(0., .6, 1.2, 1.2)[t]) for t in range(4)]
        right[0]["final_action_0"] = 1.
        mission = [dict(step=t, state_step=t+1, found_event=t == 3, found=t == 3, team_residual_contribution_ratio=.25) for t in range(4)]
        divergence, summary, _ = analysis.analyze_scenario("s", runner.HURT, [left, right], [mission, mission])
        self.assertEqual(analysis.distance(left[1], right[1], "position"), .6000000000000001)
        self.assertEqual(divergence["first_action_difference_step"], 0)
        self.assertEqual(divergence["first_position_separation_0p5_step"], 1)
        self.assertEqual(divergence["first_position_separation_1p0_step"], 2)
        self.assertFalse(divergence["same_state_counterfactual"])
        self.assertEqual(summary["before_first_position_divergence_observed_steps"], 1)
        self.assertEqual(summary["R0_residual_ratio_before_first_position_divergence"], .25)

    def test_no_interpolation_duplicate_keys_and_censored_followups(self):
        with self.assertRaises(ValueError):
            analysis.action_index([action_row(0), action_row(0)])
        short = [action_row(t) for t in range(3)]
        self.assertEqual(analysis.followups(short)["none"], 0)
        self.assertEqual(analysis.followups(short)["censored_windows"], 3)
        full = [action_row(t) for t in range(6)]
        full[2]["collision_event"] = True
        follow = analysis.followups(full)
        self.assertEqual(follow["complete_windows"], 1)
        self.assertEqual(follow["collision"], 1)
        self.assertEqual(follow["none"], 0)

    def test_native_evaluator_trace_no_op_both_modes(self):
        from tests.search_value_audit_support import small_job
        for mode in runner.MODES:
            with self.subTest(mode=mode):
                job = small_job(max_steps=1)
                job["checkpoint_info"]["evaluation_mode"] = mode
                actions_seen = []
                original_make = evaluator._make_env
                def factory(*args, **kwargs):
                    env = original_make(*args, **kwargs)
                    original_step = env.step
                    def step(actions):
                        actions_seen.append(actions.detach().cpu().clone())
                        return original_step(actions)
                    env.step = step
                    return env
                with patch.object(evaluator, "_make_env", side_effect=factory):
                    bare = evaluator._evaluate_episode_job(job)
                    rng_before = (torch.get_rng_state().clone(), random.getstate(), repr(np.random.get_state()))
                    sink = SearcherResidualTrace(job["scenario"], mode, runner.HURT)
                    traced = evaluator._evaluate_episode_job(job, searcher_trace=sink)
                self.assertEqual(bare, traced)
                self.assertTrue(torch.equal(actions_seen[0], actions_seen[1]))
                self.assertTrue(torch.equal(rng_before[0], torch.get_rng_state()))
                self.assertEqual(rng_before[1:], (random.getstate(), repr(np.random.get_state())))
                self.assertEqual(len(sink.action_rows), 3)
                self.assertEqual(len(sink.mission_rows), 1)
                self.assertEqual(np.mean([r["raw_residual_norm"] for r in sink.action_rows]), traced["episode"]["searcher_raw_residual_norm_mean_pre_found"])
                self.assertEqual(sink.mission_rows[0]["team_residual_contribution_ratio"], traced["episode"]["searcher_residual_contribution_ratio_mean_pre_found"])
                if mode == "searcher_residual_off":
                    self.assertTrue(all(r["applied_residual_norm"] == 0 for r in sink.action_rows))


class OfflineTraceFilesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        selected = [dict(scenario_id=f"s{i:02}", scenario_seed=100+i, transition_type=runner.HELP if i < 5 else runner.HURT) for i in range(15)]
        self.manifest = dict(schema=SCHEMA, status="completed", selected=selected, scenario_count=15,
                             episode_runs_expected=30, smoke=False, trace_files={})
        for mode in runner.MODES:
            actions, missions, episodes = [], [], []
            for selection in selected:
                sid, group = selection["scenario_id"], selection["transition_type"]
                success = (group == runner.HELP) == (mode == "full_prrac")
                for step in range(3):
                    base = dict(selection, mode=mode)
                    for agent in range(3):
                        actions.append(dict(base, **action_row(step, agent, .6 if mode == "searcher_residual_off" and step >= 1 else 0.)))
                    missions.append(dict(base, step=step, state_step=step+1, found=step == 2, found_event=step == 2,
                                         contact=success and step == 2, success=success and step == 2,
                                         team_residual_contribution_ratio=.25))
                episodes.append(dict(scenario_id=sid, scenario_seed=selection["scenario_seed"], evaluation_mode=mode,
                                     found=True, contact_episode=success, success=success, episode_length=3))
            for filename, rows in (("searcher_action_trace.csv", actions), ("mission_step_trace.csv", missions), ("episode_evaluation.csv", episodes)):
                path = self.root/mode/filename
                evaluator._write_csv(path, rows)
                self.manifest["trace_files"][path.relative_to(self.root).as_posix()] = runner.sha(path)
        evaluator._write_json(self.root/"trace_manifest.json", self.manifest)

    def analyze(self):
        return analysis.analyze(self.root/runner.MODES[0], self.root/runner.MODES[1], self.root/"analysis")

    def test_all_offline_outputs_and_scenario_equal_statistics(self):
        result = self.analyze()
        self.assertEqual((result["scenario_count"], result["help_count"], result["hurt_count"], result["episode_runs_expected"]), (15, 5, 10, 30))
        self.assertEqual(len(analysis.read_csv(self.root/"analysis"/"paired_first_divergence.csv")), 15)
        self.assertEqual(len(analysis.read_csv(self.root/"analysis"/"residual_help_trace_summary.csv")), 5)
        self.assertEqual(len(analysis.read_csv(self.root/"analysis"/"residual_hurt_trace_summary.csv")), 10)
        self.assertEqual(result["groups"]["hurt"]["R0_negative_alignment_before_first_position_divergence"]["valid"], 10)
        self.assertFalse(result["same_state_counterfactual"])
        with self.assertRaises(FileExistsError):
            self.analyze()

    def test_offline_detects_missing_steps_even_with_updated_file_hash(self):
        path = self.root/runner.MODES[0]/"searcher_action_trace.csv"
        rows = evaluator._read_csv(path)
        rows.pop(0)
        evaluator._write_csv(path, rows)
        self.manifest["trace_files"][path.relative_to(self.root).as_posix()] = runner.sha(path)
        evaluator._write_json(self.root/"trace_manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "missing PRE_FOUND"):
            self.analyze()

    def test_reproduction_mismatch_blocks_mechanism_report(self):
        self.manifest["status"] = "completed_with_mismatches"
        evaluator._write_json(self.root/"trace_manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "reproduction mismatched"):
            self.analyze()


if __name__ == "__main__":
    unittest.main()
