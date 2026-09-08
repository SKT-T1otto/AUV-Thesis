"""Synthetic unit fixtures only; never the real six/75-run experiment."""

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.experiments.phase1c_prrac import run_residual_branch_sensitivity as runner
from chapter3_bser.experiments.phase1c_prrac.residual_branch_diagnostic import ResidualBranchDiagnostic, scale_commands
from chapter3_bser.experiments.phase1c_prrac.searcher_residual_trace import SearcherResidualTrace
from scripts import analyze_searcher_residual_branch_sensitivity as analysis
from tests import test_searcher_residual_trace as trace_tests

action_row = trace_tests.action_row


def selection():
    return dict(scenario_id="unit", transition_type=analysis.trace.HURT, branch_step=2, anchor_step=1, branch_agents=[0, 1])


def synthetic_result(alpha, *, targets=None, horizon=10):
    selected = selection()
    rows = []
    for step in range(1, horizon+2):
        for agent in range(3):
            row = action_row(step, agent)
            row.update(selected, alpha=alpha, relative_step=step-1, sample_kind="endpoint" if step == horizon+1 else "transition", found_event=False)
            if targets and step > 1:
                row["navigation_target_x"] = targets(step)
            rows.append(row)
    return dict(scenario_id="unit", selection=selected, alpha=alpha, anchor_state_hash="same", complete=True,
                anchor_fingerprint={}, rows=rows, missions=[dict(step=t, found_event=False) for t in range(1, horizon+1)])


class BranchMetricTests(unittest.TestCase):
    def test_anchor_uses_discrete_step_not_position(self):
        left = [action_row(t, a) for t in range(5) for a in range(3)]
        right = copy.deepcopy(left)
        for row in right:
            if row["step"] >= 2 and row["agent_id"] == 1:
                row["waypoint_cursor"] = 1
        result = analysis.select_anchor(dict(scenario_id="unit", transition_type=analysis.trace.HURT,
            first_waypoint_cursor_difference_step=2, first_navigation_target_difference_step=None,
            first_position_separation_0p5_step=0), left, right)
        self.assertEqual(result["anchor_step"], 1)
        self.assertEqual(result["branch_agents"], [1])

    def test_all_tied_agents_retained(self):
        left = [action_row(t, a) for t in range(4) for a in range(3)]
        right = copy.deepcopy(left)
        for row in right:
            if row["step"] >= 2 and row["agent_id"] in (0, 2):
                row["navigation_target_x"] += 1
        result = analysis.select_anchor(dict(scenario_id="unit", transition_type=analysis.trace.HELP,
            first_navigation_target_difference_step=2, first_waypoint_cursor_difference_step=None), left, right)
        self.assertEqual(result["branch_agents"], [0, 2])

    def test_missing_or_falsified_branch_step_rejected(self):
        rows = [action_row(t, a) for t in range(3) for a in range(3)]
        for step in (None, 1):
            with self.assertRaises(ValueError):
                analysis.select_anchor(dict(scenario_id="unit", transition_type=analysis.trace.HURT,
                    first_navigation_target_difference_step=step, first_waypoint_cursor_difference_step=None), rows, rows)

    def test_alpha_zero_only_selected_searchers_and_executor_unchanged(self):
        actions = torch.arange(12, dtype=torch.float32).reshape(4, 3)/12
        original = actions.clone()
        applied = scale_commands(actions, [1], 0.)
        torch.testing.assert_close(applied[1], torch.zeros(3), rtol=0, atol=0)
        for agent in (0, 2, 3):
            torch.testing.assert_close(applied[agent], actions[agent], rtol=0, atol=0)
        torch.testing.assert_close(actions, original, rtol=0, atol=0)
        self.assertIs(scale_commands(actions, [0, 1], 1.), actions)

    def test_half_exact_and_executor_cannot_be_selected(self):
        actions = torch.tensor([[.8, -.9, 1.], [.7, .2, -.5], [.1, .2, .3], [.4, .5, .6]])
        applied = scale_commands(actions, [0, 2], .5)
        torch.testing.assert_close(applied[[0, 2]], actions[[0, 2]]*.5, rtol=0, atol=0)
        torch.testing.assert_close(applied[[1, 3]], actions[[1, 3]], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            scale_commands(actions, [3], .5)

    def test_trace_preserves_authoritative_diagnostic_norms(self):
        sink = ResidualBranchDiagnostic(dict(scenario_id="unit", scenario_seed=1), selection(), .5, horizon=1)
        sink.current = dict(step=1, positions=np.zeros((4, 3)).tolist(), targets=np.ones((4, 3)).tolist(),
                            cursors=[0]*3, route_hashes=["route"]*3, c2_active=[], c2_modes={})
        sink.raw = sink.full_applied = np.ones((4, 3), dtype=np.float32)
        sink.applied = sink.raw*.5
        sink.action_rows = [dict(step=1, agent_id=a, raw_residual_norm=1.23456789, applied_residual_norm=.23456789,
                                waypoint_switch_event=True) for a in range(3)]
        sink.mission_rows = [dict(found_event=False)]
        runtime = SimpleNamespace(_last_prior_acc=torch.zeros(4, 3), _agent_acc=torch.zeros(4, 3), collision_flags=torch.zeros(4))
        with patch.object(sink, "boundary", return_value=dict(sink.current, step=2)):
            sink.after_transition(state=SimpleNamespace(step=2), env=SimpleNamespace(unwrapped=runtime), bridge=None, guidance=None, recovery=None)
        self.assertEqual(sink.rows[0]["raw_residual_norm"], 1.23456789)
        self.assertEqual(sink.rows[0]["applied_residual_norm"], .23456789)
        self.assertTrue(sink.rows[0]["waypoint_switch_event"])
        self.assertFalse(sink.rows[0]["path_cursor_advance_event"])

    def classify(self, values, delay=2):
        indices = {a: analysis.rows_index(synthetic_result(a, targets=lambda t, v=v: 10 if t < delay else v)["rows"]) for a, v in zip(analysis.ALPHAS, values)}
        return analysis.classify(indices, 1)

    def test_monotonic_immediate_coarse_interval(self):
        result = self.classify([11, 11, 10, 10, 10])
        self.assertEqual(result["classification"], "immediate_threshold")
        self.assertEqual(result["alpha_switch_interval"], [.25, .5])
        self.assertEqual(result["first_discrete_branch_divergence"], 2)

    def test_delayed_and_stable(self):
        self.assertEqual(self.classify([11, 11, 10, 10, 10], delay=5)["classification"], "delayed_threshold")
        self.assertEqual(self.classify([10]*5)["classification"], "stable")

    def test_non_monotonic_and_unordered_not_forced(self):
        result = self.classify([10, 11, 10, 11, 10])
        self.assertEqual(result["classification"], "non_monotonic")
        self.assertIsNone(result["alpha_switch_interval"])
        self.assertEqual(self.classify([10, 11, 12, 13, 14])["classification"], "unresolved")

    def test_position_separation_and_immediate_comparison(self):
        results = [synthetic_result(a) for a in analysis.ALPHAS]
        for row in results[0]["rows"]:
            if row["step"] > 1 and row["agent_id"] == 0:
                row["position_x"] += .3
                row["position_y"] += .4
                row["waypoint_cursor"] = 2
                row["navigation_target_x"] += 1
        control = results[-1]
        rows, steps, check = analysis.summarize_scenario(results, control["rows"], control["missions"])
        self.assertTrue(check["interpretation_allowed"])
        self.assertAlmostEqual(rows[0]["position_separation_at_horizon"], .5)
        self.assertTrue(rows[0]["immediate_waypoint_difference_vs_alpha1"])
        self.assertTrue(rows[0]["immediate_navigation_target_difference_vs_alpha1"])
        self.assertEqual(rows[0]["first_position_separation_0p1_step_vs_alpha1"], 2)
        self.assertEqual(len(steps), 5*11*3)

    def test_control_failure_prohibits_interpretation(self):
        results = [synthetic_result(a) for a in analysis.ALPHAS]
        reference = copy.deepcopy(results[-1]["rows"])
        reference[0]["position_x"] += 1
        rows, _, check = analysis.summarize_scenario(results, reference, results[-1]["missions"])
        self.assertFalse(check["interpretation_allowed"])
        self.assertTrue(check["control_reproduction_mismatches"])
        self.assertTrue(all(r["classification"] == "unresolved" and r["position_separation_at_horizon"] is None for r in rows))

    def test_same_state_mismatch_and_incomplete_fail(self):
        for field, value in (("anchor_state_hash", "different"), ("complete", False)):
            results = [synthetic_result(a) for a in analysis.ALPHAS]
            results[0][field] = value
            _, _, check = analysis.summarize_scenario(results, results[-1]["rows"], results[-1]["missions"])
            self.assertFalse(check["interpretation_allowed"])

    def test_missing_control_endpoint_is_not_interpolated(self):
        result = synthetic_result(1.)
        self.assertTrue(analysis.control_reproduction(result, result["rows"][:-1], result["missions"]))


class BranchSourceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = trace_tests.TraceSourceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.root, self.trace_root = f.root, f.root/"trace"
        histories, entries = f.validate()
        selected = [dict(scenario_id=e["scenario"]["scenario_id"], scenario_seed=e["scenario"]["scenario_seed"], canonical_episode_index=e["episode_index"],
            transition_type=e["transition_type"], scenario_sha256=evaluator._hash(e["scenario"])) for e in entries]
        self.divergence = self.root/"paired.csv"
        pairs = []
        for mode in runner.source_trace.MODES:
            actions, missions, episodes = [], [], []
            for sel in selected:
                sid = sel["scenario_id"]
                for t in range(14):
                    for a in range(3):
                        row = dict(action_row(t, a), scenario_id=sid, scenario_seed=sel["scenario_seed"], mode=mode, transition_type=sel["transition_type"])
                        if mode == "searcher_residual_off" and t >= 2 and a in (0, 1):
                            row["navigation_target_x"] += 1
                        actions.append(row)
                    missions.append(dict(scenario_id=sid, scenario_seed=sel["scenario_seed"], mode=mode, transition_type=sel["transition_type"], step=t, state_step=t+1, found_event=False))
                episodes.append(dict(scenario_id=sid, scenario_seed=sel["scenario_seed"], evaluation_mode=mode))
            for filename, rows in (("searcher_action_trace.csv", actions), ("mission_step_trace.csv", missions), ("episode_evaluation.csv", episodes)):
                evaluator._write_csv(self.trace_root/mode/filename, rows)
        for sel in selected:
            pairs.append(dict(scenario_id=sel["scenario_id"], transition_type=sel["transition_type"], first_navigation_target_difference_step=2, first_waypoint_cursor_difference_step="NA"))
        evaluator._write_csv(self.divergence, pairs)
        self.manifest = dict(schema="bser.searcher_residual_trace.v1", status="completed", smoke=False, scenario_count=15, episode_runs_expected=30,
            max_steps=400, device="cpu", explore=False, training_update=False, checkpoint_path=str(f.checkpoint), checkpoint_sha256=runner.source_trace.sha(f.checkpoint),
            config_sha256=runner.source_trace.sha(f.config), source_experiment_root=str(f.source), selected=selected,
            source_full_manifest_hash=f.manifest["manifest_sha256"], source_searcher_off_manifest_hash=f.manifest["manifest_sha256"],
            source_files={str(f.source/mode/name): runner.source_trace.sha(f.source/mode/name) for mode in runner.source_trace.MODES for name in ("episode_evaluation.csv", "evaluation_manifest.json", "resolved_evaluation_config.json")},
            trace_files={p.relative_to(self.trace_root).as_posix(): runner.source_trace.sha(p) for p in self.trace_root.glob("*/*.csv")})
        evaluator._write_json(self.trace_root/"trace_manifest.json", self.manifest)
        self.args = SimpleNamespace(source_root=f.source, trace_root=self.trace_root, paired_first_divergence=self.divergence, checkpoint=f.checkpoint, config=f.config,
                                    output_dir=self.root/"out", max_per_transition=None, alphas=list(analysis.ALPHAS), workers=1, device="cpu")

    def test_formal_15_and_75_job_assembly_without_simulation(self):
        histories, anchors, _, _, _ = runner.validate_inputs(self.args)
        config = evaluator._load_config(self.fixture.config)
        payload = dict(schema=evaluator.CHECKPOINT_SCHEMA, completed_episode=100, metadata=dict(config_hash="fixture-config",
            execution_runtime_revision=config["checkpoint_runtime_revision"], architecture={}, loss={}, reward={}), prrac_training_state=dict(gamma=.95, tau=.005))
        learner = SimpleNamespace(policy_snapshot=lambda: ({}, {}, {}, {}), search_value_config={})
        with patch.object(evaluator, "load_prrac_checkpoint", return_value=(learner, payload)):
            jobs = runner.make_jobs(self.args, histories, anchors, analysis.ALPHAS)
        self.assertEqual(len(anchors), 15)
        self.assertEqual(len(jobs), 75)
        self.assertTrue(all(w["selection"]["branch_agents"] == [0, 1] for w in jobs))
        self.assertTrue(all(w["job"]["config"]["max_steps"] == 400 and w["job"]["checkpoint_info"]["evaluation_mode"] == "full_prrac" for w in jobs))
        self.assertEqual([e["episode_index"] for e in anchors], list(range(15)))

    def test_smoke_validates_all_then_selects_six_runs(self):
        self.args.max_per_transition = 1
        alphas = runner.alpha_grid(["0,0.5,1"], True)
        _, anchors, _, _, _ = runner.validate_inputs(self.args)
        self.assertEqual(len(anchors)*len(alphas), 6)
        self.assertEqual([e["episode_index"] for e in anchors], [0, 5])

    def test_parent_outputs_stably_sorted_and_analyzable_without_simulation(self):
        hashes = {str(p): runner.source_trace.sha(p) for root in (self.fixture.source, self.trace_root) for p in root.rglob("*") if p.is_file()}
        histories, anchors, _, _, _ = runner.validate_inputs(self.args)
        jobs = [dict(selection=e["selection"], alpha=a) for e in anchors for a in analysis.ALPHAS]
        def collect(work, workers, worker, on_result):
            for item in reversed(work):
                result = synthetic_result(item["alpha"])
                result.update(scenario_id=item["selection"]["scenario_id"], selection=item["selection"])
                for row in result["rows"]:
                    row.update(item["selection"])
                on_result(result)
        with patch.object(runner, "make_jobs", return_value=jobs), patch.object(runner.source_trace, "collect_jobs", side_effect=collect):
            self.assertEqual(runner.run(self.args), 0)
        rows = analysis.trace.read_csv(self.args.output_dir/"branch_sensitivity_step_trace.csv")
        keys = [(r["scenario_id"], r["alpha"], r["step"], r["agent_id"]) for r in rows]
        self.assertEqual(keys, sorted(keys))
        report = analysis.analyze(self.args.output_dir, self.args.output_dir/"analysis")
        self.assertEqual(report["branch_runs_expected"], 75)
        self.assertEqual(report["groups"]["help"]["scenario_count"], 5)
        self.assertEqual(report["groups"]["hurt"]["scenario_count"], 10)
        self.assertEqual(hashes, {str(p): runner.source_trace.sha(p) for root in (self.fixture.source, self.trace_root) for p in root.rglob("*") if p.is_file()})

    def test_checkpoint_hash_mismatch_rejected(self):
        self.manifest["checkpoint_sha256"] = "bad"
        evaluator._write_json(self.trace_root/"trace_manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            runner.validate_inputs(self.args)

    def test_seed_manifest_mismatch_rejected(self):
        self.manifest["selected"][0]["scenario_seed"] += 1
        evaluator._write_json(self.trace_root/"trace_manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "seeds"):
            runner.validate_inputs(self.args)

    def test_config_and_trace_bytes_mismatch_rejected(self):
        self.manifest["config_sha256"] = "bad"
        evaluator._write_json(self.trace_root/"trace_manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "config SHA"):
            runner.validate_inputs(self.args)

    def test_nonempty_and_historical_output_protected_before_load(self):
        original = {str(p): runner.source_trace.sha(p) for p in self.fixture.source.glob("*/*")}
        self.args.output_dir.mkdir()
        (self.args.output_dir/"retained").write_text("keep", encoding="utf-8")
        with patch.object(evaluator, "load_prrac_checkpoint", side_effect=AssertionError("no load")):
            with self.assertRaises(FileExistsError):
                runner.run(self.args)
            self.args.output_dir = self.fixture.source/"new"
            with self.assertRaisesRegex(ValueError, "protected"):
                runner.run(self.args)
        self.assertEqual(original, {str(p): runner.source_trace.sha(p) for p in self.fixture.source.glob("*/*")})

    def test_custom_formal_grid_and_missing_alpha1_rejected(self):
        for values, smoke in (([0, .5, 1], False), ([0, .5], True), ([0, 1, 1], True), ([0, "nan", 1], True)):
            with self.assertRaises(ValueError):
                runner.alpha_grid(values, smoke)


class BranchOfflineTests(unittest.TestCase):
    def test_all_offline_outputs_and_failed_manifest_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = [synthetic_result(a) for a in analysis.ALPHAS]
            rows, steps, _ = analysis.summarize_scenario(results, results[-1]["rows"], results[-1]["missions"])
            for filename, values in (("branch_sensitivity_scenario.csv", rows), ("branch_sensitivity_step_trace.csv", steps), ("selected_anchors.csv", [selection()])):
                runner.write_csv(root/filename, values)
            manifest = dict(schema=analysis.SCHEMA, status="completed", selected=[selection()], alphas=list(analysis.ALPHAS), horizon=10,
                smoke=True, training_update=False, explore=False, branch_runs_expected=5,
                output_sha256={name: runner.source_trace.sha(root/name) for name in ("branch_sensitivity_scenario.csv", "branch_sensitivity_step_trace.csv", "selected_anchors.csv")})
            evaluator._write_json(root/"branch_sensitivity_manifest.json", manifest)
            summary = analysis.analyze(root, root/"analysis")
            self.assertEqual(summary["groups"]["hurt"]["classification_counts"]["stable"], 1)
            self.assertEqual(len(list((root/"analysis").iterdir())), 3)
            with self.assertRaises(FileExistsError):
                analysis.analyze(root, root/"analysis")
            manifest["status"] = "failed"
            evaluator._write_json(root/"branch_sensitivity_manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "prohibited"):
                analysis.analyze(root, root/"other")


class BranchNativeTests(unittest.TestCase):
    def test_native_same_state_control_and_original_clip_path(self):
        from tests.search_value_audit_support import small_job
        class ActionAudit:
            def __init__(self):
                self.actions, self.physics = [], []
            def __getattr__(self, name):
                return lambda *args, **kwargs: None
            def bind(self, *, env, **kwargs):
                self.runtime = env.unwrapped
            def action(self, state, actions):
                self.actions.append(actions.detach().clone())
            def after_install(self, state, *args, **kwargs):
                if state.step:
                    runtime = self.runtime
                    self.physics.append(dict(prior=runtime._last_prior_acc.clone(), residual=runtime._last_residual_acc.clone(),
                        final=runtime._agent_acc.clone(), xy_max=runtime._a_xy_max.clone(), z_max=runtime._a_z_max.clone()))
        job = small_job(enabled=False, max_steps=3)
        selected = dict(selection(), scenario_id=job["scenario"]["scenario_id"])
        baseline = SearcherResidualTrace(job["scenario"], "full_prrac", selected["transition_type"])
        base_audit = ActionAudit()
        evaluator._evaluate_episode_job(job, searcher_trace=baseline, audit=base_audit)
        results = []
        # Three short synthetic branches, NOT the user's formal six-run smoke.
        for alpha in (0., .5, 1.):
            sink = ResidualBranchDiagnostic(job["scenario"], selected, alpha, horizon=1)
            # Avoid adding an uninventoried mock closure to the runtime fingerprint:
            # capture only via the existing public audit hook instead of monkeypatch.
            audit = ActionAudit()
            evaluator._evaluate_episode_job(job, searcher_trace=sink, branch_diagnostic=sink, audit=audit)
            captured = audit.actions
            exported = sink.export()
            self.assertTrue(exported["complete"])
            self.assertEqual(len(captured), 2)
            torch.testing.assert_close(audit.physics[0]["final"], base_audit.physics[0]["final"], rtol=0, atol=0)
            reference, physical = base_audit.physics[1], audit.physics[1]
            torch.testing.assert_close(physical["prior"], reference["prior"], rtol=0, atol=0)
            expected_residual = reference["residual"].clone()
            expected_residual[selected["branch_agents"]] *= alpha
            torch.testing.assert_close(physical["residual"], expected_residual, rtol=1e-6, atol=1e-8)
            # Test oracle only: production diagnostic never computes final actions.
            expected_final = physical["prior"] + physical["residual"]
            expected_final[:, :2] = expected_final[:, :2].clamp(-physical["xy_max"][:, None], physical["xy_max"][:, None])
            expected_final[:, 2] = expected_final[:, 2].clamp(-physical["z_max"], physical["z_max"])
            torch.testing.assert_close(physical["final"], expected_final, rtol=0, atol=0)
            torch.testing.assert_close(physical["final"][3], reference["final"][3], rtol=0, atol=0)
            for agent in range(3):
                old = next(r for r in baseline.action_rows if r["step"] == 0 and r["agent_id"] == agent)
                np.testing.assert_array_equal(captured[0][agent].numpy(), [old[f"applied_residual_{axis}"] for axis in "012"])
            for row in exported["rows"]:
                if row["sample_kind"] != "transition":
                    continue
                scale = alpha if row["agent_id"] in selected["branch_agents"] else 1.
                np.testing.assert_array_equal([row[f"applied_residual_{i}"] for i in "012"],
                    np.array([row[f"full_applied_residual_{i}"] for i in "012"], dtype=np.float32)*scale)
            results.append(exported)
        self.assertEqual(len({r["anchor_state_hash"] for r in results}), 1)
        errors = analysis.control_reproduction(results[-1], baseline.action_rows, baseline.mission_rows)
        self.assertEqual(errors, [])
        rows, _, check = analysis.summarize_scenario(results, baseline.action_rows, baseline.mission_rows, expected_alphas=(0., .5, 1.), horizon=1)
        self.assertTrue(check["interpretation_allowed"])
        self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()
