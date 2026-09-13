"""Synthetic post-found and artifact identity checks; never starts simulation."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.experiments.phase1c_prrac import evaluation_provenance as provenance
from chapter3_bser.experiments.phase1c_prrac.diagnostics import PRRACDiagnostics
from chapter3_bser.experiments.phase1c_common.transition_schema import TransitionPhase
from chapter3_bser.experiments.phase1c_prrac.execution_continuity import ExecutionVariant, NavigationMode
from tests import test_s2_0_evaluation_provenance as protocol_fixtures
from tests import test_prrac_evaluation_information_boundary as guidance_fixtures
from tests import test_prrac_replay as transition_fixtures
from tests.execution_continuity_test_support import previous_plan
from tests.prrac_evaluation_support import _ImmediateExecutor
from tests.prrac_evaluation_support import write_checkpoint


class ControllerIdentityTests(unittest.TestCase):
    def test_combo_and_activation_identity_distinguish_controller(self):
        row, _, _, _ = protocol_fixtures.EvaluationProvenanceTests().valid_protocol()
        full = evaluator._combo_key(row, "manifest")
        prior = evaluator._combo_key({**row, "controller_mode": "prior_only"}, "manifest")
        self.assertNotEqual(full, prior)
        self.assertEqual(full["controller_mode"], "full_prrac")
        activation = {"checkpoint": "c.pt", "scenario_id": "s", "step": 1}
        self.assertEqual(len(evaluator._dedupe_activation_rows([activation, {**activation, "controller_mode": "prior_only"}])), 2)

    def test_controller_is_derived_and_blank_is_rejected(self):
        self.assertEqual(provenance.derive_unique_provenance([{}])["controller_mode"], "full_prrac")
        value = provenance.derive_unique_provenance([{}, {"controller_mode": "prior_only"}])
        self.assertIsNone(value["controller_mode"])
        self.assertEqual(value["controller_mode_values"], ["full_prrac", "prior_only"])
        for invalid in (None, "", "unknown"):
            with self.assertRaises(ValueError):
                provenance.row_controller_mode({"controller_mode": invalid})

    def test_resume_rejects_controller_even_with_identical_hashes(self):
        with self.assertRaisesRegex(ValueError, "controller_mode"):
            provenance.validate_resume_config({"controller": "prior_only"}, {"controller": "full_prrac"})
        provenance.validate_resume_config({}, {"controller": "full_prrac"})

    def test_cached_progress_and_rows_are_checked_before_reuse(self):
        config = {"controller": "prior_only"}
        row = {"controller_mode": "prior_only"}
        progress = {**row, "completed": [row]}
        provenance.validate_controller_artifacts(config, progress, [row])
        for bad_progress, bad_rows in (({}, [row]), (progress, [{}]), ({**progress, "completed": [{}]}, [row])):
            with self.assertRaisesRegex(ValueError, "controller_mode"):
                provenance.validate_controller_artifacts(config, bad_progress, bad_rows)
        # Historical absent field is specifically the full_prrac controller.
        provenance.validate_controller_artifacts({}, {"completed": [{}]}, [{}])

    def test_cross_artifact_validation_rejects_wrong_controller(self):
        row, config, progress, metadata = protocol_fixtures.EvaluationProvenanceTests().valid_protocol()
        row["controller_mode"] = "prior_only"
        with self.assertRaisesRegex(ValueError, "controller_mode"):
            provenance.validate_evaluation_provenance(rows=[row], resolved_config=config, progress=progress,
                checkpoint_metadata=metadata, expected_scenarios=[{"scenario_id": "scenario", "scenario_seed": 7}])
        with self.assertRaises(ValueError):
            provenance.validate_summary_provenance([row], [{**row, "controller_mode": "full_prrac"}],
                [{"scenario_id": "scenario", "scenario_seed": 7}])

    def test_legacy_completed_key_normalizes_to_current_full_key(self):
        row, _, _, _ = protocol_fixtures.EvaluationProvenanceTests().valid_protocol()
        key = evaluator._combo_key(row, "manifest")
        old = {name: value for name, value in key.items() if name != "controller_mode"}
        normalized = {**old, "controller_mode": provenance.row_controller_mode(old)}
        self.assertEqual(evaluator._canonical_json(normalized), evaluator._canonical_json(key))

    def test_mixed_summaries_fail_before_writes(self):
        rows = [{"controller_mode": mode} for mode in ("full_prrac", "prior_only")]
        for aggregate in (evaluator._failure_funnel, evaluator._paired_rows):
            with self.assertRaisesRegex(ValueError, "cannot mix controller"):
                aggregate(rows)
        with self.assertRaisesRegex(ValueError, "cannot mix controller"):
            evaluator.aggregate_checkpoint(rows, {})
        with mock.patch.object(evaluator, "_write_csv") as writer:
            with self.assertRaisesRegex(ValueError, "cannot mix controller"):
                evaluator._write_outputs(Path("unused"), checkpoint_paths=[], scenarios=[], modes=(),
                    execution_variants=(), episode_rows=rows, summary_rows=[], trace_rows=[], trace_index=[], progress={})
            writer.assert_not_called()

    def test_exports_keep_controller_and_accept_legacy_full_rows_without_simulation(self):
        class SyntheticExecutor(_ImmediateExecutor):
            def map(self, function, jobs):
                results = super().map(function, jobs)
                for result in results:
                    if result["episode"]["controller_mode"] == "full_prrac":
                        del result["episode"]["controller_mode"]
                return results

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = write_checkpoint(root / "synthetic_fixture.pt")
            with mock.patch.object(evaluator, "ProcessPoolExecutor", SyntheticExecutor), \
                 mock.patch.object(evaluator, "_evaluate_episode_job", side_effect=AssertionError("no simulation")), \
                 mock.patch.object(evaluator, "_plot"), mock.patch.object(evaluator, "_plot_execution_variants"), \
                 mock.patch.object(evaluator, "_plot_search_recovery"):
                for mode in ("full_prrac", "prior_only"):
                    output = root / mode
                    summary = evaluator.run_evaluation(checkpoints=[checkpoint], output_dir=output,
                        episodes_override=1, workers_override=1, controller_override=mode)
                    self.assertEqual(summary["controller_mode"], mode)
                    for name in ("episode_evaluation.csv", "checkpoint_summary.csv", "execution_variant_summary.csv",
                                 "search_continuity_summary.csv", "search_collision_recovery_summary.csv"):
                        rows = evaluator._read_csv(output / name)
                        self.assertTrue(rows, name)
                        self.assertEqual({row["controller_mode"] for row in rows}, {mode}, name)


class PostFoundFixtureTests(unittest.TestCase):
    def trace(self, phase, prior=None, residual=None):
        metadata = transition_fixtures._metadata(0, before=TransitionPhase.POST_FOUND, after=phase)
        guidance = guidance_fixtures._guidance()
        runtime = SimpleNamespace(_agent_pos=torch.zeros(4, 3), _nav_targets=torch.ones(4, 3),
            _last_prior_acc=torch.ones(4, 3) if prior is None else prior,
            _last_residual_acc=torch.zeros(4, 3) if residual is None else residual,
            _agent_acc=torch.ones(4, 3), _collision_flags=torch.zeros(4, dtype=torch.bool))
        env = SimpleNamespace(unwrapped=runtime, get_target_state=lambda: SimpleNamespace(position=(1, 2, 3)))
        return evaluator._trace_step(info={"checkpoint_episode": 1, "controller_mode": "prior_only"},
            scenario={"scenario_id": "synthetic", "scenario_seed": 1}, step=1,
            task=SimpleNamespace(target_found=True, executor_knows_target=True, mission_complete=phase == TransitionPhase.SUCCESS),
            metadata=metadata, result=SimpleNamespace(events=()), controller=SimpleNamespace(),
            public_guidance=guidance, installed_guidance=guidance, env=env, actor_outputs=[],
            raw_actions=torch.zeros(4, 3), applied_actions=torch.zeros(4, 3))

    def test_intercept_hold_success_traces_without_actor(self):
        for phase in (TransitionPhase.POST_FOUND, TransitionPhase.CONTACT, TransitionPhase.HOLD, TransitionPhase.SUCCESS):
            with self.subTest(phase=phase.name):
                trace = self.trace(phase)
                self.assertEqual(trace["residual_action_norm"], 0)
                self.assertIsNone(trace["trust_gate"])
                self.assertIsNone(trace["router_prediction"])
                self.assertIsNone(trace["alignment_cosine"])
                self.assertEqual(trace["stage_after"], 1 if phase == TransitionPhase.POST_FOUND else 2)
        self.assertIsNone(PRRACDiagnostics().summary()["gate_mean"])

    def test_zero_residual_survives_original_post_found_adapters(self):
        hold = replace(previous_plan(), variant=ExecutionVariant.B3_PROXY_SAFE_SUPPRESSION,
                       navigation_mode=NavigationMode.SAFE_HOLD, safe_hold=True, reachable=False)
        for stage in (1, 2):
            for variant in ExecutionVariant:
                raw = torch.zeros(4, 3)
                mode_actions = evaluator._apply_residual_mode(raw, "full_prrac", stage)
                actions, diagnostic = evaluator.ExecutionContinuityActionAdapter().apply(
                    mode_actions, plan=hold, variant=variant, mission_phase="EXECUTION", executor_id=3)
                self.assertEqual(tuple(actions.shape), (4, 3))
                self.assertTrue(torch.equal(actions, raw))
                self.assertEqual(diagnostic.raw_norm, 0)
                self.assertEqual(diagnostic.applied_norm, 0)
                self.assertEqual(diagnostic.suppressed, variant == ExecutionVariant.B3_PROXY_SAFE_SUPPRESSION)

    def test_historical_final_norm_remains_pre_clip_physical_sum(self):
        prior, residual = torch.zeros(4, 3), torch.zeros(4, 3)
        prior[3, 0], residual[3, 0] = 4, 3
        trace = self.trace(TransitionPhase.HOLD, prior, residual)
        self.assertEqual(trace["navigation_prior_norm"], 4)
        self.assertEqual(trace["final_action_norm"], 7)
        self.assertEqual(trace["final_action_norm_semantics"], "pre_clip_prior_plus_scaled_residual_acceleration")
        self.assertNotEqual(trace["final_action_norm"], float(torch.ones(3).norm()))


if __name__ == "__main__":
    unittest.main()
