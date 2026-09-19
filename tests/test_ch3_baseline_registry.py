"""Framework contracts and bounded prior runtime integration; never training.

Learned-route tests use explicitly mocked payloads/evaluators in temporary
directories. They do not certify real B2/B3 checkpoints or performance.
"""
import copy
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from chapter3_bser.experiments.hgr import train as native_train
from chapter3_bser.experiments.hgr.provenance import fresh_source_identity
from chapter3_bser.online import allocator as allocator_module
from chapter3_bser.online.allocator import BSEROnlineAllocator
from tools.ch3_baselines import registry as reg, run_baseline as run, run_training as training
from tools.ch3_baselines.basic_search_prior import BasicSearchPriorRuntime
from tools.ch3_baselines.bser_prior import BSERPriorRuntime, ZeroResidualSource
from tools.ch3_baselines.framework_provenance import framework_sources, verify_framework_sources
from tools.ch3_baselines.provenance import ROOT, digest, write_json, file_sha256
from tests.test_ch3_basic_search_prior import fixture


def forbid_learning():
    stack = ExitStack()
    for name in ("torch.nn.Module._call_impl", "chapter3_bser.models.hgr.policy.HandoffPolicy.actions",
                 "torch.distributions.Normal.sample", "torch.distributions.Normal.rsample",
                 "torch.optim.SGD.step", "torch.optim.Adam.step",
                 "chapter3_bser.models.hgr.estimator.BoundaryPredictor.forward"):
        stack.enter_context(patch(name, side_effect=AssertionError("forbidden learning call: " + name)))
    return stack


def native_row(config, method):
    # Synthetic outcome for routing/output tests only; no environment run.
    scene = fixture()["scenarios"][0]
    row = {k: config[k] for k in ("task_protocol", "collision_detection_revision", "terminal_reward_revision",
                                 "reward_objective", "source_reward_revision")}
    row.update(method=method, scenario_id=scene["scenario_id"], scenario_seed=scene["scenario_seed"],
               termination_reason="timeout", success=False, safe_success=False, found=False,
               collision_episode=False, terminated=True, truncated=False,
               first_collision_agent_ids=[], first_collision_phase=None, episode_length=400,
               optimizer_update_count=0, team_discounted_return=1.0, team_undiscounted_return=2.0,
               wall_seconds=0.01, gamma=0.95)
    return row


class RegistryTests(unittest.TestCase):
    def test_B0_B1_planning_and_zero_residual_contracts(self):
        values = reg.load_registry()
        self.assertEqual(values["B0_search_prior"]["method"], "ch3_baseline_search_prior")
        self.assertEqual(values["B0_search_prior"]["runtime_method"], "ch3_basic_search_prior_v1")
        self.assertEqual(values["B0_search_prior"]["planner"], "search_only")
        self.assertEqual(values["B1_bser_prior"]["planner"], "bser_joint")
        for key in ("B0_search_prior", "B1_bser_prior"):
            self.assertFalse(values[key]["learning"])
            self.assertFalse(values[key]["training_required"])
            self.assertEqual(values[key]["residual_source"], "zeros_4x3")
            with self.assertRaises(ValueError):
                reg.training_config(key)

    def test_B2_B3_are_native_independent_methods_with_only_declared_config_changes(self):
        _, reference = reg.load_reference()
        conditions = reg.task_conditions(reference)
        for baseline, algorithm in (("B2_direct_mc", "stochastic_direct_mc"), ("B3_direct_boundary", "direct_boundary_corrected")):
            spec, _, config = reg.training_config(baseline)
            self.assertEqual(config["algorithm"], algorithm)
            self.assertEqual(config["method"], "ch3_" + algorithm)
            self.assertEqual(spec["method"], reg.CONTRACTS[baseline][0])
            self.assertEqual({k for k in config.keys() | reference.keys() if config.get(k) != reference.get(k)}, {"method", "algorithm", "output_dir"})
            self.assertEqual(reg.task_conditions(config), conditions)
            self.assertEqual(config["policy"], reference["policy"])
            self.assertEqual(config["rl"], reference["rl"])
            self.assertTrue(spec["training_required"])
            self.assertFalse(spec["hgr"])
            self.assertEqual(config["ablation"], "none")

    def test_four_methods_cannot_change_collision_protocol_or_horizon(self):
        _, reference = reg.load_reference()
        for baseline, spec in reg.load_registry().items():
            config = reg.training_config(baseline)[2] if spec["learning"] else reference
            for key, value in (("task_protocol", "legacy_nonterminal_v1"), ("collision_terminal_reward", -1.0),
                               ("collision_detection_revision", "endpoint_rollback_v1"),
                               ("terminal_reward_revision", "legacy_shaping_v1"), ("max_steps", 401),
                               ("reward_objective", "individual_v1")):
                with self.subTest(baseline=baseline, field=key), self.assertRaises(ValueError):
                    reg.task_conditions({**config, key: value})

    def test_registry_tampering_is_rejected(self):
        values = reg.load_registry()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"registry.json"
            values["B1_bser_prior"]["planner"] = "search_only"
            write_json(path, values)
            with patch.object(reg, "REGISTRY_PATH", path), self.assertRaises(ValueError):
                reg.load_registry()

    def test_actual_reference_training_plan_keeps_all_reference_parameters(self):
        _, reference = reg.load_reference()
        reference["policy"]["initial_std"] = 0.15  # hypothetical actual run, not repo config
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/"outputs/hgr/config.json"
            path.parent.mkdir(parents=True)
            write_json(path, reference)
            with patch.object(training, "Trainer", side_effect=AssertionError("training must not start")):
                for baseline in ("B2_direct_mc", "B3_direct_boundary"):
                    plan = training.train(baseline, reference_training_config=path,
                                          output_dir=root/"collision_terminal"/baseline, check_only=True)
                    self.assertFalse(plan["training_started"])
                    self.assertEqual(plan["reference_kind"], "run_config_reference")
                    self.assertEqual(plan["config"]["policy"], reference["policy"])
                    self.assertFalse(Path(plan["output_dir"]).exists())

    def test_production_B3_label_does_not_request_old_suffix(self):
        # Exercise only the existing label-selection function on an inert stub;
        # no Trainer construction, simulator continuation or gradient update.
        from types import SimpleNamespace
        for method, expected_calls in (("hgr", 2), ("direct_boundary_corrected", 1)):
            calls = []
            def branch(trajectory, policy, purpose):
                calls.append(purpose)
                return {"G_plus": 5.0 if "new" in purpose else 2.0}
            owner = SimpleNamespace(config={"algorithm": method, "max_steps": 400}, branch=branch, cycle=0, branches=[])
            trajectory = dict(dataset_id="synthetic", snapshot=SimpleNamespace(sha256="synthetic"), tau=2, features=np.zeros(3))
            label, row = native_train.Trainer.paired_label(owner, trajectory, object(), object(), "unit")
            self.assertEqual(len(calls), expected_calls)
            if method == "direct_boundary_corrected":
                self.assertEqual(label, 5.0)
                self.assertIsNone(row["old"])
                self.assertIsNone(row["delta_hat"])

    def test_B0_protected_sources_and_framework_script_inventory(self):
        before = framework_sources()
        verify_framework_sources(before)
        self.assertEqual(len(before["production"]["files"]), 196)
        self.assertIn("scripts/linux/train_ch3_direct_mc.sh", before["framework_entry_scripts"]["files"])
        altered = copy.deepcopy(before)
        altered["framework_entry_scripts"]["sha256"] = "changed"
        with self.assertRaises(ValueError):
            verify_framework_sources(altered)

    def test_training_sidecar_does_not_claim_budget_stop_is_completion(self):
        # An inert stand-in returns counters only: no trainer, model or update.
        from types import SimpleNamespace
        requested = reg.training_config("B2_direct_mc")[2]["total_main_trajectories"]
        with tempfile.TemporaryDirectory() as temporary:
            for completed in (1, requested):
                output = Path(temporary)/"collision_terminal"/str(completed)
                def fake_trainer(config, destination):
                    destination.mkdir(parents=True)
                    return SimpleNamespace(run=lambda: {"completed_main_trajectories": completed})
                with patch.object(training, "Trainer", side_effect=fake_trainer):
                    training.train("B2_direct_mc", output_dir=output)
                identity = json.loads((output/"baseline_training_identity.json").read_text())
                self.assertEqual(identity["training_complete"], completed == requested)
                self.assertEqual(identity["completed_main_trajectories"], completed)


class EvaluationRoutingTests(unittest.TestCase):
    def test_same_manifest_and_task_identity_for_four_baselines(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root/"synthetic_manifest.json"
            write_json(manifest, fixture())
            checkpoint = root/"mock_payload.bin"
            checkpoint.write_bytes(b"unit mock only, not a usable checkpoint")
            identities = []
            for baseline, spec in reg.load_registry().items():
                config = reg.training_config(baseline)[2] if spec["learning"] else reg.load_reference()[1]
                payload = dict(config=config, source_identity=fresh_source_identity(), config_hash=native_train.digest(config), completed_main=1, cycle=1)
                with patch.object(run, "load_checkpoint", return_value=payload), forbid_learning():
                    plan = run.evaluation_plan(baseline, manifest, root/"collision_terminal"/baseline, episodes=1,
                                               checkpoint=checkpoint if spec["learning"] else None)
                identities.append(plan["resolved"]["comparable_inputs_sha256"])
                self.assertEqual(plan["selected"]["scenarios"], fixture()["scenarios"])
                self.assertFalse(plan["output"].exists())
                self.assertEqual(plan["resolved"]["optimizer_update_count"], 0)
            self.assertEqual(len(set(identities)), 1)

    def test_checkpoint_is_required_and_cannot_be_relabelled_from_HGR_or_other_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root/"scenes.json"
            write_json(manifest, fixture())
            _, reference = reg.load_reference()
            for baseline in ("B2_direct_mc", "B3_direct_boundary"):
                with self.assertRaisesRegex(ValueError, "independently trained"):
                    run.evaluation_plan(baseline, manifest, root/"collision_terminal"/baseline, episodes=1)
                for wrong_config in (reference, reg.training_config("B3_direct_boundary" if baseline == "B2_direct_mc" else "B2_direct_mc")[2]):
                    with patch.object(run, "load_checkpoint", return_value={"config": wrong_config}), self.assertRaisesRegex(ValueError, "another method"):
                        run.evaluation_plan(baseline, manifest, root/"collision_terminal"/baseline, episodes=1, checkpoint=root/"mock.bin")
            for baseline in ("B0_search_prior", "B1_bser_prior"):
                with self.assertRaisesRegex(ValueError, "do not accept"):
                    run.evaluation_plan(baseline, manifest, root/"collision_terminal"/baseline, episodes=1, checkpoint="unused")

    def test_native_learned_evaluator_dispatch_and_uniform_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root/"synthetic_manifest.json"
            write_json(manifest, fixture())
            checkpoint = root/"unit_mock.bin"
            checkpoint.write_bytes(b"unit mocked loader input; never loaded as weights")
            for baseline in ("B2_direct_mc", "B3_direct_boundary"):
                spec, _, config = reg.training_config(baseline)
                payload = dict(config=config, source_identity=fresh_source_identity(), config_hash=native_train.digest(config), completed_main=1, cycle=1)
                def evaluate_native(checkpoint, output, **kwargs):
                    self.assertEqual(kwargs["manifest"], manifest)
                    self.assertEqual(kwargs["policy_mode"], "stochastic")
                    output.mkdir(parents=True)
                    write_json(output/"episodes.json", [native_row(config, spec["runtime_method"])])
                    write_json(output/"summary.json", {"evaluation_complete": True})
                output = root/"collision_terminal"/baseline
                with patch.object(run, "load_checkpoint", return_value=payload), patch.object(run.native_evaluation, "evaluate", side_effect=evaluate_native) as native, patch.object(training, "Trainer", side_effect=AssertionError("no training")):
                    summary = run.evaluate(baseline, manifest, output, episodes=1, checkpoint=checkpoint)
                native.assert_called_once()
                self.assertTrue(summary["evaluation_complete"])
                self.assertEqual(summary["method"], spec["method"])
                self.assertEqual(summary["runtime_method"], spec["runtime_method"])
                self.assertEqual(summary["optimizer_update_count"], 0)
                for filename in ("resolved_config.json", "evaluation_manifest.json", "summary.json", "episodes.json", "identity.json"):
                    self.assertTrue((output/filename).is_file())
                self.assertNotIn("pairing_sha256", json.loads((output/"identity.json").read_text()))

    def test_partial_summary_rates_stay_null_and_updates_cannot_be_hidden(self):
        spec = reg.method_spec("B2_direct_mc")
        config = reg.training_config("B2_direct_mc")[2]
        row = run.normalize_row(native_row(config, spec["runtime_method"]), spec)
        summary = run.unified_summary([row], spec, 2, finalized=False, wall_seconds=1, actual_steps=400)
        self.assertIsNone(summary["safe_success_rate"])
        self.assertIsNone(summary["mean_team_discounted_return"])
        self.assertEqual(summary["optimizer_update_count"], 0)
        with self.assertRaises(ValueError):
            run.normalize_row({**native_row(config, spec["runtime_method"]), "optimizer_update_count": 1}, spec)

    def test_native_failure_keeps_original_error_and_unknown_interrupted_cost(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, checkpoint = root/"synthetic_manifest.json", root/"mock.bin"
            write_json(manifest, fixture())
            checkpoint.write_bytes(b"mock input only")
            spec, _, config = reg.training_config("B2_direct_mc")
            payload = dict(config=config, source_identity=fresh_source_identity(), config_hash=native_train.digest(config), completed_main=1, cycle=1)
            for corrupt in (False, True):
                output = root/"collision_terminal"/str(corrupt)
                def failed_native(checkpoint, output, **kwargs):
                    output.mkdir(parents=True)
                    row = native_row(config, spec["runtime_method"])
                    if corrupt:
                        row["optimizer_update_count"] = 1
                    write_json(output/"episodes.json", [row])
                    raise RuntimeError("original native failure")
                with patch.object(run, "load_checkpoint", return_value=payload), patch.object(run.native_evaluation, "evaluate", side_effect=failed_native):
                    with self.assertRaisesRegex(RuntimeError, "original native failure"):
                        run.evaluate("B2_direct_mc", manifest, output, episodes=1, checkpoint=checkpoint)
                failure = json.loads((output/"evaluation_failure.json").read_text())
                summary = json.loads((output/"summary.json").read_text())
                self.assertEqual(failure["message"], "original native failure")
                self.assertEqual("native_harvest_error" in failure, corrupt)
                self.assertFalse(summary["evaluation_complete"])
                self.assertIsNone(summary["safe_success_rate"])
                self.assertIsNone(summary["actual_environment_steps"])
                self.assertEqual(summary["completed_episode_environment_steps"], 0 if corrupt else 400)

    def test_output_and_manifest_guards_run_before_training_or_runtime_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            occupied = root/"collision_terminal/occupied"
            occupied.mkdir(parents=True)
            (occupied/"keep").write_text("unchanged")
            with patch.object(run, "BSERPriorRuntime", side_effect=AssertionError("unexpected runtime")):
                with self.assertRaises(FileExistsError):
                    run.evaluate("B1_bser_prior", root/"missing.json", occupied, episodes=1)
                with self.assertRaises(FileNotFoundError):
                    run.evaluate("B1_bser_prior", root/"missing.json", root/"collision_terminal/new", episodes=1)
            self.assertEqual((occupied/"keep").read_text(), "unchanged")


class PriorRuntimeIntegrationTests(unittest.TestCase):
    def test_B1_full_task_through_unified_evaluator_without_learning(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root/"synthetic_interface_only.json"
            output = root/"collision_terminal/B1_full_task_interface_only"
            write_json(manifest, fixture())
            with forbid_learning():
                result = run.evaluate("B1_bser_prior", manifest, output, episodes=1, seed=12729)
            rows = json.loads((output/"episodes.json").read_text())
            diagnostics = json.loads((output/"controller_diagnostics.json").read_text())["episodes"]
            self.assertTrue(result["evaluation_complete"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(result["method"], "ch3_baseline_bser_prior")
            self.assertEqual(result["actual_environment_steps"], rows[0]["episode_length"])
            self.assertTrue(0 < result["actual_environment_steps"] <= 400)
            self.assertEqual(result["optimizer_update_count"], 0)
            self.assertEqual(diagnostics[0]["residual_steps_checked"], rows[0]["episode_length"])
            self.assertEqual(diagnostics[0]["residual_action_max_abs"], 0.0)
            self.assertEqual(diagnostics[0]["physical_residual_acceleration_max_abs"], 0.0)
            self.assertGreater(diagnostics[0]["physical_prior_acceleration_max_abs"], 0.0)
            self.assertEqual(diagnostics[0]["allocator_class"], "chapter3_bser.online.allocator.BSEROnlineAllocator")
            for filename in ("resolved_config.json", "evaluation_manifest.json", "summary.json", "episodes.json", "identity.json"):
                self.assertTrue((output/filename).is_file())
            print("B1 synthetic full-task interface evidence: " + json.dumps({k: rows[0].get(k) for k in (
                "episode_length", "termination_reason", "found_step", "handoff_event_step", "handoff_decision_step",
                "executor_target_received_step", "optimizer_update_count")}), flush=True)

    def test_B0_and_B1_same_task_two_steps_zero_residual_B1_uses_joint_BSER(self):
        torch.set_num_threads(1)
        _, config = reg.load_reference()
        scene = fixture()["scenarios"][0]
        before = framework_sources()
        initial = []
        for runtime_class in (BasicSearchPriorRuntime, BSERPriorRuntime):
            with forbid_learning(), patch.object(allocator_module, "solve_joint_greedy", wraps=allocator_module.solve_joint_greedy) as joint:
                runtime = runtime_class(config, scene, seed=12729)
                try:
                    initial.append((runtime.state.target_belief.probabilities.copy(), runtime.env.get_agent_state().positions))
                    self.assertEqual(runtime.env.unwrapped.max_steps, 400)
                    self.assertEqual(runtime.env.unwrapped.task_protocol, "collision_terminal_v1")
                    self.assertTrue(runtime.env.unwrapped.use_residual_prior)
                    if runtime_class is BSERPriorRuntime:
                        self.assertIs(type(runtime.controller.allocator), BSEROnlineAllocator)
                        self.assertGreater(joint.call_count, 0)
                        self.assertTrue(runtime.env.unwrapped.pse_use_standby)
                    else:
                        self.assertEqual(joint.call_count, 0)
                    for _ in range(2):
                        runtime.advance()
                        self.assertEqual(tuple(runtime.env.unwrapped._last_residual_acc.shape), (4, 3))
                        self.assertEqual(float(runtime.env.unwrapped._last_residual_acc.abs().max()), 0.0)
                    diagnostic = runtime.controller_diagnostics()
                    for name in ("actor_forward_calls", "action_sampling_calls", "optimizer_update_count", "residual_action_max_abs", "physical_residual_acceleration_max_abs"):
                        self.assertEqual(diagnostic[name], 0)
                    self.assertEqual(diagnostic["residual_steps_checked"], 2)
                    self.assertGreater(diagnostic["physical_prior_acceleration_max_abs"], 0.0)
                    self.assertFalse(runtime.terminal)  # bounded smoke, never a completed-evaluation claim
                finally:
                    runtime.close()
        np.testing.assert_array_equal(initial[0][0], initial[1][0])
        self.assertEqual(initial[0][1], initial[1][1])
        verify_framework_sources(before)

    def test_zero_source_does_not_consume_policy_generator(self):
        source = ZeroResidualSource()
        generator = torch.Generator().manual_seed(12729)
        before = generator.get_state().clone()
        actions, latents = source.actions([np.zeros(28)]*4, suffix=False, active=[True]*4, generator=generator, deterministic=False)
        self.assertTrue(torch.equal(before, generator.get_state()))
        self.assertEqual(actions.shape, (4, 3))
        self.assertEqual(actions.abs().max().item(), 0)
        self.assertEqual(latents, [None]*4)


class EntryTests(unittest.TestCase):
    def test_help_and_training_check_only_never_construct_trainer(self):
        with patch.object(training, "Trainer", side_effect=AssertionError("no trainer in help/check-only")):
            for baseline in ("B2_direct_mc", "B3_direct_boundary"):
                self.assertEqual(training.main(["--baseline", baseline, "--check-only"]), 0)
            with self.assertRaises(SystemExit) as raised:
                run.main(["--help"])
            self.assertEqual(raised.exception.code, 0)

    def test_windows_scripts_help_and_error_exit_from_other_directory(self):
        if os.name != "nt":
            self.skipTest("Windows launcher test")
        environment = os.environ.copy()
        environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment["PATH"]
        with tempfile.TemporaryDirectory(prefix="baseline framework space ") as temporary:
            for name in ("run_ch3_baseline_eval", "train_ch3_direct_mc", "train_ch3_direct_boundary"):
                for arg, code in (("--help", 0), ("--unknown-option", 2)):
                    result = subprocess.run('"' + str(ROOT/"scripts"/(name+".bat")) + '" ' + arg,
                        cwd=temporary, env=environment, shell=True, capture_output=True, text=True, timeout=120)
                    self.assertEqual(result.returncode, code, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
