"""Independent validation only: no implementation edits or real training.

B0SmokeTests and B1SmokeTests each collect exactly one complete synthetic
episode per class. All runtime outputs are temporary and cleaned after tests.
Output-schema assertions check direct fields and their resolved-config linkage.
"""
from collections import Counter
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from chapter3_bser import candidate_generator, objective
from chapter3_bser.online import allocator as joint_module
from chapter3_bser.online.allocator import BSEROnlineAllocator
from chapter3_bser.experiments.phase1c_prrac.training_env import PRRACTrainingEnv
from tools.ch3_baselines import basic_search_prior as basic, registry, run_baseline, run_training
from tools.ch3_baselines.bser_prior import BSERPriorRuntime
from tools.ch3_baselines.framework_provenance import framework_sources, verify_framework_sources
from tools.ch3_baselines.provenance import ROOT, digest, file_sha256, write_json
from tests.bser_online_test_utils import state_at
from tests.test_ch3_basic_search_prior import fixture


def prohibit_learning():
    stack = ExitStack()
    for name in (
        "torch.nn.Module._call_impl", "chapter3_bser.models.hgr.policy.HandoffPolicy.actions",
        "chapter3_bser.models.hgr.estimator.BoundaryPredictor.forward",
        "torch.distributions.Normal.sample", "torch.distributions.Normal.rsample",
        "torch.optim.SGD.step", "torch.optim.Adam.step",
        "tools.ch3_baselines.run_training.DirectMCTrainer",
        "tools.ch3_baselines.run_training.DirectBoundaryTrainer",
    ):
        stack.enter_context(patch(name, side_effect=AssertionError("Forbidden learning call: " + name)))
    return stack


class ContractValidationTests(unittest.TestCase):
    def test_registry_module_and_exact_four_contracts(self):
        process = subprocess.run([sys.executable, "-B", "-m", "tools.ch3_baselines.registry"],
                                 cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(process.returncode, 0, process.stderr)
        # The module has no CLI; explicitly invoke its validating loader here.
        entries = registry.load_registry()
        expected = {
            "B0_search_prior": ("ch3_baseline_search_prior", "search_only", None),
            "B1_bser_prior": ("ch3_baseline_bser_prior", "bser_joint", None),
            "B2_direct_mc": ("ch3_baseline_direct_mc", "bser_joint", "maddpg"),
            "B3_direct_boundary": ("ch3_baseline_direct_boundary", "bser_joint", "direct_boundary_maddpg"),
        }
        self.assertEqual(set(entries), set(expected))
        for key, (method, planner, algorithm) in expected.items():
            with self.subTest(baseline=key):
                entry = entries[key]
                self.assertEqual((entry["method"], entry["planner"], entry["algorithm"]), (method, planner, algorithm))
                self.assertIs(entry["hgr"], False)
                self.assertIs(entry["learning"], algorithm is not None)
                self.assertIs(entry["training_required"], algorithm is not None)
                if algorithm is None:
                    self.assertEqual((entry["learning_mode"], entry["controller"], entry["residual_source"]),
                                     ("none", "prior_only", "zeros_4x3"))

    def test_direct_training_configs_preserve_the_common_task(self):
        reference = registry.load_reference()[1]
        expected = dict(profile="M20_MOVING_UNKNOWN_MULTI", max_steps=400,
            task_protocol="collision_terminal_v1", collision_detection_revision="segment_closed_aabb_v1",
            terminal_reward_revision="team_failure_override_v1", collision_terminal_reward=-2.0,
            reward_objective="team_mean_v1", gamma=0.95, execution_runtime_revision="dynamic_public_intercept_v2_1")
        for key, algorithm in (("B2_direct_mc", "maddpg"), ("B3_direct_boundary", "direct_boundary_maddpg")):
            with self.subTest(baseline=key):
                config = registry.training_config(key)[2]
                self.assertEqual(config["algorithm"], algorithm)
                self.assertEqual(config["method"], registry.method_spec(key)["runtime_method"])
                conditions = registry.task_conditions(config)
                self.assertEqual({k: conditions[k] for k in expected}, expected)
                self.assertEqual(conditions, registry.task_conditions(reference))
                for hgr_only in ("policy", "predictor", "prefix_lr", "suffix_lr", "correction_draws_per_cycle"):
                    self.assertNotIn(hgr_only, config)
                self.assertEqual(config["rl"]["residual_action_reg"], 0.0)
                self.assertNotIn("policy_delay", config["rl"])
                for field in ("gamma", "tau", "lr_actor", "lr_critic", "hidden_dim", "batch_size", "replay_size"):
                    self.assertEqual(config["rl"][field], reference["rl"][field])
                self.assertEqual((config["seed"], config["total_main_trajectories"], config["checkpoint_interval"]),
                                 (2729, 1000, 100))
                summary = run_baseline.unified_summary([], registry.method_spec(key), 1,
                    finalized=False, wall_seconds=0.0, actual_steps=None)
                self.assertIn("residual_action_max_abs", summary)
                self.assertIsNone(summary["residual_action_max_abs"])

    def test_entry_help_and_training_rejection_for_priors(self):
        for module, flags in (
            ("run_training", ("--baseline", "B2_direct_mc", "B3_direct_boundary")),
            ("run_baseline", ("--baseline", "--manifest", "--episodes", "--seed", "--output-dir")),
        ):
            result = subprocess.run([sys.executable, "-B", "-m", "tools.ch3_baselines." + module, "--help"],
                                    cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            for flag in flags:
                self.assertIn(flag, result.stdout)
            if module == "run_training":
                self.assertNotIn("B0_search_prior", result.stdout)
                self.assertNotIn("B1_bser_prior", result.stdout)
        with prohibit_learning():
            for baseline in ("B0_search_prior", "B1_bser_prior"):
                with self.subTest(baseline=baseline), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        run_training.main(["--baseline", baseline, "--check-only"])
                    self.assertEqual(error.exception.code, 2)
                    with self.assertRaisesRegex(ValueError, "no trainer"):
                        run_training.train(baseline, check_only=True)

    def test_check_only_never_constructs_trainer_or_writes_outputs(self):
        with tempfile.TemporaryDirectory(prefix="ch3-baseline-validation-plan-") as temporary, prohibit_learning():
            for baseline in ("B2_direct_mc", "B3_direct_boundary"):
                output = Path(temporary)/"collision_terminal"/baseline
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(run_training.main(["--baseline", baseline, "--output-dir", str(output), "--check-only"]), 0)
                self.assertFalse(output.exists())

    def test_same_planning_state_has_distinct_objective_and_standby_paths(self):
        state = state_at(0)
        real_objective, real_context = objective.evaluate_objective, objective.build_objective_context
        records = {}
        for baseline, allocator in (
            ("B0", basic.SearchPriorAllocator(state.agents[state.executor_id].position)),
            ("B1", BSEROnlineAllocator()),
        ):
            calls, flags, contexts = Counter(), [], []
            class ObservedWeights(dict):
                def __getitem__(self, key):
                    calls["response_weight_reads"] += 1
                    if baseline == "B0":
                        raise AssertionError("B0 read Executor response weights")
                    return super().__getitem__(key)
            def context(*args, **kwargs):
                self.assertIs(args[0], state)
                value = real_context(*args, **kwargs)
                value = replace(value, response_weight_by_id=ObservedWeights(value.response_weight_by_id))
                contexts.append(value)
                return value
            def evaluate(*args, **kwargs):
                flags.append(kwargs.get("search_only", False))
                return real_objective(*args, **kwargs)
            with prohibit_learning(), patch.object(objective, "evaluate_objective", new=evaluate), \
                 patch.object(basic, "evaluate_objective", new=evaluate), \
                 patch.object(basic, "build_objective_context", new=context), \
                 patch.object(joint_module, "build_objective_context", new=context), \
                 patch.object(basic, "solve_search_only_greedy", wraps=basic.solve_search_only_greedy) as search, \
                 patch.object(joint_module, "solve_joint_greedy", wraps=joint_module.solve_joint_greedy) as joint, \
                 patch.object(candidate_generator, "generate_standby_candidates", wraps=candidate_generator.generate_standby_candidates) as standby:
                allocation = allocator.allocate(state)
            self.assertTrue(contexts)
            self.assertTrue(flags)
            if baseline == "B0":
                self.assertGreater(search.call_count, 0)
                self.assertEqual(joint.call_count, 0)
                self.assertEqual(standby.call_count, 0)
                self.assertTrue(all(flags))
                self.assertEqual(calls["response_weight_reads"], 0)
            else:
                self.assertIs(type(allocator), BSEROnlineAllocator)
                self.assertEqual(search.call_count, 0)
                self.assertGreater(joint.call_count, 0)
                self.assertGreater(standby.call_count, 0)
                self.assertFalse(any(flags))
                self.assertGreater(calls["response_weight_reads"], 0)
                generated_positions = {tuple(float(x) for x in item.waypoint) for item in contexts[0].standby_candidates}
                self.assertIn(allocation.executor_assignment.target_region, generated_positions)
            records[baseline] = dict(objective=allocation.objective_value, search_only_calls=search.call_count,
                joint_calls=joint.call_count, standby_generation_calls=standby.call_count,
                objective_search_only_flags=sorted(set(flags)), response_weight_reads=calls["response_weight_reads"])
        # No assertion that either method must have a larger numeric objective.
        print("VALIDATION_PLANNING_PATHS " + json.dumps(records), flush=True)

    def test_frozen_production_and_baseline_source_gates(self):
        before = framework_sources()
        frozen = json.loads((ROOT / "docs/chapter3/baselines/final_production_source.json").read_text(encoding="utf-8"))["production"]
        self.assertEqual(len(frozen["files"]), 196)
        self.assertTrue(set(frozen["files"]).issubset(before["production"]["files"]))
        self.assertEqual(before["production_source_sha256"], "3bf6035001e28efbe5e5cd3db7b43bc43a11e0bf83ed2037a7ac29c5c518b4bd")
        verify_framework_sources(before)


class _SmokeValidationMixin:
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        temporary = tempfile.TemporaryDirectory(prefix="ch3-baseline-validation-smoke-")
        cls.addClassCleanup(temporary.cleanup)
        cls.directory = Path(temporary.name)
        cls.output = cls.directory/"collision_terminal"/cls.baseline
        manifest = cls.directory/"synthetic_interface_manifest.json"
        frozen_smoke = json.loads((ROOT / "tests/fixtures/ch3_final/smoke_manifest.json").read_text(encoding="utf-8"))
        if frozen_smoke != fixture():
            raise AssertionError("CH3-final smoke metadata must retain the existing synthetic fixture")
        write_json(manifest, frozen_smoke)
        before = framework_sources()
        runtime_type = basic.BasicSearchPriorRuntime if cls.baseline == "B0_search_prior" else BSERPriorRuntime
        expected_allocator = basic.SearchPriorAllocator if cls.baseline == "B0_search_prior" else BSEROnlineAllocator
        cls.observed = Counter()
        original_step = PRRACTrainingEnv.step
        original_advance = runtime_type.advance
        original_search, original_joint = basic.solve_search_only_greedy, joint_module.solve_joint_greedy
        def search(*args, **kwargs):
            cls.observed["search_only_solver_calls"] += 1
            if cls.baseline != "B0_search_prior":
                raise AssertionError("B1 called the B0 search-only solver")
            return original_search(*args, **kwargs)
        def joint(*args, **kwargs):
            cls.observed["joint_solver_calls"] += 1
            if cls.baseline != "B1_bser_prior":
                raise AssertionError("B0 called the joint BSER solver")
            return original_joint(*args, **kwargs)
        def advance(runtime):
            if type(runtime.controller.allocator) is not expected_allocator:
                raise AssertionError("Incorrect actual runtime allocator")
            return original_advance(runtime)
        def step(env, actions):
            commands = torch.as_tensor(actions)
            if commands.shape != (4, 3) or not torch.isfinite(commands).all() or torch.count_nonzero(commands):
                raise AssertionError("Nonzero/invalid actual residual command")
            if not env.unwrapped.use_residual_prior or env.unwrapped.max_steps != 400:
                raise AssertionError("Prior/horizon contract changed")
            result = original_step(env, actions)
            physical = env.unwrapped._last_residual_acc
            if physical.shape != (4, 3) or not torch.isfinite(physical).all() or torch.count_nonzero(physical):
                raise AssertionError("Nonzero/invalid physical residual acceleration")
            cls.observed["physical_steps_checked"] += 1
            cls.observed["prior_max_abs"] = max(cls.observed["prior_max_abs"], float(env.unwrapped._last_prior_acc.abs().max()))
            if cls.observed["physical_steps_checked"] % 100 == 0:
                print(f"VALIDATION_SMOKE_PROGRESS {cls.baseline} steps={cls.observed['physical_steps_checked']}", flush=True)
            return result
        with prohibit_learning(), patch.object(basic, "solve_search_only_greedy", new=search), \
             patch.object(joint_module, "solve_joint_greedy", new=joint), \
             patch.object(PRRACTrainingEnv, "step", new=step), \
             patch.object(runtime_type, "advance", new=advance):
            run_baseline.main(["--baseline", cls.baseline, "--manifest", str(manifest), "--episodes", "1", "--seed", "12729", "--output-dir", str(cls.output)])
        verify_framework_sources(before)
        cls.documents = {name: json.loads((cls.output/(name+".json")).read_text(encoding="utf-8")) for name in
            ("summary", "episodes", "identity", "resolved_config", "evaluation_manifest", "controller_diagnostics")}
        cls.manifest_hash = file_sha256(manifest)
        row, summary = cls.documents["episodes"][0], cls.documents["summary"]
        print("VALIDATION_SMOKE_RESULT " + json.dumps(dict(baseline=cls.baseline,
            task_outcome=row["termination_reason"], steps=row["episode_length"], found_step=row.get("found_step"),
            handoff_decision_step=row.get("handoff_decision_step"), program_failure=(cls.output/"evaluation_failure.json").exists(),
            summary_complete=summary["evaluation_complete"], observations=dict(cls.observed),
            selected_content_sha256=cls.documents["identity"]["selected_content_sha256"],
            comparable_inputs_sha256=cls.documents["identity"]["comparable_inputs_sha256"],
            summary_residual_scalar_present="residual_action_max_abs" in summary,
            identity_direct_reference_present="reference_config_path" in cls.documents["identity"],
            identity_direct_task_protocol_present="task_protocol" in cls.documents["identity"])), flush=True)

    def test_runtime_no_learning_and_zero_residual(self):
        row = self.documents["episodes"][0]
        diagnostics = self.documents["controller_diagnostics"]["episodes"][0]
        self.assertEqual(self.observed["physical_steps_checked"], row["episode_length"])
        self.assertTrue(0 < row["episode_length"] <= 400)
        self.assertGreater(self.observed["prior_max_abs"], 0)
        for field in ("actor_forward_calls", "action_sampling_calls", "optimizer_update_count", "residual_action_max_abs", "physical_residual_acceleration_max_abs"):
            self.assertEqual(diagnostics[field], 0, field)
        self.assertEqual(diagnostics["residual_steps_checked"], row["episode_length"])
        solver = "search_only_solver_calls" if self.baseline == "B0_search_prior" else "joint_solver_calls"
        self.assertGreater(self.observed[solver], 0)
        self.assertIn(row["termination_reason"], ("success", "timeout", "collision"))
        self.assertFalse((self.output/"evaluation_failure.json").exists())

    def test_required_summary_and_artifacts(self):
        summary = self.documents["summary"]
        spec = registry.method_spec(self.baseline)
        required = {"method", "planner_mode", "learning_mode", "n_success", "safe_success_rate",
                    "collision_failure_rate", "timeout_failure_rate", "training_update", "optimizer_update_count"}
        self.assertTrue(required <= summary.keys(), str(required-summary.keys()))
        self.assertEqual(summary["method"], spec["method"])
        self.assertEqual(summary["planner_mode"], spec["planner"])
        self.assertEqual(summary["learning_mode"], "none")
        self.assertIs(summary["training_update"], False)
        self.assertEqual(summary["optimizer_update_count"], 0)
        self.assertTrue(summary["evaluation_complete"])
        self.assertEqual(len(self.documents["episodes"]), 1)
        for filename in ("identity.json", "summary.json", "episodes.json", "resolved_config.json", "evaluation_manifest.json"):
            self.assertTrue((self.output/filename).is_file())
        self.assertFalse(list(self.directory.rglob("*.pt")))

    def test_provenance_is_verifiable_through_linked_resolved_config(self):
        identity, resolved = self.documents["identity"], self.documents["resolved_config"]
        self.assertEqual(identity["method"], registry.method_spec(self.baseline)["method"])
        self.assertEqual(identity["manifest_file_sha256"], self.manifest_hash)
        self.assertEqual(identity["resolved_config_sha256"], digest(resolved))
        self.assertEqual(resolved["common_task_conditions"]["task_protocol"], "collision_terminal_v1")
        self.assertEqual(resolved["reference_config_sha256"], file_sha256(resolved["reference_config_path"]))
        self.assertEqual(identity["sources_before"], identity["sources_after"])

    def test_summary_contains_requested_residual_scalar(self):
        self.assertEqual(self.documents["summary"].get("residual_action_max_abs"), 0.0,
                         "summary.json must report the prior-only zero residual contract")

    def test_identity_directly_records_requested_reference_and_protocol(self):
        identity, resolved = self.documents["identity"], self.documents["resolved_config"]
        self.assertEqual(identity["reference_config_path"], resolved["reference_config_path"])
        for field, expected in (("task_protocol", "collision_terminal_v1"), ("reward_objective", "team_mean_v1")):
            self.assertEqual(identity[field], expected)
            self.assertEqual(identity[field], resolved["common_task_conditions"][field])


class B0SmokeTests(_SmokeValidationMixin, unittest.TestCase):
    baseline = "B0_search_prior"


class B1SmokeTests(_SmokeValidationMixin, unittest.TestCase):
    baseline = "B1_bser_prior"


if __name__ == "__main__":
    unittest.main()
