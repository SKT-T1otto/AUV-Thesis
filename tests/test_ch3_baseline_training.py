"""Bounded CPU mechanism tests for the independent B2/B3 learning pipelines.

One four-step synthetic episode per baseline exercises real replay, gradients
and checkpoints. Temporary artifacts are not thesis performance evidence.
"""
import ast
import copy
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from core.algorithms.maddpg import MADDPG
from core.algorithms.networks import MLPNetwork
from core.replay.ch3_buffer import CH3ReplayBuffer
from chapter3_bser.experiments.baselines.common import checkpoint, model as models, train as common_train
from chapter3_bser.experiments.baselines.direct_mc.train import B2Trainer, DirectMCTrainer
from chapter3_bser.experiments.baselines.direct_boundary.train import B3Trainer, DirectBoundaryTrainer
from tools.ch3_baselines import learned_evaluation, registry, run_baseline, run_training
from tools.ch3_baselines.bser_prior import ZeroResidualSource
from tools.ch3_baselines.provenance import ROOT, file_sha256


def independent_calls_only():
    stack = ExitStack()
    for target in (
        "chapter3_bser.experiments.hgr.train.Trainer.__init__",
        "chapter3_bser.experiments.hgr.runtime.MissionRuntime.__init__",
        "chapter3_bser.models.hgr.policy.HandoffPolicy.actions",
        "chapter3_bser.models.hgr.estimator.BoundaryPredictor.forward",
    ):
        stack.enter_context(patch(target, side_effect=AssertionError("independent baseline called " + target)))
    return stack


def changed(before, after):
    return any(not torch.equal(before[name], value) for name, value in after.items())


def tiny_config(baseline):
    name = "direct_mc_train.json" if baseline == "B2_direct_mc" else "direct_boundary_train.json"
    config = json.loads((ROOT / "configs/chapter3/baselines" / name).read_text(encoding="utf-8"))
    config.update(total_main_trajectories=1, checkpoint_interval=1, max_steps=4)
    config["rl"].update(batch_size=2, replay_size=32, warmup_steps=0,
                        update_frequency=1, updates_per_train=1, hidden_dim=16)
    return config


class IndependentTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous_threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory(prefix="ch3-baseline-real-training-")
        cls.addClassCleanup(temporary.cleanup)
        cls.directory = Path(temporary.name)
        cls.trained = {}
        scene = json.loads((ROOT / "tests/fixtures/hgr/handoff_manifest.json").read_text(encoding="utf-8"))["scenarios"][0]
        scene["max_steps"] = 4
        scene["scenario_id"] = "independent_baseline_four_step_mechanism_only"
        for baseline, trainer_type in (("B2_direct_mc", B2Trainer), ("B3_direct_boundary", B3Trainer)):
            initial = {}
            def build(*args, **kwargs):
                result = models.build_model(*args, **kwargs)
                initial["state"] = copy.deepcopy(result.training_state_dict())
                return result
            output = cls.directory / "collision_terminal" / baseline
            with independent_calls_only(), patch.object(common_train, "build_model", side_effect=build):
                trainer = trainer_type(tiny_config(baseline), output)
                trainer.scenarios = [copy.deepcopy(scene)]
                result = trainer.run()
            cls.trained[baseline] = dict(trainer=trainer, summary=result, initial=initial["state"],
                                          checkpoint=Path(result["latest_checkpoint"]))

    def test_b2_has_independent_trainer(self):
        self.assertIs(B2Trainer, DirectMCTrainer)
        record = self.trained["B2_direct_mc"]
        trainer, summary = record["trainer"], record["summary"]
        self.assertIs(type(trainer.model), MADDPG)
        self.assertIs(type(trainer.replay), CH3ReplayBuffer)
        self.assertEqual(summary["completed_main_trajectories"], 1)
        self.assertEqual(summary["actual_total_environment_steps"], 4)
        self.assertEqual(summary["replay_transition_count"], 4)
        self.assertGreater(summary["optimizer_updates"], 0)
        self.assertFalse(summary["formal_experiments_completed"])
        self.assertFalse(summary["performance_claims_supported"])
        for agent, initial in zip(trainer.model.agents, record["initial"]["agent_params"]):
            self.assertIs(type(agent.policy), MLPNetwork)
            self.assertEqual((agent.policy.fc1.in_features, agent.policy.fc3.out_features,
                              agent.critic1.fc1.in_features), (28, 3, 124))
            self.assertTrue(changed(initial["policy"], agent.policy.state_dict()))
            self.assertTrue(changed(initial["critic1"], agent.critic1.state_dict()))
            self.assertTrue(changed(initial["critic2"], agent.critic2.state_dict()))
            self.assertTrue(agent.policy_optimizer.state)
            self.assertTrue(agent.critic1_optimizer.state)
            self.assertTrue(agent.critic2_optimizer.state)
        for path in (ROOT / "chapter3_bser/experiments/baselines").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
            imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
            self.assertFalse(any(name and (name.startswith("chapter3_bser.models.hgr") or
                                            name.startswith("chapter3_bser.experiments.hgr")) for name in imports), path)

    def test_b3_has_boundary_encoder(self):
        self.assertIs(B3Trainer, DirectBoundaryTrainer)
        record = self.trained["B3_direct_boundary"]
        model = record["trainer"].model
        self.assertIs(type(model), models.BoundaryConditionedMADDPG)
        self.assertIs(type(model).update, MADDPG.update)
        self.assertIs(type(model).update_all_targets, MADDPG.update_all_targets)
        for agent, initial in zip(model.agents, record["initial"]["agent_params"]):
            self.assertIsInstance(agent.policy.boundary_encoder, models.BoundaryEncoder)
            self.assertEqual((agent.policy.actor.fc1.in_features, agent.policy.actor.fc3.out_features,
                              agent.critic1.fc1.in_features), (28, 3, 124))
            encoder_before = {k: v for k, v in initial["policy"].items() if k.startswith("boundary_encoder.")}
            encoder_after = {k: v for k, v in agent.policy.state_dict().items() if k.startswith("boundary_encoder.")}
            self.assertTrue(changed(encoder_before, encoder_after))
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                                for p in agent.policy.boundary_encoder.parameters()))
            self.assertTrue(changed(initial["target_policy"], agent.target_policy.state_dict()))

    def test_boundary_features_condition_current_and_target_actions(self):
        torch.manual_seed(2729)
        model = models.build_model(tiny_config("B3_direct_boundary"))
        observations = torch.zeros(4, 28)
        observations[:, 22:26] = torch.eye(4)
        observations[:, :3] = torch.arange(12).reshape(4, 3) / 20
        observations[:, 12:15] = 0.9  # Unknown targets must be masked in boundary features.
        unknown = models.boundary_features(observations)
        self.assertEqual(tuple(unknown.shape), (4, 18))
        self.assertEqual(torch.count_nonzero(unknown[:, 8:11]).item(), 0)
        discovered = observations.clone()
        discovered[0, 27] = 1
        found = models.boundary_features(discovered)
        self.assertTrue(torch.equal(found[:, 0], torch.ones(4)))
        self.assertTrue(torch.equal(found[:, 1], torch.zeros(4)))
        self.assertEqual(torch.count_nonzero(found[3, 8:11]).item(), 0)
        handoff = discovered.clone()
        handoff[3, 27] = 1
        delivered = models.boundary_features(handoff)
        self.assertTrue(torch.equal(delivered[:, 1], torch.ones(4)))
        self.assertTrue(torch.equal(delivered[3, 8:11], handoff[3, 12:15]))
        model.prep_rollouts(device="cpu")
        before = model.step(list(observations), explore=False)[1]
        after = model.step(list(handoff), explore=False)[1]
        self.assertTrue(torch.equal(observations[1], handoff[1]))
        self.assertGreater(float((before - after).abs().max()), 1e-8)

        current = observations.repeat(2, 1, 1)
        next_values = handoff.repeat(2, 1, 1)
        sample = ([current[:, i] for i in range(4)], [torch.zeros(2, 3) for _ in range(4)],
                  [torch.full((2,), 0.25) for _ in range(4)], [next_values[:, i] for i in range(4)],
                  [torch.zeros(2) for _ in range(4)], torch.ones(2), torch.arange(2), torch.zeros(2, dtype=torch.bool))
        observed = {}
        hooks = []
        for index, actor in enumerate(model.target_policies):
            def target_inputs(module, values, index=index):
                observed[index] = values[1].detach().clone()
            hooks.append(actor.boundary_encoder.register_forward_pre_hook(target_inputs))
        try:
            model.prep_training(device="cpu")
            model.update(sample, 0)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(set(observed), set(range(4)))
        expected = models.boundary_features(next_values)
        for index in range(4):
            torch.testing.assert_close(observed[index], expected[:, index])
        current_expected = models.boundary_features(current)
        for index, actor in enumerate(model.policies):
            torch.testing.assert_close(actor._boundary_context, current_expected[:, index])

    def test_b2_checkpoint_identity(self):
        for baseline, record in self.trained.items():
            with self.subTest(baseline=baseline):
                saved = checkpoint.load_checkpoint(record["checkpoint"], expected_baseline=baseline)
                self.assertEqual(saved["baseline"], baseline)
                self.assertEqual(saved["episode_count"], 1)
                self.assertEqual(saved["config_hash"], checkpoint.digest(saved["config"]))
                self.assertEqual(saved["source_identity"], checkpoint.fresh_source_identity())
                restored = checkpoint.load_model(saved)
                self.assertEqual(checkpoint.state_digest(restored.training_state_dict()), saved["model_state_sha256"])
                observations = record["trainer"].replay.obs_buffs
                values = [ob[0].numpy() for ob in observations]
                original_actions = record["trainer"].model.step(values, explore=False)
                restored_actions = restored.step(values, explore=False)
                for expected, actual in zip(original_actions, restored_actions):
                    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
                other = "B3_direct_boundary" if baseline == "B2_direct_mc" else "B2_direct_mc"
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    checkpoint.load_checkpoint(record["checkpoint"], expected_baseline=other)
                altered = copy.deepcopy(saved)
                altered["baseline"] = other
                path = self.directory / (baseline + "_relabelled.pt")
                torch.save(altered, path)
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    checkpoint.load_checkpoint(path, expected_baseline=baseline)
                altered = copy.deepcopy(saved)
                altered["model_training_state"]["agent_params"][0]["policy_opt"]["state"].clear()
                altered["model_state_sha256"] = checkpoint.state_digest(altered["model_training_state"])
                torch.save(altered, path)
                with self.assertRaisesRegex(ValueError, "optimizer parameter/state inventory"):
                    checkpoint.load_checkpoint(path, expected_baseline=baseline)
        hgr = self.directory / "hgr_identity_only.pt"
        torch.save({"schema": "hgr.complete_cycle.v1", "config": {"method": "ch3_hgr"}}, hgr)
        for baseline in self.trained:
            with self.assertRaisesRegex(ValueError, "an HGR checkpoint cannot be relabeled"):
                checkpoint.load_checkpoint(hgr, expected_baseline=baseline)

    def test_real_checkpoints_evaluate_without_optimizer_updates(self):
        """Exercise the real learned evaluator and source gate with saved weights."""
        fixture = json.loads((ROOT / "tests/fixtures/hgr/handoff_manifest.json").read_text(encoding="utf-8"))
        fixture["purpose"] = "four-step checkpoint/evaluation mechanism test only; no performance evidence"
        scene = fixture["scenarios"][0]
        scene.update(scenario_id="independent_baseline_checkpoint_evaluation_mechanism_only",
                     scenario_split="validation", scenario_role="validation", max_steps=4)
        manifest = self.directory / "independent_validation_manifest.json"
        manifest.write_text(json.dumps(fixture), encoding="utf-8")
        manifest_hash = file_sha256(manifest)
        for baseline, record in self.trained.items():
            with self.subTest(baseline=baseline), independent_calls_only(), ExitStack() as stack:
                for owner, name in ((torch.optim.Adam, "step"), (common_train.BaselineTrainer, "__init__"),
                                    (CH3ReplayBuffer, "__init__"), (CH3ReplayBuffer, "push"), (CH3ReplayBuffer, "sample")):
                    stack.enter_context(patch.object(owner, name, side_effect=AssertionError("evaluation called " + name)))
                checkpoint_hash = file_sha256(record["checkpoint"])
                trainer_state = checkpoint.state_digest(record["trainer"].model.training_state_dict())
                output = self.directory / "collision_terminal" / (baseline + "_positive_evaluation")
                result = learned_evaluation.evaluate(record["checkpoint"], output, episodes=1,
                    seed=12729, policy_mode="deterministic", manifest=manifest)
                self.assertTrue(result["evaluation_complete"])
                self.assertEqual((result["n_valid_episodes"], result["n_expected_episodes"]), (1, 1))
                self.assertEqual(result["optimizer_update_count"], 0)
                self.assertFalse(result["training_update"])
                self.assertEqual(result["evaluation_policy_mode"], "deterministic")
                self.assertEqual(result["baseline"], baseline)
                rows = json.loads((output / "episodes.json").read_text(encoding="utf-8"))
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertEqual(row["scenario_id"], scene["scenario_id"])
                self.assertEqual(row["method"], record["trainer"].config["method"])
                self.assertEqual(row["baseline"], baseline)
                self.assertEqual(row["optimizer_update_count"], 0)
                self.assertFalse(row["training_update"])
                self.assertIn(row["termination_reason"], ("success", "obstacle_collision", "timeout"))
                self.assertTrue(0 < row["episode_length"] <= 4)
                self.assertEqual(result["actual_environment_steps"], row["episode_length"])
                self.assertEqual(row["task_protocol"], "collision_terminal_v1")
                self.assertEqual(row["reward_objective"], "team_mean_v1")
                saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(saved, result)
                progress = json.loads((output / "evaluation_progress.json").read_text(encoding="utf-8"))
                self.assertEqual(progress["status"], "complete")
                self.assertTrue(progress["evaluation_complete"])
                self.assertEqual(file_sha256(record["checkpoint"]), checkpoint_hash)
                self.assertEqual(file_sha256(manifest), manifest_hash)
                self.assertEqual(checkpoint.state_digest(record["trainer"].model.training_state_dict()), trainer_state)


class TrainingEntryTests(unittest.TestCase):
    """Check-only, evaluation rejection and priors never construct a trainer."""

    def test_evaluation_rejects_hgr_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="ch3-baseline-hgr-rejection-") as temporary:
            directory = Path(temporary)
            hgr = directory / "hgr_evaluation_identity_only.pt"
            torch.save({"schema": "hgr.complete_cycle.v1", "config": {"method": "ch3_hgr"}}, hgr)
            for baseline in ("B2_direct_mc", "B3_direct_boundary"):
                with self.assertRaisesRegex(ValueError, "an HGR checkpoint cannot be relabeled"):
                    run_baseline.evaluation_plan(baseline, ROOT / "tests/fixtures/ch3_final/smoke_manifest.json",
                        directory / "collision_terminal" / (baseline + "_invalid_eval"), episodes=1, checkpoint=hgr)

    def test_training_check_only(self):
        with tempfile.TemporaryDirectory(prefix="ch3-baseline-no-training-") as temporary, ExitStack() as stack:
            for owner, name in ((run_training, "DirectMCTrainer"), (run_training, "DirectBoundaryTrainer"),
                                (models, "build_model"), (torch.nn.Module, "__init__"),
                                (torch, "save"), (torch.optim.Adam, "step")):
                stack.enter_context(patch.object(owner, name, side_effect=AssertionError("check-only called " + name)))
            before = torch.get_rng_state().clone()
            for baseline in ("B2_direct_mc", "B3_direct_boundary"):
                output = Path(temporary) / "collision_terminal" / baseline
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(run_training.main(["--baseline", baseline, "--episodes", "7", "--seed", "99",
                                                        "--output-dir", str(output), "--check-only"]), 0)
                plan = run_training.train(baseline, output_dir=output, check_only=True, episodes=7, seed=99)
                self.assertFalse(plan["training_started"])
                self.assertEqual((plan["config"]["total_main_trajectories"], plan["config"]["seed"]), (7, 99))
                self.assertFalse(output.exists())
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_b0_b1_unchanged(self):
        historical = json.loads((ROOT / "docs/chapter3/baselines/framework_protected_baseline.json").read_text(encoding="utf-8"))
        for relative in ("tools/ch3_baselines/basic_search_prior.py", "tools/ch3_baselines/bser_prior.py",
                         "tools/ch3_baselines/evaluate.py", "configs/chapter3/baselines/search_prior_eval.json",
                         "configs/chapter3/baselines/bser_prior_eval.json"):
            self.assertEqual(file_sha256(ROOT / relative), historical["files"][relative], relative)
        for baseline in ("B0_search_prior", "B1_bser_prior"):
            spec = registry.method_spec(baseline)
            self.assertFalse(spec["learning"])
            self.assertEqual(spec["residual_source"], "zeros_4x3")
            with self.assertRaisesRegex(ValueError, "no trainer"):
                run_training.train(baseline, check_only=True)
        generator = torch.Generator().manual_seed(12729)
        before = generator.get_state().clone()
        source = ZeroResidualSource()
        actions, latents = source.actions([torch.zeros(28)] * 4, suffix=False, active=[True] * 4,
                                          generator=generator, deterministic=False)
        self.assertTrue(torch.equal(actions, torch.zeros(4, 3)))
        self.assertEqual(latents, [None] * 4)
        self.assertTrue(torch.equal(before, generator.get_state()))


if __name__ == "__main__":
    unittest.main()
