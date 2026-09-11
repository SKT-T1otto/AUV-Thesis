"""Controller checks and exactly one real evaluation episode, with no training.

Set PRIOR_ONLY_SMOKE_OUTPUT to a new directory to retain smoke artifacts.
The checkpoint is a fresh, untrained test fixture, never an experiment result.
"""

from contextlib import nullcontext
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.models.prrac.phase_routed_actor import PhaseRoutedResidualActor
from chapter3_bser.models.prrac.prrac_maddpg import PRRACMADDPG
from tests.prrac_evaluation_support import ARCHITECTURE, LOSS, write_checkpoint


PRIOR_CONFIG = evaluator.ROOT / "configs/chapter3/prior_only_eval.json"
ORIGINAL_EPISODE_JOB = evaluator._evaluate_episode_job


def _checked_episode_job(job):
    """Spawn-safe observer around the real worker; all physics calls delegate."""
    original_step = evaluator.PRRACTrainingEnv.step
    original_adapter = evaluator.ExecutionContinuityActionAdapter.apply
    steps, adapter_calls, terminal = 0, 0, False

    def checked_adapter(self, actions, **kwargs):
        nonlocal adapter_calls
        if tuple(actions.shape) != (4, 3) or torch.count_nonzero(actions).item():
            raise AssertionError("prior_only must pass (4,3) zero residuals to the adapter")
        adapter_calls += 1
        return original_adapter(self, actions, **kwargs)

    def checked_step(self, actions):
        nonlocal steps, terminal
        if tuple(actions.shape) != (4, 3) or torch.count_nonzero(actions).item():
            raise AssertionError("env.step must receive (4,3) zero residuals")
        result = original_step(self, actions)
        if torch.count_nonzero(self.unwrapped._last_residual_acc).item():
            raise AssertionError("physical residual acceleration must remain zero")
        steps += 1
        terminal = all(bool(value) for value in result[2])
        return result

    with (
        mock.patch.object(PhaseRoutedResidualActor, "forward", side_effect=AssertionError("actor forward is forbidden")) as forward,
        mock.patch.object(evaluator.ExecutionContinuityActionAdapter, "apply", checked_adapter),
        mock.patch.object(evaluator.PRRACTrainingEnv, "step", checked_step),
        mock.patch.object(evaluator, "_trace_step", wraps=evaluator._trace_step) as trace,
    ):
        result = ORIGINAL_EPISODE_JOB(job)
        forward.assert_not_called()
        if not terminal or steps < 1 or adapter_calls != steps:
            raise AssertionError("episode must terminate normally through the existing adapter")
        if trace.call_count != steps or any(call.kwargs["actor_outputs"] for call in trace.call_args_list):
            raise AssertionError("trace must accept absent actor outputs on every step")
    checks = {
        "controller_mode": "prior_only",
        "episodes": 1,
        "steps": steps,
        "action_shape": [4, 3],
        "max_abs_residual": 0.0,
        "physical_residual_zero": True,
        "actor_forward_calls": 0,
        "adapter_calls": adapter_calls,
        "episode_ended": terminal,
        "checkpoint_kind": "untrained_test_fixture",
    }
    Path(job["config"]["output_dir"], "smoke_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    return result


class PriorOnlyControllerTests(unittest.TestCase):
    def test_config_diff_is_only_controller(self):
        baseline = json.loads(evaluator.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        prior = json.loads(PRIOR_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(prior.pop("controller"), "prior_only")
        self.assertEqual(prior, baseline)
        self.assertEqual(evaluator._controller_mode(baseline), "full_prrac")

    def test_prior_skips_policy_and_preserves_adapter_input(self):
        observations = [torch.zeros(28) for _ in range(4)]
        with mock.patch.object(evaluator, "_policy_outputs", side_effect=AssertionError("no policy forward")):
            outputs, actions = evaluator._controller_actions(None, observations, torch.device("cpu"), "prior_only")
        self.assertEqual(outputs, [])
        self.assertEqual(tuple(actions.shape), (4, 3))
        self.assertEqual(actions.dtype, torch.float32)
        self.assertEqual(torch.count_nonzero(actions).item(), 0)

    def test_full_prrac_actions_match_original_path_exactly(self):
        with torch.random.fork_rng():
            torch.manual_seed(1729)
            learner = PRRACMADDPG(architecture=ARCHITECTURE, loss=LOSS)
            learner.prep_rollouts(torch.device("cpu"))
            observations = [torch.randn(28) for _ in range(4)]
            with torch.no_grad():
                original = evaluator._policy_outputs(learner, observations, torch.device("cpu"))
                expected = torch.stack([item.gated_residual_action.squeeze(0) for item in original])
                outputs, actual = evaluator._controller_actions(learner, observations, torch.device("cpu"), "full_prrac")
            self.assertTrue(torch.equal(actual, expected))
            for old, new in zip(original, outputs):
                for old_tensor, new_tensor in zip(old, new):
                    self.assertTrue(torch.equal(old_tensor, new_tensor))

    def test_unknown_controller_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported evaluation controller"):
            evaluator._controller_mode({"controller": "pvdrll"})

    def test_prior_cannot_enable_oracle_ablation(self):
        with self.assertRaisesRegex(ValueError, "prior_only requires"):
            evaluator.run_evaluation(config_path=PRIOR_CONFIG, modes_override=["oracle_current_target_diagnostic"])

    def test_summary_rejects_mixed_controller_rows(self):
        with mock.patch.object(evaluator, "derive_unique_provenance", return_value={}):
            with self.assertRaisesRegex(ValueError, "cannot mix controller"):
                evaluator._evaluation_summary(
                    checkpoint_paths=[], scenarios=[], modes=("full_prrac",), execution_variants=(),
                    summary_rows=[{"controller_mode": "prior_only"}, {"controller_mode": "full_prrac"}],
                    output=Path("unused"),
                )


class PriorOnlySmokeTests(unittest.TestCase):
    def test_one_episode_csv_and_summary(self):
        retained = os.environ.get("PRIOR_ONLY_SMOKE_OUTPUT")
        context = nullcontext(retained) if retained else tempfile.TemporaryDirectory()
        with context as directory:
            root = Path(directory).resolve()
            if retained:
                root.mkdir(parents=True, exist_ok=False)
            checkpoint = write_checkpoint(root / "untrained_test_fixture.pt")
            checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            output = root / "evaluation"
            with mock.patch.object(evaluator, "_evaluate_episode_job", _checked_episode_job):
                summary = evaluator.run_evaluation(
                    config_path=PRIOR_CONFIG,
                    checkpoints=[checkpoint],
                    output_dir=output,
                    episodes_override=1,
                    workers_override=1,
                )
            with (output / "episode_evaluation.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["controller_mode"], "prior_only")
            self.assertGreater(int(rows[0]["episode_length"]), 0)
            self.assertEqual(summary["controller_mode"], "prior_only")
            self.assertEqual(summary["scenario_count"], 1)
            self.assertEqual(summary["optimizer_update_count"], 0)
            self.assertEqual(summary["replay_sample_count"], 0)
            self.assertEqual(summary["parameter_update_count"], 0)
            saved = json.loads((output / "evaluation_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["controller_mode"], "prior_only")
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), checkpoint_hash)
            checks = json.loads((output / "smoke_checks.json").read_text(encoding="utf-8"))
            self.assertEqual(checks["steps"], int(rows[0]["episode_length"]))
            checks.update(csv_rows=1, csv_generated=True, checkpoint_unchanged=True,
                          success=rows[0]["success"], found=rows[0]["found"])
            (output / "smoke_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
            print(json.dumps({"smoke": checks, "artifacts": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
