from __future__ import annotations

import copy
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from tests.prrac_evaluation_support import checkpoint_payload, worker_jobs, write_checkpoint


class _ImmediateExecutor:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def map(self, function, jobs):
        del function
        results = []
        for job in jobs:
            info = dict(job["checkpoint_info"])
            results.append(
                {
                    "episode": {
                        **info,
                        "scenario_id": str(job["scenario"]["scenario_id"]),
                        "scenario_seed": int(job["scenario"]["scenario_seed"]),
                        "found": True,
                        "contact_episode": True,
                        "hold_episode": False,
                        "success": False,
                        "collision_episode": False,
                        "post_found_collision_count": 0,
                        "executor_invalid_count": 0,
                        "executor_invalid_assignment_unreachable_count": 0,
                        "executor_min_distance_to_target": 1.0,
                        "executor_final_distance_to_target": 2.0,
                        "executor_replan_count": 0,
                        "executor_residual_ratio_post_found": 0.1,
                        "handoff_delay": 1,
                        "found_to_success_steps": None,
                        "failure_stage": "FOUND_NO_CONTACT",
                        "router_confusion_matrix": [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                        "gate_mean": 0.5,
                        "gate_p10": 0.4,
                        "gate_p90": 0.6,
                        "alignment_negative_rate": 0.0,
                    },
                    "failure_trace": [],
                    "trace_index": None,
                }
            )
        return results


class PRRACCheckpointEvaluatorTests(unittest.TestCase):
    def test_prrac_checkpoint_loads_into_fresh_learner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_checkpoint(Path(directory) / "checkpoint.pt")
            learner, payload = evaluator.load_prrac_checkpoint(path)

        self.assertEqual(payload["schema"], evaluator.CHECKPOINT_SCHEMA)
        self.assertEqual(len(learner.agents), 4)
        expected = payload["prrac_training_state"]["agents"]
        for agent, state in zip(learner.agents, expected):
            actual = agent.actor.state_dict()
            for key, value in state["actor"].items():
                torch.testing.assert_close(actual[key].cpu(), value.cpu())

    def test_old_checkpoint_schemas_are_explicitly_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for schema in (
                "bser.phase1c.training_state.v1",
                "bser.phase1c.training_state.v2",
            ):
                path = Path(directory) / f"{schema[-2:]}.pt"
                torch.save({"schema": schema}, path)
                with self.assertRaisesRegex(ValueError, "incompatible with PRRAC") as caught:
                    evaluator.load_prrac_checkpoint(path)
                self.assertEqual(str(caught.exception), evaluator.OLD_CHECKPOINT_MESSAGE)

    def test_dimension_drift_and_architecture_mismatch_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for field, value in (("observation_dim", 29), ("action_dim", 4), ("critic_dim", 125)):
                payload = checkpoint_payload()
                payload["metadata"][field] = value
                path = write_checkpoint(Path(directory) / f"{field}.pt", payload)
                with self.assertRaisesRegex(ValueError, field):
                    evaluator.load_prrac_checkpoint(path)
            payload = checkpoint_payload()
            payload["prrac_training_state"] = copy.deepcopy(payload["prrac_training_state"])
            payload["prrac_training_state"]["architecture"]["encoder_hidden_dim"] = 10
            path = write_checkpoint(Path(directory) / "architecture.pt", payload)
            with self.assertRaisesRegex(ValueError, "architecture mismatch"):
                evaluator.load_prrac_checkpoint(path)

    def test_evaluator_has_no_replay_or_update_calls(self) -> None:
        source = inspect.getsource(evaluator._evaluate_episode_job)
        for forbidden in (
            "learner.update(",
            "actor.update(",
            "update_critic_only(",
            "update_all_targets(",
            "optimizer.step(",
            "PRRACReplayAdapter(",
        ):
            self.assertNotIn(forbidden, source)

    def test_loaded_parameters_are_unchanged_by_episode_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_checkpoint(Path(directory) / "checkpoint.pt")
            learner, _ = evaluator.load_prrac_checkpoint(path)
            before = learner.policy_snapshot()
            result = evaluator._evaluate_episode_job(worker_jobs(path, 1)[0])
            after = learner.policy_snapshot()

        self.assertFalse(evaluator._contains_tensor(result))
        for left, right in zip(before, after):
            for key in left:
                np.testing.assert_array_equal(left[key], right[key])

    def test_incremental_outputs_and_resume_do_not_repeat_completed_combo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_checkpoint(root / "checkpoint.pt")
            output = root / "evaluation"
            with mock.patch.object(evaluator, "ProcessPoolExecutor", _ImmediateExecutor):
                summary = evaluator.run_evaluation(
                    checkpoints=[path],
                    output_dir=output,
                    episodes_override=1,
                    workers_override=1,
                )
                resumed = evaluator.run_evaluation(
                    checkpoints=[path],
                    output_dir=output,
                    episodes_override=1,
                    workers_override=1,
                    resume_evaluation=True,
                )

            for name in evaluator.OUTPUT_FILES:
                self.assertTrue((output / name).is_file(), name)
            self.assertTrue((output / "plots/router_balanced_accuracy_curve.png").is_file())
            self.assertTrue(summary["same_scenarios_for_all_checkpoints"])
            self.assertFalse(summary["explore"])
            self.assertFalse(summary["training_update"])
            self.assertIsNone(summary["performance_passed"])
            self.assertEqual(resumed["recommended_checkpoint"], summary["recommended_checkpoint"])

    def test_cuda_multi_worker_request_fails_fast(self) -> None:
        with mock.patch.object(evaluator.torch.cuda, "is_available", return_value=True):
            with self.assertRaisesRegex(ValueError, "workers=1"):
                evaluator.run_evaluation(device_override="cuda", workers_override=2)


if __name__ == "__main__":
    unittest.main()
