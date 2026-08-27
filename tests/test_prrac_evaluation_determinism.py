from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from tests.prrac_evaluation_support import worker_jobs, write_checkpoint


class PRRACEvaluationDeterminismTests(unittest.TestCase):
    def test_same_checkpoint_and_scenario_produce_identical_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_checkpoint(Path(directory) / "checkpoint.pt")
            job = worker_jobs(path, 1)[0]
            first = evaluator._evaluate_episode_job(job)["episode"]
            second = evaluator._evaluate_episode_job(job)["episode"]

        for field in (
            "scenario_id",
            "scenario_seed",
            "success",
            "found",
            "contact_episode",
            "hold_episode",
            "reward",
            "router_confusion_matrix",
            "router_accuracy",
            "gate_mean",
            "alignment_mean",
            "executor_invalid_count",
            "executor_min_distance_to_target",
            "failure_stage",
        ):
            self.assertEqual(first[field], second[field], field)
        self.assertFalse(first["explore"])
        self.assertFalse(first["training_update"])
        self.assertEqual(first["optimizer_update_count"], 0)
        self.assertEqual(first["replay_sample_count"], 0)
        self.assertEqual(first["parameter_update_count"], 0)


if __name__ == "__main__":
    unittest.main()
