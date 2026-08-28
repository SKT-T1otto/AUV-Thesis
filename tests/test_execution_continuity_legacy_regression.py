import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.experiments.phase1c_prrac.execution_continuity import ExecutionVariant
from chapter3_bser.online.types import ExecutorAssignment, OnlineAllocation
from tests.prrac_evaluation_support import worker_jobs, write_checkpoint


class ExecutionContinuityLegacyRegressionTests(unittest.TestCase):
    def test_b0_worker_is_exactly_the_default_legacy_path(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = write_checkpoint(Path(directory) / "checkpoint.pt")
            default_job = worker_jobs(checkpoint, 1)[0]
            explicit_job = dict(default_job)
            explicit_job["checkpoint_info"] = dict(default_job["checkpoint_info"])
            explicit_job["checkpoint_info"]["execution_variant"] = ExecutionVariant.B0_LEGACY_V2_1.value
            first = evaluator._evaluate_episode_job(default_job)["episode"]
            second = evaluator._evaluate_episode_job(explicit_job)["episode"]
        for field in (
            "scenario_id", "scenario_seed", "episode_length", "reward", "found",
            "contact_episode", "hold_episode", "success", "collision_episode",
            "executor_invalid_count", "executor_invalid_assignment_unreachable_count",
            "executor_min_distance_to_target", "executor_final_distance_to_target",
            "router_confusion_matrix", "gate_mean",
        ):
            self.assertEqual(first[field], second[field], field)

    def test_legacy_allocation_hash_payload_is_unchanged(self):
        executor = ExecutorAssignment(3, (1.0, 2.0, 3.0), (), 1.0, "source", True)
        allocation = OnlineAllocation((), executor, 2.0, 0.5, 1.0, "trigger", search_frozen=True)
        payload = {
            "search": [], "executor": (3, (1.0, 2.0, 3.0), "source"),
            "objective": 2.0, "detection": 0.5, "response": 1.0,
            "status": "OK", "search_frozen": True,
        }
        expected = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()).hexdigest()
        self.assertEqual(allocation.allocation_sha256, expected)


if __name__ == "__main__":
    unittest.main()
