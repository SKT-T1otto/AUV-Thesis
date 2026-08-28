import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import tempfile
import unittest

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from tests.prrac_evaluation_support import worker_jobs, write_checkpoint


class SearchContinuitySpawnTests(unittest.TestCase):
    def test_spawn_one_and_two_workers_are_ordered_and_tensor_free(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = write_checkpoint(Path(directory) / "checkpoint.pt")
            jobs = worker_jobs(checkpoint, count=2)
            for job in jobs:
                job["checkpoint_info"]["execution_variant"] = "B1_ATOMIC_LAST_VALID"
                job["checkpoint_info"]["runtime_integration_mode"] = "overlay"
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
                one = list(pool.map(evaluator._evaluate_episode_job, jobs[:1]))
            mixed = []
            for mode in ("full_prrac", "searcher_residual_off"):
                for job in jobs:
                    copied = dict(job)
                    copied["checkpoint_info"] = dict(job["checkpoint_info"], evaluation_mode=mode)
                    mixed.append(copied)
            with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
                two = list(pool.map(evaluator._evaluate_episode_job, mixed))
            self.assertFalse(evaluator._contains_tensor(one))
            self.assertFalse(evaluator._contains_tensor(two))
            self.assertEqual([item["episode"]["scenario_id"] for item in two[:2]], [item["episode"]["scenario_id"] for item in two[2:]])


if __name__ == "__main__":
    unittest.main()
