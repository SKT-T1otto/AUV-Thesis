from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import tempfile
import unittest

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from tests.prrac_evaluation_support import worker_jobs, write_checkpoint


class PRRACEvaluationSpawnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.checkpoint = write_checkpoint(Path(cls.temporary.name) / "checkpoint.pt")
        cls.jobs = worker_jobs(cls.checkpoint, 2)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_spawn_one_worker_roundtrip_is_tensor_free(self) -> None:
        self.assertFalse(evaluator._contains_tensor(self.jobs[0]))
        with ProcessPoolExecutor(
            max_workers=1, mp_context=mp.get_context("spawn")
        ) as executor:
            result = executor.submit(evaluator._evaluate_episode_job, self.jobs[0]).result(
                timeout=240
            )
        self.assertFalse(evaluator._contains_tensor(result))

    def test_spawn_two_workers_preserves_single_process_manifest_order(self) -> None:
        single = [evaluator._evaluate_episode_job(job) for job in self.jobs]
        with ProcessPoolExecutor(
            max_workers=2, mp_context=mp.get_context("spawn")
        ) as executor:
            spawned = list(executor.map(evaluator._evaluate_episode_job, self.jobs))

        expected = [result["episode"]["scenario_id"] for result in single]
        actual = [result["episode"]["scenario_id"] for result in spawned]
        self.assertEqual(actual, expected)
        self.assertFalse(evaluator._contains_tensor(spawned))


if __name__ == "__main__":
    unittest.main()
