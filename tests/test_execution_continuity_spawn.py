from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import tempfile
import unittest

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.experiments.phase1c_prrac.execution_continuity import ExecutionVariant
from tests.prrac_evaluation_support import worker_jobs, write_checkpoint


def variant_job(job, variant):
    value = dict(job)
    info = dict(value["checkpoint_info"])
    info["execution_variant"] = variant.value
    info["runtime_overlay_enabled"] = variant is not ExecutionVariant.B0_LEGACY_V2_1
    info["checkpoint_runtime_revision"] = "dynamic_public_intercept_v2_1"
    info["evaluation_runtime_revision"] = (
        "dynamic_public_intercept_v2_1" if variant is ExecutionVariant.B0_LEGACY_V2_1
        else "dynamic_public_intercept_v3_reachable_proxy"
    )
    info["manifest_sha256"] = "spawn-test"
    info["execution_overlay_config_hash"] = "overlay-test"
    value["checkpoint_info"] = info
    return value


class ExecutionContinuitySpawnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        checkpoint = write_checkpoint(Path(cls.temporary.name) / "checkpoint.pt")
        jobs = worker_jobs(checkpoint, 2)
        cls.jobs = [variant_job(job, ExecutionVariant.B3_PROXY_SAFE_SUPPRESSION) for job in jobs]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_spawn_one_and_two_worker_roundtrips_are_tensor_free(self):
        for workers, jobs in ((1, self.jobs[:1]), (2, self.jobs)):
            self.assertFalse(evaluator._contains_tensor(jobs))
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
                results = list(pool.map(evaluator._evaluate_episode_job, jobs))
            self.assertFalse(evaluator._contains_tensor(results))
            self.assertEqual(
                [result["episode"]["scenario_id"] for result in results],
                [job["scenario"]["scenario_id"] for job in jobs],
            )


if __name__ == "__main__":
    unittest.main()
