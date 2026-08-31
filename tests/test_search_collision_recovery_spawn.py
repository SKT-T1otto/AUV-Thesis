from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import tempfile
import unittest

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.experiments.phase1c_prrac.search_collision_recovery import SEARCH_COLLISION_RECOVERY_SCHEMA, SearchRecoveryVariant
from tests.prrac_evaluation_support import worker_jobs, write_checkpoint


def recovery_job(job, variant):
    value=dict(job); info=dict(value["checkpoint_info"])
    info.update({"execution_variant":"B1_ATOMIC_LAST_VALID","runtime_integration_mode":"overlay","runtime_overlay_enabled":True,"checkpoint_runtime_revision":"dynamic_public_intercept_v2_1","evaluation_runtime_revision":"dynamic_public_intercept_v3_reachable_proxy","manifest_sha256":"shared-s2a-manifest","execution_overlay_config_hash":"overlay-hash","search_continuity_diagnostics_hash":"search-hash","search_recovery_variant":variant.value,"search_collision_recovery_schema":SEARCH_COLLISION_RECOVERY_SCHEMA,"search_collision_recovery_config_hash":"recovery-hash","report_schema":"bser.phase1c.prrac.evaluation_report.v2"})
    value["checkpoint_info"]=info
    return value


class SearchCollisionRecoverySpawnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(); checkpoint=write_checkpoint(Path(cls.temporary.name)/"checkpoint.pt"); cls.jobs=worker_jobs(checkpoint,2)

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    def test_spawn_one_and_two_workers_are_ordered_plain_and_paired(self):
        one_job=recovery_job(self.jobs[0],SearchRecoveryVariant.S2A_C0_BASELINE)
        with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context("spawn")) as pool: one=list(pool.map(evaluator._evaluate_episode_job,[one_job]))
        mixed=[recovery_job(job,variant) for variant in SearchRecoveryVariant for job in self.jobs]
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context("spawn")) as pool: two=list(pool.map(evaluator._evaluate_episode_job,mixed))
        self.assertFalse(evaluator._contains_tensor(one)); self.assertFalse(evaluator._contains_tensor(two))
        expected=[job["scenario"]["scenario_id"] for job in self.jobs]
        for offset in (0,2,4): self.assertEqual([item["episode"]["scenario_id"] for item in two[offset:offset+2]],expected)
        self.assertEqual({item["episode"]["manifest_sha256"] for item in two},{"shared-s2a-manifest"})


if __name__ == "__main__": unittest.main()
