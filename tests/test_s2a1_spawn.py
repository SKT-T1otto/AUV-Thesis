from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import tempfile
import unittest

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.experiments.phase1c_prrac.search_collision_recovery import (
    ACTIVATION_DIAGNOSTICS_SCHEMA,
    SEARCH_COLLISION_RECOVERY_SCHEMA_V2,
    SearchRecoveryVariantV2,
)
from tests.prrac_evaluation_support import worker_jobs, write_checkpoint


def s2a1_job(job, variant):
    value = dict(job); info = dict(value["checkpoint_info"])
    info.update({"execution_variant":"B1_ATOMIC_LAST_VALID","runtime_integration_mode":"overlay","runtime_overlay_enabled":True,"checkpoint_runtime_revision":"dynamic_public_intercept_v2_1","evaluation_runtime_revision":"dynamic_public_intercept_v3_reachable_proxy","manifest_sha256":"shared-s2a1-manifest","execution_overlay_config_hash":"overlay-hash","search_continuity_diagnostics_hash":"search-hash","search_recovery_variant":variant.value,"search_collision_recovery_schema":SEARCH_COLLISION_RECOVERY_SCHEMA_V2,"search_collision_recovery_config_hash":"s2a1-hash","activation_diagnostics_schema":ACTIVATION_DIAGNOSTICS_SCHEMA,"report_schema":"bser.phase1c.prrac.evaluation_report.v2"})
    value["checkpoint_info"] = info
    return value


class S2A1SpawnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.jobs = worker_jobs(write_checkpoint(Path(cls.temporary.name) / "checkpoint.pt"), 2)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_spawn_one_and_two_are_ordered_plain_and_share_manifest(self):
        one_job = s2a1_job(self.jobs[0], SearchRecoveryVariantV2.S2A1_C0_BASELINE)
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
            one = list(pool.map(evaluator._evaluate_episode_job, [one_job]))
        mixed = [s2a1_job(job, variant) for variant in SearchRecoveryVariantV2 for job in self.jobs]
        with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
            two = list(pool.map(evaluator._evaluate_episode_job, mixed))
        self.assertFalse(evaluator._contains_tensor(one)); self.assertFalse(evaluator._contains_tensor(two))
        expected = [job["scenario"]["scenario_id"] for job in self.jobs]
        for offset in (0, 2, 4):
            self.assertEqual([item["episode"]["scenario_id"] for item in two[offset:offset+2]], expected)
        self.assertEqual({item["episode"]["manifest_sha256"] for item in two}, {"shared-s2a1-manifest"})
        self.assertTrue(all(isinstance(item.get("recovery_planning_failures"), list) for item in two))


if __name__ == "__main__":
    unittest.main()
