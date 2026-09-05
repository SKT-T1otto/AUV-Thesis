import unittest
import tempfile
from tests.search_value_audit_support import small_job
from chapter3_bser.experiments.phase1c_prrac.search_value_audit.runner import execute_work, no_op_check, run_work
from chapter3_bser.experiments.phase1c_prrac.search_value_audit.provenance import Progress


class AuditRuntimeTests(unittest.TestCase):
    def test_synthetic_shadow_closed_loop_is_no_op(self):
        job = small_job()
        bare = execute_work(dict(unit="bare", job=job, no_hook=True))
        control = execute_work(dict(unit="control", job=job, options=dict(capture=False)))
        shadow = execute_work(dict(unit="shadow", job=job, options=dict(shadow=True)))
        self.assertTrue(no_op_check(control["audit"], shadow["audit"], bare["payload"])["passed"])
        self.assertEqual(len(shadow["audit"]["predictions"]), 6)
        self.assertTrue(shadow["audit"]["candidate_representation"])

    def test_native_ON_no_op_and_spawn_branches_are_identical(self):
        job = small_job(enabled=True, max_steps=1)
        bare = execute_work(dict(unit="bare", job=job, no_hook=True))
        control = execute_work(dict(unit="control", job=job, options=dict(capture=False)))
        observed = execute_work(dict(unit="observed", job=job, options=dict(shadow=True)))
        self.assertTrue(no_op_check(control["audit"], observed["audit"], bare["payload"])["passed"])
        jobs = [dict(unit=b, job=job, branch=b, options=dict(root_step=0, intervention=b)) for b in ("A", "B")]
        with tempfile.TemporaryDirectory() as folder:
            serial = run_work(jobs, 1, Progress(folder, 2))
            parallel = run_work(jobs, 2, Progress(folder, 2))
        self.assertEqual(serial, parallel)
        for result in serial:
            self.assertEqual(result["audit"]["root_fingerprint"], observed["audit"]["boundaries"]["0"]["fingerprint"])
            self.assertEqual(result["audit"]["guided_after_root_count"], 0)


if __name__ == "__main__":
    unittest.main()
