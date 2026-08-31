from __future__ import annotations

import unittest
from pathlib import Path

from chapter3_bser.experiments.phase1c_prrac.evaluation_provenance import derive_unique_provenance, validate_evaluation_provenance, validate_resume_config
from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.experiments.phase1c_prrac.execution_continuity import ExecutionVariant


class EvaluationProvenanceTests(unittest.TestCase):
    def test_unique_and_plural_provenance(self):
        result = derive_unique_provenance([{"checkpoint_runtime_revision":"a"},{"checkpoint_runtime_revision":"b"}])
        self.assertIsNone(result["checkpoint_runtime_revision"])
        self.assertEqual(result["checkpoint_runtime_revision_values"], ["a","b"])

    def test_mismatch_raises(self):
        config={"schema":"bser.phase1c.prrac.evaluation_report.v2","manifest_sha256":"m","resolved_checkpoint_paths":[],"resolved_evaluation_modes":["full_prrac"],"resolved_execution_variants":["B1_ATOMIC_LAST_VALID"],"resolved_search_recovery_variants":["S2A_C0_BASELINE"]}
        progress={"schema":"bser.phase1c.prrac.evaluation_progress.v2","manifest_sha256":"wrong"}
        with self.assertRaises(ValueError):
            validate_evaluation_provenance(rows=[],resolved_config=config,progress=progress,checkpoint_metadata=[],expected_scenarios=[])

    def test_v1_cannot_resume_into_v2(self):
        expected={"schema":"bser.phase1c.prrac.evaluation_report.v2","resolved_config_hash":"h","manifest_sha256":"m","search_collision_recovery_config_hash":"r","search_continuity_diagnostics_hash":"s","resolved_scenario_ids":["x"]}
        with self.assertRaises(ValueError): validate_resume_config({**expected,"schema":"bser.phase1c.prrac.evaluation.v1"},expected)

    def test_summary_runtime_is_derived_from_rows(self):
        runtime="dynamic_public_intercept_v3_atomic_continuity"
        result=evaluator._execution_variant_summary(scenarios=[{}],variants=(ExecutionVariant.B1_ATOMIC_LAST_VALID,),summary_rows=[{"checkpoint_runtime_revision":runtime,"evaluation_runtime_revision":runtime,"execution_variant":"B1_ATOMIC_LAST_VALID"}],output=Path("."))
        self.assertEqual(result["checkpoint_runtime_revision"],runtime); self.assertEqual(result["evaluation_runtime_revision"],runtime)


if __name__ == "__main__": unittest.main()
