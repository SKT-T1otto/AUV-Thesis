from __future__ import annotations

import unittest
from pathlib import Path

from chapter3_bser.experiments.phase1c_prrac.evaluation_provenance import derive_unique_provenance, validate_evaluation_provenance, validate_resume_config, validate_summary_provenance
from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.experiments.phase1c_prrac.execution_continuity import ExecutionVariant


class EvaluationProvenanceTests(unittest.TestCase):
    def valid_protocol(self):
        checkpoint = str(Path("synthetic.pt").resolve())
        row = {
            "checkpoint": checkpoint, "checkpoint_episode": 12,
            "checkpoint_config_hash": "checkpoint-hash",
            "checkpoint_runtime_revision": "runtime", "evaluation_runtime_revision": "runtime",
            "runtime_integration_mode": "native", "execution_variant": "B1_ATOMIC_LAST_VALID",
            "evaluation_mode": "full_prrac", "manifest_sha256": "manifest",
            "execution_overlay_config_hash": "overlay", "search_continuity_diagnostics_hash": "search",
            "search_recovery_variant": "S2A1_C0_BASELINE",
            "search_collision_recovery_schema": "bser.phase1c.prrac.search_collision_recovery.v2",
            "search_collision_recovery_config_hash": "recovery",
            "activation_diagnostics_schema": "bser.phase1c.prrac.search_collision_recovery.activation.v2",
            "activation_artifact_revision": "s2a1.activation_artifact.v1",
            "s2a1_activation_artifact_revision": "s2a1.activation_artifact.v1",
            "report_schema": "bser.phase1c.prrac.evaluation_report.v2",
            "scenario_id": "scenario", "scenario_seed": 7,
        }
        config = {
            "schema": row["report_schema"], "report_schema": row["report_schema"],
            "manifest_sha256": row["manifest_sha256"], "resolved_checkpoint_paths": [checkpoint],
            "resolved_evaluation_modes": [row["evaluation_mode"]],
            "resolved_execution_variants": [row["execution_variant"]],
            "resolved_search_recovery_variants": [row["search_recovery_variant"]],
            "resolved_evaluation_runtime_revisions": [row["evaluation_runtime_revision"]],
            "resolved_runtime_integration_modes": [row["runtime_integration_mode"]],
            "resolved_evaluation_episodes": 1, "checkpoint_runtime_revision": "runtime",
            "search_continuity_diagnostics_hash": "search",
            "search_collision_recovery_schema": row["search_collision_recovery_schema"],
            "search_collision_recovery_config_hash": "recovery",
            "activation_diagnostics_schema": row["activation_diagnostics_schema"],
            "activation_artifact_revision": row["activation_artifact_revision"],
            "s2a1_activation_artifact_revision": row["s2a1_activation_artifact_revision"],
        }
        combo_fields = (
            "checkpoint", "checkpoint_config_hash", "checkpoint_episode", "checkpoint_runtime_revision",
            "evaluation_runtime_revision", "runtime_integration_mode", "execution_variant", "evaluation_mode",
            "search_recovery_variant", "manifest_sha256", "execution_overlay_config_hash",
            "search_continuity_diagnostics_hash", "search_collision_recovery_schema",
            "search_collision_recovery_config_hash", "activation_diagnostics_schema",
            "activation_artifact_revision", "s2a1_activation_artifact_revision", "report_schema",
        )
        progress = {
            "schema": "bser.phase1c.prrac.evaluation_progress.v2", "resolved_evaluation_episodes": 1,
            **{key: config[key] for key in (
                "manifest_sha256", "search_collision_recovery_schema", "activation_diagnostics_schema",
                "activation_artifact_revision", "s2a1_activation_artifact_revision", "report_schema",
            )},
            "completed": [{key: row[key] for key in combo_fields}],
        }
        metadata = [{"checkpoint": checkpoint, "completed_episode": 12, "metadata": {"config_hash":"checkpoint-hash", "execution_runtime_revision":"runtime"}}]
        return row, config, progress, metadata

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

    def test_row_recovery_and_activation_schema_mismatches_raise(self):
        for field, wrong in (
            ("search_collision_recovery_schema", "bser.phase1c.prrac.search_collision_recovery.v1"),
            ("activation_diagnostics_schema", "wrong.activation.schema"),
        ):
            row, config, progress, metadata = self.valid_protocol()
            row[field] = wrong
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_evaluation_provenance(rows=[row], resolved_config=config, progress=progress,
                    checkpoint_metadata=metadata, expected_scenarios=[{"scenario_id":"scenario", "scenario_seed":7}])

    def test_progress_activation_schema_mismatch_raises(self):
        row, config, progress, metadata = self.valid_protocol()
        progress["completed"][0]["activation_diagnostics_schema"] = "wrong.activation.schema"
        with self.assertRaises(ValueError):
            validate_evaluation_provenance(rows=[row], resolved_config=config, progress=progress,
                checkpoint_metadata=metadata, expected_scenarios=[{"scenario_id":"scenario", "scenario_seed":7}])

    def test_summary_cannot_mix_v1_and_v2_schema(self):
        row, _, _, _ = self.valid_protocol()
        summary = dict(row)
        summary["search_collision_recovery_schema"] = "bser.phase1c.prrac.search_collision_recovery.v1"
        with self.assertRaises(ValueError):
            validate_summary_provenance([row], [summary], [{"scenario_id":"scenario", "scenario_seed":7}])


if __name__ == "__main__": unittest.main()
