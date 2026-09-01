from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_phase1c_prrac_s2a1_activation import (
    ACTIVATION_ARTIFACT_REVISION,
    ACTIVATION_SCHEMA,
    RECOVERY_SCHEMA,
    REPORT_SCHEMA,
    REQUIRED_VARIANTS,
    validate_output_dir,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ActivationProtocolTests(unittest.TestCase):
    def test_complete_protocol_passes_and_rejects_c0_activation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            checkpoint = "checkpoint.pt"
            common = {
                "checkpoint": checkpoint,
                "checkpoint_episode": 12,
                "checkpoint_config_hash": "checkpoint-config-hash",
                "checkpoint_runtime_revision": "dynamic_public_intercept_v3_atomic_continuity",
                "evaluation_runtime_revision": "dynamic_public_intercept_v3_atomic_continuity",
                "runtime_integration_mode": "native",
                "execution_variant": "B1_ATOMIC_LAST_VALID",
                "evaluation_mode": "full_prrac",
                "manifest_sha256": "manifest-hash",
                "search_collision_recovery_schema": RECOVERY_SCHEMA,
                "search_collision_recovery_config_hash": "recovery-hash",
                "activation_diagnostics_schema": ACTIVATION_SCHEMA,
                "activation_artifact_revision": ACTIVATION_ARTIFACT_REVISION,
                "s2a1_activation_artifact_revision": ACTIVATION_ARTIFACT_REVISION,
                "report_schema": REPORT_SCHEMA,
            }
            config = {
                "schema": REPORT_SCHEMA,
                **{key: common[key] for key in (
                    "manifest_sha256", "search_collision_recovery_schema",
                    "search_collision_recovery_config_hash", "activation_diagnostics_schema",
                    "activation_artifact_revision", "report_schema",
                    "s2a1_activation_artifact_revision",
                )},
                "checkpoint_runtime_revision": "dynamic_public_intercept_v3_atomic_continuity",
                "resolved_evaluation_runtime_revisions": ["dynamic_public_intercept_v3_atomic_continuity"],
                "resolved_runtime_integration_modes": ["native"],
                "resolved_checkpoint_paths": [checkpoint],
                "resolved_evaluation_modes": ["full_prrac"],
                "resolved_execution_variants": ["B1_ATOMIC_LAST_VALID"],
                "requested_evaluation_episodes": 5,
                "generated_scenario_count": 5,
                "selected_scenario_count": 1,
                "resolved_evaluation_episodes": 1,
                "evaluation_episodes": 1,
                "diagnostic_only": True,
                "scenario_selection_mode": "baseline_collision_targeted_smoke",
            }
            manifest = {
                "manifest_sha256": "manifest-hash", "evaluation_episodes": 1,
                "requested_evaluation_episodes": 5, "generated_scenario_count": 5,
                "selected_scenario_count": 1, "resolved_evaluation_episodes": 1,
                "diagnostic_only": True, "scenario_selection_mode": "baseline_collision_targeted_smoke",
                "scenarios": [{"scenario_id": "scenario-1", "scenario_seed": 7}],
            }
            completed = [{**common, "search_recovery_variant": variant} for variant in REQUIRED_VARIANTS]
            progress = {
                "schema": "bser.phase1c.prrac.evaluation_progress.v2",
                **{key: common[key] for key in (
                    "manifest_sha256", "search_collision_recovery_schema",
                    "search_collision_recovery_config_hash", "activation_diagnostics_schema",
                    "activation_artifact_revision", "report_schema",
                    "s2a1_activation_artifact_revision",
                )},
                "selected_scenario_count": 1, "resolved_evaluation_episodes": 1,
                "diagnostic_only": True, "scenario_selection_mode": "baseline_collision_targeted_smoke",
                "completed": completed,
            }
            episodes = [{**common, "search_recovery_variant": variant, "scenario_id": "scenario-1", "scenario_seed": 7} for variant in REQUIRED_VARIANTS]
            summaries = []
            for variant in REQUIRED_VARIANTS:
                active = variant != REQUIRED_VARIANTS[0]
                local = variant == REQUIRED_VARIANTS[2]
                summaries.append({
                    **common, "search_recovery_variant": variant, "evaluation_episodes": 1,
                    "search_recovery_entry_count": int(active),
                    "forced_public_refresh_count": int(active),
                    "recovery_plan_active_step_count": int(active),
                    "recovery_guidance_changed_step_count": int(active),
                    "recovery_effective_intervention_count": int(active),
                    "recovery_effective_intervention_episode_count": int(active),
                    "local_connector_attempt_count": int(local),
                    "local_connector_plan_count": int(local),
                })
            activation = [{
                **common, "search_recovery_variant": REQUIRED_VARIANTS[1],
                "scenario_id": "scenario-1", "scenario_seed": 7, "step": 2,
                "agent_id": 0, "attempt_id": 1, "recovery_mode": "ROUTE_REFRESH",
                "recovery_plan_installed": True, "guidance_changed": True,
                "path_changed": True, "tracking_waypoint_delta_norm": 1.0,
            }]
            for name, value in (("resolved_evaluation_config.json", config), ("evaluation_manifest.json", manifest), ("evaluation_progress.json", progress)):
                (output / name).write_text(json.dumps(value), encoding="utf-8")
            write_csv(output / "search_collision_recovery_episode.csv", episodes)
            write_csv(output / "search_collision_recovery_summary.csv", summaries)
            write_csv(output / "search_collision_recovery_activation_steps.csv", activation)
            self.assertEqual(validate_output_dir(output)["status"], "PASS")
            activation.append({**activation[0], "search_recovery_variant": REQUIRED_VARIANTS[0], "step": 3})
            write_csv(output / "search_collision_recovery_activation_steps.csv", activation)
            result = validate_output_dir(output)
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("c0_has_zero_activation_rows", result["failures"])

            write_csv(output / "search_collision_recovery_activation_steps.csv", activation[:1])
            bad_seed = [dict(row) for row in episodes]
            bad_seed[-1]["scenario_seed"] = 8
            write_csv(output / "search_collision_recovery_episode.csv", bad_seed)
            self.assertIn("scenario_ids_and_seeds_paired", validate_output_dir(output)["failures"])

            write_csv(output / "search_collision_recovery_episode.csv", episodes)
            inert = [dict(row) for row in summaries]
            for row in inert:
                if row["search_recovery_variant"] != REQUIRED_VARIANTS[0]:
                    row["recovery_plan_active_step_count"] = 0
                    row["recovery_guidance_changed_step_count"] = 0
                    row["recovery_effective_intervention_episode_count"] = 0
            write_csv(output / "search_collision_recovery_summary.csv", inert)
            self.assertIn("active_intervention", validate_output_dir(output)["failures"])

            write_csv(output / "search_collision_recovery_summary.csv", summaries)
            config["checkpoint_runtime_revision"] = "wrong-runtime"
            (output / "resolved_evaluation_config.json").write_text(json.dumps(config), encoding="utf-8")
            self.assertIn("native_b1_protocol_identity", validate_output_dir(output)["failures"])

            config["checkpoint_runtime_revision"] = "dynamic_public_intercept_v3_atomic_continuity"
            (output / "resolved_evaluation_config.json").write_text(json.dumps(config), encoding="utf-8")
            (output / "search_collision_recovery_activation_steps.csv").unlink()
            self.assertIn("required_artifacts_present", validate_output_dir(output)["checks"])
            self.assertEqual(validate_output_dir(output)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
