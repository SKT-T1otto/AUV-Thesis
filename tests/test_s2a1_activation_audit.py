from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.select_phase1c_prrac_s2a1_activation_scenarios import select_scenarios
from scripts.validate_phase1c_prrac_s2a1_activation import validate_activation
from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from tests.prrac_evaluation_support import evaluation_config


class ActivationInfrastructureTests(unittest.TestCase):
    def test_targeted_manifest_resolves_final_count_and_preserves_file_order(self):
        config = evaluation_config(scenario_count=5)
        config["requested_evaluation_episodes"] = 5
        generated, _ = evaluator._build_evaluation_manifest(config)
        selected = [generated[3]["scenario_id"], generated[1]["scenario_id"]]
        config["scenario_ids"] = selected
        scenarios, manifest = evaluator._build_evaluation_manifest(config)
        self.assertEqual([row["scenario_id"] for row in scenarios], selected)
        self.assertEqual(manifest["requested_evaluation_episodes"], 5)
        self.assertEqual(manifest["generated_scenario_count"], 5)
        self.assertEqual(manifest["selected_scenario_count"], 2)
        self.assertEqual(manifest["resolved_evaluation_episodes"], 2)
        self.assertEqual(manifest["evaluation_episodes"], 2)

    def test_scenario_id_file_rejects_empty_duplicate_and_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ids.json"
            for payload in ([], ["same", "same"], "not-a-list"):
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    evaluator._load_scenario_id_file(path)
            config = evaluation_config(scenario_count=2)
            config["requested_evaluation_episodes"] = 2
            config["scenario_ids"] = ["unknown"]
            with self.assertRaisesRegex(ValueError, "outside the generated manifest"):
                evaluator._build_evaluation_manifest(config)
            path.write_text(json.dumps(["scenario"]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "--formal cannot be combined"):
                evaluator.run_evaluation(formal=True, scenario_id_file=path)

    def test_activation_dedupe_uses_stable_identity(self):
        row = {"checkpoint":"c", "search_recovery_variant":"S2A1_C1_FORCED_REFRESH", "scenario_id":"s", "step":1, "agent_id":0, "attempt_id":1}
        self.assertEqual(evaluator._dedupe_activation_rows([row, dict(row)]), [row])

    def test_targeted_selector_uses_only_baseline_prefound_collision_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "episodes.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("search_recovery_variant", "scenario_id", "searcher_collision_episode_pre_found"))
                writer.writeheader()
                writer.writerows((
                    {"search_recovery_variant":"S2A_C0_BASELINE","scenario_id":"a","searcher_collision_episode_pre_found":"False"},
                    {"search_recovery_variant":"S2A_C0_BASELINE","scenario_id":"b","searcher_collision_episode_pre_found":"True"},
                    {"search_recovery_variant":"S2A_C1_ROUTE_REFRESH","scenario_id":"c","searcher_collision_episode_pre_found":"True"},
                    {"search_recovery_variant":"S2A_C0_BASELINE","scenario_id":"d","searcher_collision_episode_pre_found":"True"},
                ))
            result = select_scenarios(path, 2)
            self.assertEqual(result["scenario_ids"], ["b", "d"])
            self.assertTrue(result["diagnostic_only"])

    def test_validator_rejects_observationally_inert_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.csv"
            fields = ("search_recovery_variant", "search_collision_recovery_schema", "manifest_sha256", "search_recovery_entry_count", "forced_public_refresh_count", "recovery_plan_active_step_count", "recovery_guidance_changed_step_count", "recovery_effective_intervention_count", "recovery_effective_intervention_episode_count", "local_connector_attempt_count", "local_connector_plan_count")
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
                for variant in ("S2A1_C0_BASELINE", "S2A1_C1_FORCED_REFRESH", "S2A1_C2_LOCAL_CONNECTOR"):
                    writer.writerow({"search_recovery_variant":variant,"search_collision_recovery_schema":"bser.phase1c.prrac.search_collision_recovery.v2","manifest_sha256":"m","search_recovery_entry_count":0 if variant.endswith("BASELINE") else 1,"forced_public_refresh_count":0 if variant.endswith("BASELINE") else 1,"recovery_plan_active_step_count":0,"recovery_guidance_changed_step_count":0,"recovery_effective_intervention_count":0,"recovery_effective_intervention_episode_count":0,"local_connector_attempt_count":1 if variant.endswith("LOCAL_CONNECTOR") else 0,"local_connector_plan_count":0})
            self.assertEqual(validate_activation(path)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
