from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.select_phase1c_prrac_s2a1_activation_scenarios import select_scenarios
from scripts.validate_phase1c_prrac_s2a1_activation import validate_activation


class ActivationInfrastructureTests(unittest.TestCase):
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
