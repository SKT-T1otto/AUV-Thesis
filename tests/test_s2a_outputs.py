from __future__ import annotations

import unittest

from chapter3_bser.experiments.phase1c_prrac.search_collision_recovery import aggregate_search_collision_recovery, baseline_recovery_summary, search_collision_recovery_failure_funnel


class OutputTests(unittest.TestCase):
    def test_scalar_baseline_and_distribution_fields(self):
        rows=[]
        for count in (0,1,10):
            row={**baseline_recovery_summary(),"found":count==0,"success":False,"contact_episode":False,"searcher_collision_episode_pre_found":count>0,"searcher_collision_count_pre_found_total":count,"searcher_collision_max_streak_pre_found":count,"executor_collision_count_post_found":0,"post_found_safe_hold_step_count":0,"post_found_route_inactive_step_count":0}
            rows.append(row)
        summary=aggregate_search_collision_recovery(rows,{})
        self.assertEqual(summary["pre_found_collision_count_median_all"],1.0); self.assertEqual(summary["pre_found_collision_count_p95_all"],9.1)

    def test_recovery_funnel_is_deterministic(self):
        base={"checkpoint":"c","execution_variant":"b","evaluation_mode":"m","search_recovery_variant":"v","manifest_sha256":"h"}
        rows=[{**base,"found":False,"searcher_collision_episode_pre_found":True,"search_recovery_entry_count":1},{**base,"found":True,"searcher_collision_episode_pre_found":False,"search_recovery_entry_count":0}]
        funnel=search_collision_recovery_failure_funnel(rows); counts={row["category"]:row["count"] for row in funnel}; self.assertEqual(counts["RECOVERY_TRIGGERED_NOT_FOUND"],1); self.assertEqual(counts["RECOVERY_NOT_TRIGGERED_FOUND"],1)


if __name__ == "__main__": unittest.main()
