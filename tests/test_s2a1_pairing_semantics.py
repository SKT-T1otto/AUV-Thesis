from __future__ import annotations

import unittest

from chapter3_bser.experiments.phase1c_prrac.search_collision_recovery import paired_search_collision_recovery_comparisons


def row(variant, scenario_id, seed, collision, gain, distance):
    return {"checkpoint":"c","checkpoint_config_hash":"h","checkpoint_runtime_revision":"r","evaluation_runtime_revision":"r","runtime_integration_mode":"native","execution_variant":"B1_ATOMIC_LAST_VALID","evaluation_mode":"full_prrac","manifest_sha256":"m","search_collision_recovery_schema":"bser.phase1c.prrac.search_collision_recovery.v2","search_collision_recovery_config_hash":"x","search_recovery_variant":variant,"scenario_id":scenario_id,"scenario_seed":seed,"found":True,"success":False,"contact_episode":False,"searcher_collision_episode_pre_found":True,"searcher_collision_count_pre_found_total":collision,"searcher_collision_max_streak_pre_found":collision,"found_step":10,"searcher_distance_travelled_pre_found":distance,"map_known_fraction_gain_pre_found":gain}


class PairingSemanticsTests(unittest.TestCase):
    def test_directions_and_per_id_seed_validation(self):
        rows = [row("S2A1_C0_BASELINE","s0",1,3,.1,2), row("S2A1_C1_FORCED_REFRESH","s0",1,2,.2,3)]
        result = paired_search_collision_recovery_comparisons(rows)[0]
        self.assertEqual(result["pre_found_collision_count_better_direction"], "lower")
        self.assertEqual(result["map_known_fraction_gain_better_direction"], "higher")
        self.assertEqual(result["searcher_distance_travelled_better_direction"], "report_only")
        rows[1]["scenario_seed"] = 2
        with self.assertRaises(ValueError): paired_search_collision_recovery_comparisons(rows)


if __name__ == "__main__":
    unittest.main()
