from __future__ import annotations

import unittest

from chapter3_bser.experiments.phase1c_prrac.search_collision_recovery import SearchRecoveryVariant, paired_search_collision_recovery_baseline_strata, paired_search_collision_recovery_comparisons


def rows():
    output=[]
    for variant in SearchRecoveryVariant:
        for index in range(2):
            output.append({"checkpoint":"c","checkpoint_config_hash":"h","checkpoint_runtime_revision":"r","evaluation_runtime_revision":"r","runtime_integration_mode":"native","execution_variant":"B1_ATOMIC_LAST_VALID","evaluation_mode":"full_prrac","manifest_sha256":"m","search_collision_recovery_config_hash":"x","search_recovery_variant":variant.value,"scenario_id":f"s{index}","scenario_seed":index,"found":index==0 or variant is SearchRecoveryVariant.S2A_C2_EGRESS_ROUTE,"success":index==0,"contact_episode":index==0,"searcher_collision_episode_pre_found":index==0,"searcher_collision_count_pre_found_total":2-index,"searcher_collision_max_streak_pre_found":2-index,"found_step":10+index,"searcher_distance_travelled_pre_found":3+index,"map_known_fraction_gain_pre_found":.1*index})
    return output


class PairingTests(unittest.TestCase):
    def test_three_pairs_and_c0_strata(self):
        paired=paired_search_collision_recovery_comparisons(rows()); self.assertEqual(len(paired),3); self.assertEqual(paired[0]["continuous_difference_direction"],"right_minus_left")
        strata=paired_search_collision_recovery_baseline_strata(rows()); self.assertEqual(len(strata),4); self.assertTrue(all(row["stratum_definition"].startswith("S2A_C0") for row in strata))

    def test_scenario_mismatch_raises(self):
        value=rows(); value.pop()
        with self.assertRaises(ValueError): paired_search_collision_recovery_comparisons(value)


if __name__ == "__main__": unittest.main()
