import unittest

from chapter3_bser.experiments.phase1c_prrac.search_continuity import paired_searcher_residual_comparisons


def row(mode, scenario, found, success):
    return {
        "checkpoint": "checkpoint.pt", "checkpoint_config_hash": "cfg",
        "checkpoint_runtime_revision": "dynamic_public_intercept_v2_1",
        "evaluation_runtime_revision": "overlay", "runtime_integration_mode": "overlay",
        "execution_variant": "B1_ATOMIC_LAST_VALID", "manifest_sha256": "manifest",
        "search_continuity_diagnostics_hash": "search", "evaluation_mode": mode,
        "scenario_id": scenario, "found": found, "success": success,
        "contact_episode": success, "found_step": 10 if found else None,
        "max_steps": 40, "searcher_collision_episode_pre_found": False,
        "searcher_route_active_rate_pre_found": 1.0,
        "searcher_hold_rate_pre_found": 0.0,
        "searcher_distance_travelled_pre_found": 2.0,
        "map_known_fraction_gain_pre_found": 0.2,
        "target_belief_entropy_delta_pre_found": -0.1,
        "target_belief_peak_delta_pre_found": 0.1,
        "searcher_raw_residual_norm_mean_pre_found": 0.5,
        "searcher_applied_residual_norm_mean_pre_found": 0.5 if mode == "full_prrac" else 0.0,
        "searcher_raw_action_norm_pre_found": 0.5,
        "searcher_applied_action_norm_pre_found": 0.5 if mode == "full_prrac" else 0.0,
        "searcher_assignment_switch_count_pre_found": 2,
        "searcher_tracking_subgoal_switch_count_pre_found": 1 if mode == "full_prrac" else 0,
        "searcher_residual_suppressed_env_step_count_pre_found": 0 if mode == "full_prrac" else 1,
        "searcher_residual_suppressed_agent_step_count_pre_found": 0 if mode == "full_prrac" else 3,
        "searcher_residual_alignment_zero_navigation_count_pre_found": 0,
        "searcher_residual_alignment_zero_residual_count_pre_found": 0,
        "searcher_residual_negative_alignment_rate_pre_found": 0.25,
    }


class SearcherResidualPairedEvaluationTests(unittest.TestCase):
    def test_exact_scenario_pair_produces_one_row(self):
        rows = [row("full_prrac", "a", True, True), row("full_prrac", "b", False, False), row("searcher_residual_off", "a", True, False), row("searcher_residual_off", "b", True, True)]
        result = paired_searcher_residual_comparisons(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["paired_scenario_count"], 2)
        self.assertEqual(result[0]["searcher_off_only_found"], 1)
        self.assertEqual(result[0]["searcher_applied_action_norm_difference"], -0.5)
        self.assertEqual(result[0]["searcher_tracking_subgoal_switch_count_difference"], -1.0)

    def test_manifest_mismatch_is_rejected(self):
        rows = [row("full_prrac", "a", True, True), row("searcher_residual_off", "b", True, True)]
        with self.assertRaisesRegex(ValueError, "identical scenario_id"):
            paired_searcher_residual_comparisons(rows)


if __name__ == "__main__":
    unittest.main()
