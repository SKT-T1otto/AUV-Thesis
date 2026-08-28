import unittest

import numpy as np

from chapter3_bser.experiments.phase1c_prrac.search_continuity import SearchContinuityDiagnostics
from tests.search_continuity_test_support import Guidance, outputs, state


class SearchContinuityDiagnosticsTests(unittest.TestCase):
    def test_agent_step_denominators_and_public_deltas(self):
        diagnostic = SearchContinuityDiagnostics()
        before = state()
        after = state(step=1, offset=1.0, known=(True, True), entropy=0.75, peak=0.6)
        diagnostic.begin_episode(before)
        raw = np.ones((4, 3), dtype=np.float32)
        diagnostic.observe_transition(
            stage_before=0, stage_after=0, installed_guidance=Guidance(),
            planning_state_before=before, planning_state_after=after,
            collision_flags=(True, False, False, False), raw_actions=raw,
            applied_actions=raw.copy(), actor_outputs=outputs(-0.5),
        )
        row = diagnostic.summary(found=False, max_steps=400)
        self.assertEqual(row["pre_found_step_count"], 1)
        self.assertEqual(row["searcher_route_active_rate_pre_found"], 1.0)
        self.assertEqual(row["searcher_route_active_rate_pre_found_agent_0"], 1.0)
        self.assertEqual(row["searcher_collision_count_pre_found_agent_0"], 1)
        self.assertEqual(row["map_known_fraction_gain_pre_found"], 0.5)
        self.assertEqual(row["searcher_residual_negative_alignment_rate_pre_found"], 1.0)
        for agent_id in range(3):
            self.assertEqual(row[f"searcher_hold_rate_pre_found_agent_{agent_id}"], 0.0)
        self.assertFalse(row["searcher_residual_off_enabled"])
        self.assertEqual(
            row["searcher_raw_action_norm_pre_found"],
            row["searcher_raw_residual_norm_mean_pre_found"],
        )

    def test_zero_denominators_are_null(self):
        diagnostic = SearchContinuityDiagnostics()
        diagnostic.begin_episode(state())
        row = diagnostic.summary(found=False, max_steps=400)
        self.assertIsNone(row["searcher_route_active_rate_pre_found"])
        self.assertIsNone(row["searcher_route_active_rate_pre_found_agent_0"])
        for agent_id in range(3):
            self.assertIsNone(row[f"searcher_hold_rate_pre_found_agent_{agent_id}"])

    def test_suppression_counts_env_steps_and_agent_steps(self):
        before, after = state(), state(step=1)

        def summarize(applied):
            diagnostic = SearchContinuityDiagnostics()
            diagnostic.begin_episode(before)
            raw = np.ones((4, 3), dtype=np.float32)
            diagnostic.observe_transition(
                stage_before=0, stage_after=0, installed_guidance=Guidance(),
                planning_state_before=before, planning_state_after=after,
                collision_flags=(False,) * 4, raw_actions=raw,
                applied_actions=applied, actor_outputs=outputs(),
            )
            return diagnostic.summary(
                found=False, max_steps=400, searcher_residual_off_enabled=True
            )

        all_suppressed = np.ones((4, 3), dtype=np.float32)
        all_suppressed[:3] = 0.0
        one_suppressed = np.ones((4, 3), dtype=np.float32)
        one_suppressed[0] = 0.0
        for applied, env_steps, agent_steps in (
            (all_suppressed, 1, 3),
            (one_suppressed, 1, 1),
            (np.ones((4, 3), dtype=np.float32), 0, 0),
        ):
            row = summarize(applied)
            self.assertEqual(row["searcher_residual_suppressed_env_step_count_pre_found"], env_steps)
            self.assertEqual(row["searcher_residual_suppressed_agent_step_count_pre_found"], agent_steps)
            self.assertEqual(row["searcher_residual_suppressed_step_count_pre_found"], env_steps)
            self.assertTrue(row["searcher_residual_off_enabled"])

    def test_assignment_and_tracking_switches_are_separate(self):
        diagnostic = SearchContinuityDiagnostics()
        before = state()
        diagnostic.begin_episode(before)
        actions = np.ones((4, 3), dtype=np.float32)
        guidances = [Guidance() for _ in range(5)]
        guidances[1].assignments[0].assignment_id = "changed-id"
        guidances[2].assignments[0].assignment_id = "changed-id"
        guidances[2].assignments[0].final_waypoint = (9.0, 0.0, 0.0)
        guidances[3].assignments[0].assignment_id = "changed-id"
        guidances[3].assignments[0].final_waypoint = (9.0, 0.0, 0.0)
        guidances[3].assignments[0].tracking_waypoint = (8.0, 0.0, 0.0)
        guidances[4].assignments[0].assignment_id = "changed-id"
        guidances[4].assignments[0].final_waypoint = (9.0, 0.0, 0.0)
        guidances[4].assignments[0].tracking_waypoint = (8.0, 0.0, 0.0)
        current = before
        for step, guidance in enumerate(guidances, start=1):
            following = state(step=step)
            diagnostic.observe_transition(
                stage_before=0, stage_after=0, installed_guidance=guidance,
                planning_state_before=current, planning_state_after=following,
                collision_flags=(False,) * 4, raw_actions=actions,
                applied_actions=actions, actor_outputs=outputs(),
            )
            current = following
        row = diagnostic.summary(found=False, max_steps=400)
        self.assertEqual(row["searcher_assignment_switch_count_pre_found"], 2)
        self.assertEqual(row["searcher_tracking_subgoal_switch_count_pre_found"], 1)
        self.assertEqual(row["searcher_waypoint_switch_count_pre_found"], 3)

    def test_alignment_validity_excludes_zero_hold_and_unreachable(self):
        before, after = state(), state(step=1)
        actions = np.ones((4, 3), dtype=np.float32)

        def summarize(guidance, actor_outputs):
            diagnostic = SearchContinuityDiagnostics()
            diagnostic.begin_episode(before)
            diagnostic.observe_transition(
                stage_before=0, stage_after=0, installed_guidance=guidance,
                planning_state_before=before, planning_state_after=after,
                collision_flags=(False,) * 4, raw_actions=actions,
                applied_actions=actions, actor_outputs=actor_outputs,
            )
            return diagnostic.summary(found=False, max_steps=400)

        active = summarize(Guidance(), outputs(-0.5))
        self.assertEqual(active["searcher_residual_alignment_valid_count_pre_found"], 3)
        zero_navigation_guidance = Guidance()
        for agent_id in range(3):
            zero_navigation_guidance.assignments[agent_id].tracking_waypoint = before.agents[agent_id].position
        zero_navigation = summarize(zero_navigation_guidance, outputs(-0.5))
        self.assertEqual(zero_navigation["searcher_residual_alignment_valid_count_pre_found"], 0)
        self.assertEqual(zero_navigation["searcher_residual_alignment_zero_navigation_count_pre_found"], 3)
        zero_residual = summarize(Guidance(), outputs(-0.5, residual=(0.0, 0.0, 0.0)))
        self.assertEqual(zero_residual["searcher_residual_alignment_valid_count_pre_found"], 0)
        self.assertEqual(zero_residual["searcher_residual_alignment_zero_residual_count_pre_found"], 3)
        self.assertEqual(summarize(Guidance(hold=True), outputs(-0.5))["searcher_residual_alignment_valid_count_pre_found"], 0)
        self.assertEqual(summarize(Guidance(reachable=False), outputs(-0.5))["searcher_residual_alignment_valid_count_pre_found"], 0)


if __name__ == "__main__":
    unittest.main()
