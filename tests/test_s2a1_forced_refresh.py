from __future__ import annotations

import unittest
from dataclasses import replace

from chapter3_bser.experiments.phase1c_prrac.search_collision_recovery import SearchRecoveryVariantV2, apply_search_recovery_guidance, build_search_recovery_controller
from tests.bser_test_utils import synthetic_state
from tests.search_collision_recovery_test_support import guidance, state_at


class ForcedRefreshTests(unittest.TestCase):
    def test_collision_edge_requests_one_force_and_rearms_after_clear(self):
        controller = build_search_recovery_controller(SearchRecoveryVariantV2.S2A1_C1_FORCED_REFRESH)
        state = synthetic_state(); base = guidance()
        controller.observe_transition(stage_before=0, planning_state_before=state, planning_state_after=state_at(state, 1),
                                      installed_guidance_before=base, collision_flags=(True, False, False, False))
        self.assertTrue(controller.force_refresh_requested)
        controller.prepare_next_guidance(state_at(state, 1), base)
        controller.observe_transition(stage_before=0, planning_state_before=state_at(state, 1), planning_state_after=state_at(state, 2),
                                      installed_guidance_before=base, collision_flags=(True, False, False, False))
        self.assertFalse(controller.force_refresh_requested)
        controller.observe_transition(stage_before=0, planning_state_before=state_at(state, 2), planning_state_after=state_at(state, 3),
                                      installed_guidance_before=base, collision_flags=(False, False, False, False))
        controller.observe_transition(stage_before=0, planning_state_before=state_at(state, 3), planning_state_after=state_at(state, 4),
                                      installed_guidance_before=base, collision_flags=(True, False, False, False))
        self.assertTrue(controller.force_refresh_requested)

    def test_found_terminates_without_force(self):
        controller = build_search_recovery_controller(SearchRecoveryVariantV2.S2A1_C2_LOCAL_CONNECTOR)
        state = synthetic_state(); base = guidance(); found = state_at(state, 1, found=True)
        controller.observe_transition(stage_before=0, planning_state_before=state, planning_state_after=found,
                                      installed_guidance_before=base, collision_flags=(True, False, False, False))
        self.assertFalse(controller.force_refresh_requested)
        self.assertEqual(controller.summary()["search_recovery_entry_count"], 0)

    def test_v2_overlay_preserves_collision_edge_semantic_identity_and_allocation(self):
        controller = build_search_recovery_controller(SearchRecoveryVariantV2.S2A1_C1_FORCED_REFRESH)
        state = synthetic_state(); base = guidance(); after = state_at(state, 1)
        controller.observe_transition(stage_before=0, planning_state_before=state, planning_state_after=after,
                                      installed_guidance_before=base, collision_flags=(True, False, False, False))
        changed_assignment = replace(base.assignment_for(0), assignment_id="new-candidate", final_waypoint=(2.5, 2.5, 1.0))
        refreshed_base = replace(base, allocation_version="new-version", allocation_hash="new-hash",
                                 agent_assignments=(changed_assignment, *base.agent_assignments[1:]))
        controller.prepare_next_guidance(after, refreshed_base)
        overlay = apply_search_recovery_guidance(refreshed_base, after, controller)
        self.assertEqual(overlay.assignment_for(0).assignment_id, base.assignment_for(0).assignment_id)
        self.assertEqual(overlay.assignment_for(0).final_waypoint, base.assignment_for(0).final_waypoint)
        self.assertEqual(overlay.allocation_hash, base.allocation_hash)


if __name__ == "__main__":
    unittest.main()
