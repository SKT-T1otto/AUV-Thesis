from __future__ import annotations

from dataclasses import replace
import unittest

from chapter3_bser.controllers.path_tracker import PathTracker
from chapter3_bser.experiments.phase1c_prrac.search_collision_recovery import (
    SearchRecoveryVariantV2,
    apply_search_recovery_guidance,
    build_search_recovery_controller,
    plan_forced_route_refresh,
)
from tests.bser_test_utils import synthetic_state
from tests.search_collision_recovery_test_support import guidance, state_at


class NoPlanAccountingTests(unittest.TestCase):
    def test_identical_refresh_is_not_effective_or_plan_active(self):
        state = synthetic_state(); base = guidance(); assignment = base.assignment_for(0)
        plan, _, _ = plan_forced_route_refresh(state, 0, assignment.assignment_id, assignment.final_waypoint,
                                                assignment.tracking_waypoint, None, attempt_id=1)
        tracking = PathTracker().tracking_target(0, state.agents[0].position, plan.path, plan.navigation_endpoint)
        identical_assignment = replace(assignment, planned_path=plan.path, tracking_waypoint=tracking)
        identical = replace(base, agent_assignments=(identical_assignment, *base.agent_assignments[1:]))
        controller = build_search_recovery_controller(SearchRecoveryVariantV2.S2A1_C1_FORCED_REFRESH)
        after = state_at(state, 1)
        controller.observe_transition(stage_before=0, planning_state_before=state, planning_state_after=after,
                                      installed_guidance_before=identical, collision_flags=(True, False, False, False))
        controller.prepare_next_guidance(after, identical)
        overlay = apply_search_recovery_guidance(identical, after, controller)
        controller.observe_activation(identical, overlay)
        summary = controller.summary()
        self.assertEqual(summary["route_refresh_identical_to_base_count"], 1)
        self.assertEqual(summary["recovery_plan_active_step_count"], 0)
        self.assertEqual(summary["recovery_guidance_changed_step_count"], 0)
        self.assertEqual(summary["recovery_effective_intervention_count"], 0)
        self.assertEqual(summary["recovery_failed_pass_through_count"], 1)


if __name__ == "__main__":
    unittest.main()
