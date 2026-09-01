from __future__ import annotations

from dataclasses import replace
import unittest

from chapter3_bser.experiments.phase1c_prrac.search_collision_recovery import (
    SearchRecoveryVariantV2,
    apply_search_recovery_guidance,
    build_search_recovery_controller,
)
from tests.bser_test_utils import synthetic_state
from tests.search_collision_recovery_test_support import guidance, state_at


class GraphReconnectTests(unittest.TestCase):
    def test_unconnectable_collision_position_gets_local_plan_then_reconnects(self):
        controller = build_search_recovery_controller(SearchRecoveryVariantV2.S2A1_C2_LOCAL_CONNECTOR)
        state = synthetic_state(); base = guidance(); clear = state_at(state, 1)
        controller.observe_transition(stage_before=0, planning_state_before=state, planning_state_after=clear,
                                      installed_guidance_before=base, collision_flags=(False, False, False, False))
        moved_agent = replace(state.agents[0], position=(0.75, 0.5, 1.0))
        collision = replace(state_at(state, 2), agents=(moved_agent, *state.agents[1:]))
        controller.observe_transition(stage_before=0, planning_state_before=clear, planning_state_after=collision,
                                      installed_guidance_before=base, collision_flags=(True, False, False, False))
        controller.prepare_next_guidance(collision, base)
        plan = controller.agents[0].plan
        self.assertIsNotNone(plan)
        self.assertEqual(plan.endpoint_tier, 0)
        overlay = apply_search_recovery_guidance(base, collision, controller)
        controller.observe_activation(base, overlay)
        self.assertNotEqual(overlay.assignment_for(0).tracking_waypoint, base.assignment_for(0).tracking_waypoint)
        self.assertEqual(overlay.executor_assignment, base.executor_assignment)
        endpoint_agent = replace(state.agents[0], position=plan.local_endpoint)
        endpoint_state = replace(state_at(state, 3), agents=(endpoint_agent, *state.agents[1:]))
        controller.observe_transition(stage_before=0, planning_state_before=collision, planning_state_after=endpoint_state,
                                      installed_guidance_before=overlay, collision_flags=(False, False, False, False))
        self.assertTrue(controller.force_refresh_requested)
        controller.prepare_next_guidance(endpoint_state, base)
        summary = controller.summary()
        self.assertEqual(summary["local_connector_reached_count"], 1)
        self.assertEqual(summary["graph_reconnect_attempt_count"], 1)
        self.assertEqual(summary["graph_reconnect_success_count"], 1)


if __name__ == "__main__":
    unittest.main()
