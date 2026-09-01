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


class S2A1DiagnosticCorrectnessTests(unittest.TestCase):
    def test_only_agent_one_counts_and_global_sums_match(self):
        controller = build_search_recovery_controller(SearchRecoveryVariantV2.S2A1_C1_FORCED_REFRESH)
        state = synthetic_state()
        base = guidance()
        after = state_at(state, 1)
        controller.observe_transition(
            stage_before=0, planning_state_before=state, planning_state_after=after,
            installed_guidance_before=base, collision_flags=(False, True, False, False),
        )
        controller.prepare_next_guidance(after, base)
        overlay = apply_search_recovery_guidance(base, after, controller)
        controller.observe_activation(base, overlay)
        summary = controller.summary()
        for field in (
            "search_recovery_entry_count", "route_refresh_attempt_count",
            "route_refresh_success_count", "recovery_effective_intervention_count",
        ):
            self.assertEqual(summary[f"{field}_agent_0"], 0)
            self.assertEqual(summary[f"{field}_agent_2"], 0)
            self.assertEqual(
                summary[field],
                sum(summary[f"{field}_agent_{agent_id}"] for agent_id in range(3)),
            )
        self.assertEqual(summary["search_recovery_entry_count_agent_1"], 1)
        self.assertEqual(summary["route_refresh_attempt_count_agent_1"], 1)

    def test_local_collision_audit_endpoint_exclusion_agent_counts_and_history(self):
        controller = build_search_recovery_controller(SearchRecoveryVariantV2.S2A1_C2_LOCAL_CONNECTOR)
        state = synthetic_state()
        base = guidance()
        clear = state_at(state, 1)
        controller.observe_transition(
            stage_before=0, planning_state_before=state, planning_state_after=clear,
            installed_guidance_before=base, collision_flags=(False, False, False, False),
        )
        moved_agent = replace(state.agents[0], position=(0.75, 0.5, 1.0))
        collision = replace(state_at(state, 2), agents=(moved_agent, *state.agents[1:]))
        controller.observe_transition(
            stage_before=0, planning_state_before=clear, planning_state_after=collision,
            installed_guidance_before=base, collision_flags=(True, False, False, False),
        )
        controller.prepare_next_guidance(collision, base)
        first_plan = controller.agents[0].plan
        self.assertIsNotNone(first_plan)
        first_overlay = apply_search_recovery_guidance(base, collision, controller)
        controller.observe_activation(base, first_overlay)

        collision2 = replace(collision, step=3)
        controller.observe_transition(
            stage_before=0, planning_state_before=collision, planning_state_after=collision2,
            installed_guidance_before=first_overlay, collision_flags=(True, False, False, False),
        )
        failure = [row for row in controller.planning_failure_rows() if row.get("final_failure_reason") == "LOCAL_EGRESS_COLLISION"][-1]
        self.assertEqual(failure["endpoint_cell_index"], first_plan.endpoint_cell_index)
        self.assertEqual(failure["selected_tier"], first_plan.endpoint_tier)
        self.assertEqual(failure["plan_source"], first_plan.source)
        self.assertEqual(failure["segment_audit"]["endpoint"], first_plan.public_segment_audit.endpoint)
        self.assertTrue(failure["guidance_changed"])

        controller.prepare_next_guidance(collision2, base)
        second_plan = controller.agents[0].plan
        self.assertIsNotNone(second_plan)
        self.assertNotEqual(second_plan.endpoint_cell_index, first_plan.endpoint_cell_index)
        second_overlay = apply_search_recovery_guidance(base, collision2, controller)
        controller.observe_activation(base, second_overlay)
        collision3 = replace(collision, step=4)
        controller.observe_transition(
            stage_before=0, planning_state_before=collision2, planning_state_after=collision3,
            installed_guidance_before=second_overlay, collision_flags=(True, False, False, False),
        )
        controller.observe_transition(
            stage_before=0, planning_state_before=collision3,
            planning_state_after=state_at(state, 5, found=True),
            installed_guidance_before=second_overlay, collision_flags=(False, False, False, False),
        )
        summary = controller.summary()
        self.assertEqual(summary["recovery_collision_count"], 2)
        self.assertEqual(summary["recovery_max_collision_streak"], 3)
        self.assertEqual(summary["local_connector_collision_count"], 2)
        self.assertEqual(summary["search_recovery_entry_count_agent_0"], 1)
        self.assertGreaterEqual(summary["route_refresh_attempt_count_agent_0"], 1)
        self.assertGreaterEqual(summary["local_connector_attempt_count_agent_0"], 2)
        self.assertGreaterEqual(summary["local_connector_plan_count_agent_0"], 2)
        self.assertEqual(summary["local_connector_collision_count_agent_0"], 2)
        self.assertEqual(summary["egress_failure_count_agent_0"], 2)
        self.assertEqual(summary["local_connector_collision_count_agent_1"], 0)
        activation_rows = controller.activation_rows()
        self.assertTrue(activation_rows)
        self.assertEqual(
            [(row["step"], row["agent_id"], row["attempt_id"]) for row in activation_rows],
            sorted((row["step"], row["agent_id"], row["attempt_id"]) for row in activation_rows),
        )
        required = {
            "step", "agent_id", "attempt_id", "recovery_mode",
            "base_tracking_waypoint", "overlay_tracking_waypoint",
            "base_path_hash", "overlay_path_hash", "guidance_changed",
            "recovery_plan_installed", "recovery_plan_source",
        }
        self.assertTrue(all(required <= set(row) for row in activation_rows))


if __name__ == "__main__":
    unittest.main()
