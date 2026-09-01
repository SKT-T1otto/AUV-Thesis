from __future__ import annotations

from dataclasses import replace
import unittest

from chapter3_bser.experiments.phase1c_prrac.search_collision_recovery import SearchRecoveryVariantV2, build_search_recovery_controller
from tests.bser_test_utils import synthetic_state
from tests.search_collision_recovery_test_support import guidance, state_at


class LastCollisionFreeStateTests(unittest.TestCase):
    def test_clear_updates_collision_does_not_and_nonsearch_is_ignored(self):
        controller = build_search_recovery_controller(SearchRecoveryVariantV2.S2A1_C2_LOCAL_CONNECTOR)
        state = synthetic_state(); base = guidance(); clear = state_at(state, 1)
        controller.observe_transition(stage_before=0, planning_state_before=state, planning_state_after=clear,
                                      installed_guidance_before=base, collision_flags=(False, False, False, False))
        saved = controller.last_collision_free[0]
        self.assertEqual(saved.step, 1)
        collision_agent = replace(state.agents[0], position=(0.75, 0.5, 1.0))
        collision = replace(state_at(state, 2), agents=(collision_agent, *state.agents[1:]))
        controller.observe_transition(stage_before=0, planning_state_before=clear, planning_state_after=collision,
                                      installed_guidance_before=base, collision_flags=(True, False, False, False))
        self.assertEqual(controller.last_collision_free[0], saved)
        controller.observe_transition(stage_before=1, planning_state_before=collision, planning_state_after=state_at(state, 3),
                                      installed_guidance_before=base, collision_flags=(False, False, False, False))
        self.assertEqual(controller.last_collision_free[0], saved)

    def test_nonfinite_clear_position_is_rejected(self):
        controller = build_search_recovery_controller(SearchRecoveryVariantV2.S2A1_C1_FORCED_REFRESH)
        state = synthetic_state(); base = guidance()
        invalid_agent = replace(state.agents[0], position=(float("nan"), 0.5, 1.0))
        invalid = replace(state_at(state, 1), agents=(invalid_agent, *state.agents[1:]))
        controller.observe_transition(stage_before=0, planning_state_before=state, planning_state_after=invalid,
                                      installed_guidance_before=base, collision_flags=(False, False, False, False))
        self.assertIsNone(controller.last_collision_free[0])


if __name__ == "__main__":
    unittest.main()
