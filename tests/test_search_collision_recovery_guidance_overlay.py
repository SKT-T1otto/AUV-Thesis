from __future__ import annotations

import unittest

from chapter3_bser.experiments.phase1c_prrac.search_collision_recovery import SearchRecoveryVariant, apply_search_recovery_guidance, build_search_recovery_controller
from tests.bser_test_utils import synthetic_state
from tests.search_collision_recovery_test_support import guidance, state_at


class GuidanceOverlayTests(unittest.TestCase):
    def test_c0_identity_and_c1_semantics(self):
        state=synthetic_state(); base=guidance()
        self.assertIs(apply_search_recovery_guidance(base,state,None),base)
        controller=build_search_recovery_controller(SearchRecoveryVariant.S2A_C1_ROUTE_REFRESH)
        after=state_at(state,1); controller.observe_transition(stage_before=0,planning_state_after=after,collision_flags=(True,False,False,False)); controller.prepare_next_guidance(after,base)
        overlaid=apply_search_recovery_guidance(base,after,controller)
        self.assertEqual(overlaid.allocation_hash,base.allocation_hash); self.assertEqual(overlaid.decision_reason,base.decision_reason)
        self.assertEqual(overlaid.assignment_for(0).assignment_id,base.assignment_for(0).assignment_id)
        self.assertIs(overlaid.assignment_for(3),base.assignment_for(3))

    def test_post_found_is_noop(self):
        state=state_at(synthetic_state(),2,found=True); base=guidance(); controller=build_search_recovery_controller(SearchRecoveryVariant.S2A_C2_EGRESS_ROUTE)
        self.assertIs(apply_search_recovery_guidance(base,state,controller),base)


if __name__ == "__main__": unittest.main()
