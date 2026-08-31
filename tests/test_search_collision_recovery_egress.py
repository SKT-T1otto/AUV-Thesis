from __future__ import annotations

import unittest

from chapter3_bser.experiments.phase1c_prrac.search_collision_recovery import SearchRecoveryVariant, build_search_recovery_controller, select_egress_route
from tests.bser_test_utils import synthetic_state
from tests.search_collision_recovery_test_support import guidance, state_at


class EgressTests(unittest.TestCase):
    def test_candidate_is_deterministic_known_free_reachable(self):
        state=synthetic_state(); a,_=select_egress_route(state,0,"candidate",(1.5,1.5,1.0)); b,_=select_egress_route(state,0,"candidate",(1.5,1.5,1.0))
        self.assertEqual(a,b); self.assertTrue(state.occupancy.free_mask[a.endpoint_cell_index])

    def test_c2_does_refresh_before_egress(self):
        c=build_search_recovery_controller(SearchRecoveryVariant.S2A_C2_EGRESS_ROUTE); base=guidance(); state=synthetic_state()
        first=state_at(state,1); c.observe_transition(stage_before=0,planning_state_after=first,collision_flags=(True,False,False,False)); c.prepare_next_guidance(first,base)
        self.assertEqual(c.summary()["egress_attempt_count_agent_0"],0)
        second=state_at(state,2); c.observe_transition(stage_before=0,planning_state_after=second,collision_flags=(True,False,False,False)); c.prepare_next_guidance(second,base)
        self.assertEqual(c.summary()["egress_attempt_count_agent_0"],1)


if __name__ == "__main__": unittest.main()
