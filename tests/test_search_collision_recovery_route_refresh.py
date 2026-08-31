from __future__ import annotations

import unittest

from chapter3_bser.experiments.phase1c_prrac.search_collision_recovery import SearchRecoveryVariant, build_search_recovery_controller
from tests.bser_test_utils import synthetic_state
from tests.search_collision_recovery_test_support import guidance, state_at


class RouteRefreshTests(unittest.TestCase):
    def test_refresh_once_per_continuous_collision(self):
        c=build_search_recovery_controller(SearchRecoveryVariant.S2A_C1_ROUTE_REFRESH); base=guidance(); state=synthetic_state()
        for step in (1,2,3):
            after=state_at(state,step); c.observe_transition(stage_before=0,planning_state_after=after,collision_flags=(True,False,False,False)); c.prepare_next_guidance(after,base)
        self.assertEqual(c.summary()["route_refresh_attempt_count_agent_0"],1)
        clear=state_at(state,4); c.observe_transition(stage_before=0,planning_state_after=clear,collision_flags=(False,False,False,False)); c.prepare_next_guidance(clear,base)
        again=state_at(state,5); c.observe_transition(stage_before=0,planning_state_after=again,collision_flags=(True,False,False,False)); c.prepare_next_guidance(again,base)
        self.assertEqual(c.summary()["route_refresh_attempt_count_agent_0"],2)


if __name__ == "__main__": unittest.main()
