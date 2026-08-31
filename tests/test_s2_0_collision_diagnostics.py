from __future__ import annotations

import unittest

import numpy as np

from chapter3_bser.experiments.phase1c_prrac.execution_continuity.diagnostics import ExecutionContinuityDiagnostics
from chapter3_bser.experiments.phase1c_prrac.execution_continuity.types import ExecutionVariant, ResidualSuppressionDiagnostics
from chapter3_bser.experiments.phase1c_prrac.search_continuity.diagnostics import SearchContinuityDiagnostics
from tests.search_continuity_test_support import Guidance, outputs, state


class CollisionDiagnosticsTests(unittest.TestCase):
    def test_search_count_streak_first_last_and_zero(self):
        diagnostics=SearchContinuityDiagnostics(); diagnostics.begin_episode(state())
        for step,flags in ((1,(True,False,False,False)),(2,(True,True,False,False)),(3,(False,False,False,False))):
            diagnostics.observe_transition(stage_before=0,stage_after=0,installed_guidance=Guidance(),planning_state_before=state(step-1),planning_state_after=state(step),collision_flags=flags,raw_actions=np.zeros((4,3)),applied_actions=np.zeros((4,3)),actor_outputs=outputs())
        row=diagnostics.summary(found=False,max_steps=10)
        self.assertEqual(row["searcher_collision_count_pre_found_total"],3)
        self.assertEqual(row["searcher_collision_max_streak_pre_found"],2)
        self.assertEqual((row["searcher_first_collision_step_pre_found"],row["searcher_last_collision_step_pre_found"]),(1,2))
        self.assertEqual(row["searcher_collision_agent_count_pre_found"],2)
        empty=SearchContinuityDiagnostics(); empty.begin_episode(state()); zero=empty.summary(found=False,max_steps=1)
        self.assertIsNone(zero["searcher_first_collision_step_pre_found"]); self.assertEqual(zero["searcher_collision_max_streak_pre_found"],0)

    def test_post_found_executor_and_terminal_streaks(self):
        diagnostics=ExecutionContinuityDiagnostics(ExecutionVariant.B1_ATOMIC_LAST_VALID)
        suppression=ResidualSuppressionDiagnostics(False,3,0,0,"")
        for step,collision,active in ((5,True,False),(6,True,False),(7,False,True)):
            diagnostics.observe_step(post_found=True,plan=None,detection=None,suppression=suppression,legacy_route_active=active,executor_collision=collision,transition_step=step)
        row=diagnostics.summary()
        self.assertEqual(row["executor_collision_count_post_found"],2)
        self.assertEqual(row["executor_collision_max_streak_post_found"],2)
        self.assertEqual(row["executor_first_collision_step_post_found"],5)
        self.assertEqual(row["post_found_route_inactive_max_streak"],2)
        self.assertEqual(row["post_found_route_inactive_terminal_streak"],0)


if __name__ == "__main__": unittest.main()
