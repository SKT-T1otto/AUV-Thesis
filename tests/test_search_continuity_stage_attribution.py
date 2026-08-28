import unittest

import numpy as np

from chapter3_bser.experiments.phase1c_prrac.search_continuity import SearchContinuityDiagnostics
from tests.search_continuity_test_support import Guidance, outputs, state


class SearchContinuityStageAttributionTests(unittest.TestCase):
    def test_discovery_transition_and_installed_guidance_are_authoritative(self):
        diagnostic = SearchContinuityDiagnostics()
        before, after = state(), state(step=1)
        diagnostic.begin_episode(before)
        actions = np.zeros((4, 3), dtype=np.float32)
        diagnostic.observe_transition(
            stage_before=0, stage_after=1, installed_guidance=Guidance(reachable=False, hold=True),
            planning_state_before=before, planning_state_after=after,
            collision_flags=(False, True, False, False), raw_actions=actions,
            applied_actions=actions, actor_outputs=outputs(),
        )
        diagnostic.observe_transition(
            stage_before=1, stage_after=1, installed_guidance=Guidance(),
            planning_state_before=after, planning_state_after=state(step=2),
            collision_flags=(True, True, True, False), raw_actions=actions,
            applied_actions=actions, actor_outputs=outputs(),
        )
        row = diagnostic.summary(found=True, max_steps=400)
        self.assertEqual(row["pre_found_step_count"], 1)
        self.assertEqual(row["found_step"], 1)
        self.assertEqual(row["searcher_hold_step_count_pre_found"], 3)
        self.assertEqual(row["searcher_collision_count_pre_found"], 1)


if __name__ == "__main__":
    unittest.main()
