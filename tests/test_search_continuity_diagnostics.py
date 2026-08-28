import unittest

import numpy as np

from chapter3_bser.experiments.phase1c_prrac.search_continuity import SearchContinuityDiagnostics
from tests.search_continuity_test_support import Guidance, outputs, state


class SearchContinuityDiagnosticsTests(unittest.TestCase):
    def test_agent_step_denominators_and_public_deltas(self):
        diagnostic = SearchContinuityDiagnostics()
        before = state()
        after = state(step=1, offset=1.0, known=(True, True), entropy=0.75, peak=0.6)
        diagnostic.begin_episode(before)
        raw = np.ones((4, 3), dtype=np.float32)
        diagnostic.observe_transition(
            stage_before=0, stage_after=0, installed_guidance=Guidance(),
            planning_state_before=before, planning_state_after=after,
            collision_flags=(True, False, False, False), raw_actions=raw,
            applied_actions=raw.copy(), actor_outputs=outputs(-0.5),
        )
        row = diagnostic.summary(found=False, max_steps=400)
        self.assertEqual(row["pre_found_step_count"], 1)
        self.assertEqual(row["searcher_route_active_rate_pre_found"], 1.0)
        self.assertEqual(row["searcher_route_active_rate_pre_found_agent_0"], 1.0)
        self.assertEqual(row["searcher_collision_count_pre_found_agent_0"], 1)
        self.assertEqual(row["map_known_fraction_gain_pre_found"], 0.5)
        self.assertEqual(row["searcher_residual_negative_alignment_rate_pre_found"], 1.0)

    def test_zero_denominators_are_null(self):
        diagnostic = SearchContinuityDiagnostics()
        diagnostic.begin_episode(state())
        row = diagnostic.summary(found=False, max_steps=400)
        self.assertIsNone(row["searcher_route_active_rate_pre_found"])
        self.assertIsNone(row["searcher_route_active_rate_pre_found_agent_0"])


if __name__ == "__main__":
    unittest.main()
