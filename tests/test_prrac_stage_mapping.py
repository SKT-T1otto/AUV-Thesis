import unittes

from chapter3_bser.experiments.phase1c_common import TransitionPhase
from chapter3_bser.models.prrac.stage_mapping import (
    PRRACStage,
    transition_phase_to_prrac_stage,
)


class PRRACStageMappingTests(unittest.TestCase):
    def test_complete_fixed_mapping(self):
        expected = {
            TransitionPhase.PRE_FOUND: PRRACStage.SEARCH,
            TransitionPhase.POST_FOUND: PRRACStage.INTERCEPT,
            TransitionPhase.CONTACT: PRRACStage.HOLD,
            TransitionPhase.HOLD: PRRACStage.HOLD,
            TransitionPhase.SUCCESS: PRRACStage.HOLD,
        }
        self.assertEqual(
            {phase: transition_phase_to_prrac_stage(phase) for phase in TransitionPhase},
            expected,
        )


if __name__ == "__main__":
    unittest.main()
