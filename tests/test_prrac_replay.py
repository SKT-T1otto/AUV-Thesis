import unittest

import torch

from chapter3_bser.experiments.phase1c_common import Phase1CTransitionMetadata, TransitionPhase
from chapter3_bser.experiments.phase1c_prrac.replay_adapter import PRRACReplayAdapter
from chapter3_bser.experiments.phase1c_prrac.transition_protocol import PRRACTransitionMetadata
from chapter3_bser.models.prrac.stage_mapping import PRRACStage, transition_phase_to_prrac_stage


def _metadata(index: int, before=TransitionPhase.PRE_FOUND, after=TransitionPhase.POST_FOUND):
    found = after != TransitionPhase.PRE_FOUND
    contact = after in {TransitionPhase.CONTACT, TransitionPhase.HOLD, TransitionPhase.SUCCESS}
    hold = after in {TransitionPhase.HOLD, TransitionPhase.SUCCESS}
    success = after == TransitionPhase.SUCCESS
    base = Phase1CTransitionMetadata.build(
        episode_id=1,
        episode_index=1,
        step=index + 1,
        task_found=found,
        executor_target_assigned=found,
        contact=contact,
        full_hold=hold,
        hold_counter=int(hold),
        mission_complete=success,
    )
    return PRRACTransitionMetadata(
        base, transition_phase_to_prrac_stage(before), transition_phase_to_prrac_stage(after)
    )


def _push(replay, index, metadata):
    obs = tuple(torch.full((28,), float(index + agent)) for agent in range(4))
    action = torch.full((4, 3), 0.1)
    reward = torch.arange(4, dtype=torch.float32)
    replay.push(obs, action, reward, tuple(value + 1 for value in obs), (False,) * 4, (False,) * 4, metadata)


class PRRACReplayTests(unittest.TestCase):
    def test_ring_replacement_roundtrip_generator_and_priority(self):
        config = {
            "pre_found_fraction": 0.4,
            "post_found_fraction": 0.3,
            "contact_hold_fraction": 0.2,
            "success_tail_fraction": 0.1,
            "rare_stratum_with_replacement": True,
            "success_tail_steps": 3,
        }
        replay = PRRACReplayAdapter(8, config=config, generator_seed=91)
        phases = [
            (TransitionPhase.PRE_FOUND, TransitionPhase.PRE_FOUND),
            (TransitionPhase.PRE_FOUND, TransitionPhase.POST_FOUND),
            (TransitionPhase.POST_FOUND, TransitionPhase.CONTACT),
            (TransitionPhase.CONTACT, TransitionPhase.HOLD),
        ]
        for index in range(20):
            before, after = phases[index % len(phases)]
            _push(replay, index, _metadata(index, before, after))
        self.assertEqual(len(replay), 8)
        state = replay.state_dict()
        restored = PRRACReplayAdapter(8, config=config, generator_seed=1)
        restored.load_state_dict(state)
        first = replay.sample(8, norm_rews=False)
        second = restored.sample(8, norm_rews=False)
        torch.testing.assert_close(first.indices, second.indices)
        torch.testing.assert_close(first.stage_before, second.stage_before)
        torch.testing.assert_close(first.stage_after, second.stage_after)
        self.assertEqual(tuple(first.obs[0].shape), (8, 28))
        self.assertTrue(torch.isfinite(first.importance_weights).all())
        restored.update_priorities(second.indices, torch.ones(8))


if __name__ == "__main__":
    unittest.main()
