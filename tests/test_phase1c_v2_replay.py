from __future__ import annotations

import unittest

import torch

from chapter3_bser.experiments.phase1c_common import Phase1CTransitionMetadata
from chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.phase_aware_replay import (
    PhaseAwareReplayBuffer,
)


def metadata(phase: str, episode: int, step: int):
    values = {
        "pre": dict(task_found=False, contact=False, full_hold=False, mission_complete=False),
        "post": dict(task_found=True, contact=False, full_hold=False, mission_complete=False),
        "contact": dict(task_found=True, contact=True, full_hold=False, mission_complete=False),
        "hold": dict(task_found=True, contact=True, full_hold=True, mission_complete=False),
        "success": dict(task_found=True, contact=True, full_hold=True, mission_complete=True),
    }[phase]
    return Phase1CTransitionMetadata.build(
        episode_id=episode,
        episode_index=episode,
        step=step,
        executor_target_assigned=values["task_found"],
        hold_counter=step if values["full_hold"] else 0,
        **values,
    )


def transition(value: float, meta):
    obs = tuple(torch.full((28,), value + agent) for agent in range(4))
    action = torch.full((4, 3), value)
    rewards = torch.tensor([value, value + 1, value + 2, value + 3])
    next_obs = tuple(item + 0.5 for item in obs)
    dones = (False, False, False, False)
    success = tuple(meta.mission_complete for _ in range(4))
    return obs, action, rewards, next_obs, dones, success, meta


def buffer(**overrides):
    config = {
        "pre_found_fraction": 0.40,
        "post_found_fraction": 0.30,
        "contact_hold_fraction": 0.20,
        "success_tail_fraction": 0.10,
        "success_tail_steps": 5,
        "rare_stratum_with_replacement": True,
    }
    config.update(overrides)
    return PhaseAwareReplayBuffer(
        256,
        4,
        (28, 28, 28, 28),
        (3, 3, 3, 3),
        config=config,
        generator_seed=123,
    )


def push(buf, phase: str, episode: int, count: int, start: int = 0):
    indices = []
    for offset in range(count):
        meta = metadata(phase, episode, start + offset + 1)
        indices.append(buf.push(*transition(float(offset), meta)))
    return indices


class Phase1CV2ReplayTests(unittest.TestCase):
    def test_phase_aware_sample_preserves_eight_item_maddpg_contract(self) -> None:
        replay = buffer()
        push(replay, "pre", 1, 30)
        push(replay, "post", 2, 30)
        push(replay, "contact", 3, 30)
        push(replay, "post", 4, 10)
        replay.finalize_episode(4, success=True)
        sample = replay.sample(20, norm_rews=False)
        self.assertEqual(len(sample), 8)
        self.assertEqual(len(sample[0]), 4)
        self.assertEqual(len(sample[1]), 4)
        self.assertEqual(len(sample[2]), 4)
        self.assertEqual(tuple(sample[0][0].shape), (20, 28))
        self.assertEqual(tuple(sample[1][0].shape), (20, 3))
        self.assertEqual(tuple(sample[5].shape), (20,))
        self.assertEqual(tuple(sample[6].shape), (20,))
        self.assertEqual(sample[7].dtype, torch.bool)
        self.assertEqual(
            replay.last_sample_diagnostics["actual_counts"],
            {
                "pre_found": 8,
                "post_found": 6,
                "contact_hold": 4,
                "success_tail": 2,
            },
        )
        self.assertTrue(torch.isfinite(sample[5]).all().item())

    def test_empty_strata_are_redistributed_and_reported(self) -> None:
        replay = buffer()
        push(replay, "pre", 1, 20)
        sample = replay.sample(10, norm_rews=False)
        diagnostics = replay.last_sample_diagnostics
        self.assertEqual(tuple(sample[5].shape), (10,))
        self.assertEqual(diagnostics["actual_counts"]["pre_found"], 10)
        self.assertEqual(
            set(diagnostics["fallback"]["empty_strata"]),
            {"post_found", "contact_hold", "success_tail"},
        )

    def test_success_tail_round_trip_and_duplicate_priority_update(self) -> None:
        replay = buffer()
        indices = push(replay, "post", 9, 8)
        self.assertEqual(replay.finalize_episode(9, success=True), 5)
        self.assertEqual(replay.phase_counts()["success_tail"], 5)
        target = indices[-1]
        replay.update_priorities([target, target], [1.0, 5.0])
        self.assertAlmostEqual(replay.priorities[target].item(), 5.00001, places=4)

        state = replay.state_dict()
        restored = buffer()
        restored.load_state_dict(state)
        self.assertEqual(restored.phase_counts(), replay.phase_counts())
        self.assertEqual(restored.success_tail_mark_count, 5)
        self.assertTrue(torch.equal(restored.success_tail_flags, replay.success_tail_flags))
        self.assertTrue(torch.allclose(restored.priorities, replay.priorities))

    def test_ring_buffer_wraparound_keeps_batch_contract(self) -> None:
        replay = PhaseAwareReplayBuffer(
            16,
            4,
            (28, 28, 28, 28),
            (3, 3, 3, 3),
            config={
                "pre_found_fraction": 0.40,
                "post_found_fraction": 0.30,
                "contact_hold_fraction": 0.20,
                "success_tail_fraction": 0.10,
                "success_tail_steps": 5,
                "rare_stratum_with_replacement": True,
            },
            generator_seed=123,
        )
        push(replay, "pre", 1, 40)
        self.assertEqual(len(replay), 16)
        sample = replay.sample(16, norm_rews=False)
        self.assertEqual(tuple(sample[5].shape), (16,))
        self.assertTrue(torch.isfinite(sample[5]).all().item())


if __name__ == "__main__":
    unittest.main()
