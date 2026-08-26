from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from chapter3_bser.experiments.phase1c_common import Phase1CTransitionMetadata
from chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.phase_aware_replay import (
    PhaseAwareReplayBuffer,
)
from chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.train_phase1c_v2 import (
    _apply_transitions,
    _load_checkpoint,
    _save_checkpoint,
)


class FakeAgent:
    def __init__(self) -> None:
        self.policy = torch.nn.Linear(28, 3)
        self.critic1 = torch.nn.Linear(124, 1)


class FakeMADDPG:
    def __init__(self) -> None:
        self.agents = [FakeAgent() for _ in range(4)]
        self.loaded = False

    def prep_training(self, device="cpu"):
        return None

    def prep_rollouts(self, device="cpu"):
        return None

    def update(self, sample, agent_i):
        with torch.no_grad():
            next(self.agents[agent_i].policy.parameters()).add_(0.001)
        error = sample[2][agent_i].abs().reshape(-1)
        return 1.0, 0.5, error

    def update_critic_only(self, sample, agent_i):
        error = sample[2][agent_i].abs().reshape(-1)
        return 1.0, error

    def update_all_targets(self, compute_diff=False):
        return None

    def training_state_dict(self):
        return {
            "schema": "fake.maddpg.v1",
            "weights": [agent.policy.state_dict() for agent in self.agents],
        }

    def load_training_state_dict(self, state):
        self.loaded = True
        for agent, weights in zip(self.agents, state["weights"]):
            agent.policy.load_state_dict(weights)


def config():
    return {
        "seed": 2729,
        "profile": "M20_MOVING_UNKNOWN_MULTI",
        "bser_integration_version": "bser.control_context.v1",
        "reward": {
            "schema": "bser.phase1c.execution_reward.v2",
            "enabled": True,
            "freeze_searchers_after_found": True,
            "preserve_discovery_reward": True,
            "contact_entry_bonus": 0.25,
            "hold_increment_bonus": 0.2,
            "terminal_success_bonus": 2.0,
            "reward_clip": 3.0,
            "executor_id": 3,
            "searcher_ids": [0, 1, 2],
        },
        "replay": {
            "schema": "bser.phase1c.phase_aware_replay.v1",
            "pre_found_fraction": 0.4,
            "post_found_fraction": 0.3,
            "contact_hold_fraction": 0.2,
            "success_tail_fraction": 0.1,
            "rare_stratum_with_replacement": True,
            "success_tail_steps": 4,
            "success_priority_multiplier": 2.0,
            "alpha": 0.6,
            "beta_start": 0.4,
            "beta_frames": 100,
        },
    }


def meta(index: int):
    found = index >= 2
    contact = index >= 5
    full_hold = index >= 7
    success = index == 9
    return Phase1CTransitionMetadata.build(
        episode_id=0,
        episode_index=0,
        step=index + 1,
        task_found=found,
        executor_target_assigned=found,
        contact=contact,
        full_hold=full_hold,
        hold_counter=max(0, index - 6),
        mission_complete=success,
    )


def transitions():
    rows = []
    for index in range(10):
        metadata = meta(index)
        obs = tuple(torch.full((28,), float(index)) for _ in range(4))
        action = torch.full((4, 3), 0.1)
        reward = torch.tensor([0.0, 0.0, 0.0, 1.0 + index])
        next_obs = tuple(value + 1.0 for value in obs)
        done = tuple(metadata.mission_complete for _ in range(4))
        success = done
        rows.append((obs, action, reward, next_obs, done, success, metadata))
    return rows


def replay(config_value):
    return PhaseAwareReplayBuffer(
        64,
        4,
        (28, 28, 28, 28),
        (3, 3, 3, 3),
        config=config_value["replay"],
        generator_seed=5,
    )


class Phase1CV2TrainingSmokeTests(unittest.TestCase):
    def test_training_update_replay_checkpoint_and_resume_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            cfg = config()
            learner = FakeMADDPG()
            before = next(learner.agents[0].policy.parameters()).detach().clone()
            replay_buffer = replay(cfg)
            update = _apply_transitions(
                learner,
                replay_buffer,
                transitions(),
                {"episode_id": 0, "success": True},
                {
                    "warmup_steps": 4,
                    "update_frequency": 1,
                    "batch_size": 4,
                    "updates_per_train": 1,
                    "policy_delay": 2,
                },
                global_step=0,
                update_step=0,
                device="cpu",
            )
            self.assertGreater(update["replay_sample_count"], 0)
            self.assertGreater(update["optimizer_update_count"], 0)
            self.assertEqual(update["success_tail_marked"], 4)
            self.assertFalse(
                torch.equal(
                    before,
                    next(learner.agents[0].policy.parameters()).detach(),
                )
            )

            checkpoint_dir = tmp_path / "checkpoints"
            checkpoint_dir.mkdir()
            path = _save_checkpoint(
                learner,
                replay_buffer,
                checkpoint_dir,
                cfg,
                1,
                global_step=update["global_step"],
                update_step=update["update_step"],
                replay_sample_count=update["replay_sample_count"],
                optimizer_update_count=update["optimizer_update_count"],
                episode_rows=[{"episode": 1}],
                execution_rows=[{"episode": 1}],
            )
            self.assertTrue(path.is_file())
            restored_learner = FakeMADDPG()
            restored_replay = replay(cfg)
            payload = _load_checkpoint(path, restored_learner, restored_replay, cfg)
            self.assertEqual(payload["schema"], "bser.phase1c.training_state.v2")
            self.assertIs(restored_learner.loaded, True)
            self.assertEqual(restored_replay.phase_counts(), replay_buffer.phase_counts())

    def test_v1_checkpoint_is_explicitly_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1.pt"
            torch.save({"schema": "bser.phase1c.training_state.v1"}, path)
            with self.assertRaisesRegex(ValueError, "intentionally incompatible"):
                _load_checkpoint(path, FakeMADDPG(), replay(config()), config())


if __name__ == "__main__":
    unittest.main()
