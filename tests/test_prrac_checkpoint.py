from pathlib import Path
import tempfile
import unittest

import torch

from chapter3_bser.experiments.phase1c_prrac.replay_adapter import PRRACReplayAdapter
from chapter3_bser.experiments.phase1c_prrac.train_phase1c_prrac import (
    INCOMPATIBLE_MESSAGE,
    _load_checkpoint,
    _save_checkpoint,
)
from chapter3_bser.models.prrac.prrac_maddpg import PRRACMADDPG


def _config():
    return {
        "seed": 7,
        "profile": "M20_MOVING_UNKNOWN_MULTI",
        "execution_runtime_revision": "dynamic_public_intercept_v2_1",
        "architecture": {
            "num_stages": 3,
            "encoder_hidden_dim": 8,
            "expert_hidden_dim": 8,
            "critic_hidden_dim": 16,
            "router_temperature": 1.0,
            "gate_initial_mean": 0.75,
            "alignment_scale_init": 1.0,
        },
        "loss": {
            "router_ce_coef": 0.05,
            "gate_conflict_coef": 0.01,
            "gate_entropy_coef": 0.001,
            "residual_action_reg": 0.01,
        },
        "reward": {"schema": "bser.phase1c.execution_reward.v2"},
        "replay": {"schema": "bser.phase1c.phase_aware_replay.v1"},
    }


def _learner(config):
    return PRRACMADDPG(architecture=config["architecture"], loss=config["loss"])


class PRRACCheckpointTests(unittest.TestCase):
    def test_algorithm_optimizer_replay_roundtrip_and_mismatch_rejection(self):
        config = _config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            learner = _learner(config)
            replay = PRRACReplayAdapter(8, config={})
            path = _save_checkpoint(
                learner, replay, root, config, 1,
                global_step=0, update_step=0, replay_sample_count=0,
                optimizer_update_count=0, episode_rows=[], execution_rows=[], prrac_rows=[]
            )
            restored = _learner(config)
            restored_replay = PRRACReplayAdapter(8, config={})
            payload = _load_checkpoint(path, restored, restored_replay, config)
            self.assertEqual(payload["completed_episode"], 1)
            for left, right in zip(learner.policy_snapshot(), restored.policy_snapshot()):
                for key in left:
                    torch.testing.assert_close(left[key], right[key])
            mismatch = dict(config)
            mismatch["seed"] = 9
            with self.assertRaisesRegex(ValueError, "config hash"):
                _load_checkpoint(path, _learner(config), PRRACReplayAdapter(8), mismatch)
            architecture_mismatch = dict(config)
            architecture_mismatch["architecture"] = dict(config["architecture"])
            architecture_mismatch["architecture"]["encoder_hidden_dim"] = 10
            with self.assertRaisesRegex(ValueError, "architecture config"):
                _load_checkpoint(
                    path,
                    _learner(architecture_mismatch),
                    PRRACReplayAdapter(8),
                    architecture_mismatch,
                )

    def test_v1_and_v2_are_explicitly_rejected(self):
        config = _config()
        with tempfile.TemporaryDirectory() as directory:
            for schema in ("bser.phase1c.training_state.v1", "bser.phase1c.training_state.v2"):
                path = Path(directory) / f"{schema[-2:]}.pt"
                torch.save({"schema": schema}, path)
                with self.assertRaisesRegex(ValueError, "v1/v2") as caught:
                    _load_checkpoint(path, _learner(config), PRRACReplayAdapter(8), config)
                self.assertEqual(str(caught.exception), INCOMPATIBLE_MESSAGE)


if __name__ == "__main__":
    unittest.main()
