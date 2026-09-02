from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from chapter3_bser.experiments.phase1c_prrac.replay_adapter import (
    PRRACReplayAdapter,
)
from chapter3_bser.experiments.phase1c_common import Phase1CTransitionMetadata
from chapter3_bser.experiments.phase1c_prrac.transition_protocol import (
    PRRACTransitionMetadata,
)
from chapter3_bser.experiments.phase1c_prrac.train_phase1c_prrac import (
    ROOT,
    _collect_episode,
    _load_config,
    _load_checkpoint,
    _save_checkpoint,
    _verify_checkpoint_roundtrip,
)
from chapter3_bser.models.prrac.prrac_maddpg import PRRACMADDPG
from chapter3_bser.models.prrac.stage_mapping import PRRACStage
from chapter3_bser.models.search_value_head import (
    SEARCH_FEATURE_DIM,
    SearchValueHead,
    future_found_labels,
)
from tests.test_prrac_cross_platform_runtime import _worker_jobs


def _config(search_value=None):
    config = {
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
    if search_value is not None:
        config["search_value"] = dict(search_value)
    return config


def _learner(config):
    return PRRACMADDPG(
        architecture=config["architecture"],
        loss=config["loss"],
        search_value=config.get("search_value"),
    )


class SearchValueHeadTests(unittest.TestCase):
    def test_config_defaults_disabled_and_opt_in_file_enabled(self):
        default = _load_config(
            ROOT / "configs/chapter3/bser_phase1c_prrac_train.json"
        )
        enabled = _load_config(
            ROOT / "configs/chapter3/bser_phase1c_search_value.json"
        )
        self.assertFalse(default["search_value"]["enabled"])
        self.assertTrue(enabled["search_value"]["enabled"])
        self.assertEqual(enabled["search_value"]["hidden_dim"], 128)
        self.assertEqual(enabled["search_value"]["horizon"], 50)
        self.assertEqual(enabled["search_value"]["loss_weight"], 0.05)
        self.assertEqual(enabled["search_value"]["threshold"], 0.5)

    def test_shape(self):
        head = SearchValueHead(feature_dim=SEARCH_FEATURE_DIM, hidden_dim=32)
        result = head(torch.randn(11, SEARCH_FEATURE_DIM))
        self.assertEqual(tuple(result.shape), (11, 1))
        self.assertTrue(torch.all((result >= 0.0) & (result <= 1.0)))

    def test_future_found_horizon_labels(self):
        labels = future_found_labels(
            [False, False, False, True, True, True], horizon=3
        )
        np.testing.assert_array_equal(
            labels, np.asarray([0, 1, 1, 1, 0, 0], dtype=np.float32)
        )

    def test_old_checkpoint_load_initializes_missing_head(self):
        old_config = _config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_path = _save_checkpoint(
                _learner(old_config),
                PRRACReplayAdapter(8),
                root,
                old_config,
                1,
                global_step=0,
                update_step=0,
                replay_sample_count=0,
                optimizer_update_count=0,
                episode_rows=[],
                execution_rows=[],
                prrac_rows=[],
            )
            payload = torch.load(old_path, map_location="cpu", weights_only=True)
            payload["metadata"].pop("search_value", None)
            payload["metadata"].pop("search_value_parameter_count", None)
            payload["prrac_training_state"].pop("search_value", None)
            payload["prrac_training_state"].pop("search_value_head", None)
            payload["prrac_training_state"].pop("search_value_optimizer", None)
            legacy_path = root / "legacy_without_search_value.pt"
            torch.save(payload, legacy_path)

            enabled = {
                "enabled": True,
                "hidden_dim": 32,
                "horizon": 50,
                "loss_weight": 0.05,
                "threshold": 0.5,
            }
            new_config = _config(enabled)
            learner = _learner(new_config)
            replay = PRRACReplayAdapter(8, search_value_config=enabled)
            restored = _load_checkpoint(
                legacy_path, learner, replay, new_config
            )
            self.assertTrue(restored["search_value_head_initialized"])
            self.assertTrue(learner.search_value_head_initialized)
            self.assertIsNotNone(learner.search_value_head)
            self.assertFalse(bool(replay.search_value_valid.any()))

    def test_enabled_checkpoint_roundtrip_preserves_auxiliary_state(self):
        search_value = {
            "enabled": True,
            "hidden_dim": 16,
            "horizon": 50,
            "loss_weight": 0.05,
            "threshold": 0.5,
        }
        config = _config(search_value)
        with tempfile.TemporaryDirectory() as directory:
            path = _save_checkpoint(
                _learner(config),
                PRRACReplayAdapter(8, search_value_config=search_value),
                Path(directory),
                config,
                1,
                global_step=0,
                update_step=0,
                replay_sample_count=0,
                optimizer_update_count=0,
                episode_rows=[],
                execution_rows=[],
                prrac_rows=[],
            )
            self.assertTrue(_verify_checkpoint_roundtrip(path, config))

    def test_disabled_mode_preserves_original_policy_and_replay_shape(self):
        torch.manual_seed(19)
        original = _learner(_config())
        torch.manual_seed(19)
        disabled_config = _config(
            {
                "enabled": False,
                "hidden_dim": 128,
                "horizon": 50,
                "loss_weight": 0.05,
                "threshold": 0.5,
            }
        )
        disabled = _learner(disabled_config)
        self.assertIsNone(original.search_value_head)
        self.assertIsNone(disabled.search_value_head)
        for left, right in zip(original.policy_snapshot(), disabled.policy_snapshot()):
            for key in left:
                np.testing.assert_array_equal(left[key], right[key])
        replay_state = PRRACReplayAdapter(
            8, search_value_config=disabled_config["search_value"]
        ).state_dict()
        self.assertNotIn("search_features", replay_state)
        self.assertNotIn("future_found", replay_state)

    def test_enabling_head_does_not_change_actor_parameter_count(self):
        disabled = _learner(_config())
        enabled = _learner(
            _config(
                {
                    "enabled": True,
                    "hidden_dim": 128,
                    "horizon": 50,
                    "loss_weight": 0.05,
                    "threshold": 0.5,
                }
            )
        )
        disabled_count = sum(
            parameter.numel()
            for agent in disabled.agents
            for parameter in agent.actor.parameters()
        )
        enabled_count = sum(
            parameter.numel()
            for agent in enabled.agents
            for parameter in agent.actor.parameters()
        )
        self.assertEqual(disabled_count, enabled_count)
        self.assertEqual(disabled.OBS_DIMS, enabled.OBS_DIMS)
        self.assertEqual(disabled.ACTION_DIMS, enabled.ACTION_DIMS)
        self.assertEqual(disabled.CRITIC_DIM, enabled.CRITIC_DIM)

    def test_auxiliary_update_changes_only_search_head(self):
        search_value = {
            "enabled": True,
            "hidden_dim": 16,
            "horizon": 50,
            "loss_weight": 0.05,
            "threshold": 0.5,
        }
        learner = _learner(_config(search_value))
        replay = PRRACReplayAdapter(8, search_value_config=search_value)
        metadata = PRRACTransitionMetadata(
            base=Phase1CTransitionMetadata.build(
                episode_id=0,
                episode_index=0,
                step=1,
                task_found=False,
                executor_target_assigned=False,
                contact=False,
                full_hold=False,
                hold_counter=0,
                mission_complete=False,
            ),
            stage_before=PRRACStage.SEARCH,
            stage_after=PRRACStage.SEARCH,
        )
        for index in range(4):
            replay.push(
                tuple(torch.zeros(28) for _ in range(4)),
                torch.zeros(4, 3),
                torch.zeros(4),
                tuple(torch.zeros(28) for _ in range(4)),
                (False,) * 4,
                (False,) * 4,
                metadata,
                search_features=torch.randn(3, SEARCH_FEATURE_DIM),
                future_found=torch.full((3, 1), float(index % 2)),
            )
        batch = replay.sample(4, norm_rews=False, device="cpu")
        actor_before = learner.policy_snapshot()
        head_before = [
            parameter.detach().clone()
            for parameter in learner.search_value_head.parameters()
        ]
        result = learner.update_search_value(batch)
        self.assertTrue(result["search_value_optimizer_updated"])
        self.assertGreater(result["search_value_sample_count"], 0)
        self.assertIsNotNone(result["search_value_loss"])
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(
                    head_before, learner.search_value_head.parameters()
                )
            )
        )
        for expected, actual in zip(actor_before, learner.policy_snapshot()):
            for key in expected:
                np.testing.assert_array_equal(expected[key], actual[key])

    def test_worker_emits_side_channel_features_without_actor_shape_change(self):
        job = _worker_jobs(1)[0]
        job["search_value"] = {
            "enabled": True,
            "hidden_dim": 16,
            "horizon": 50,
            "loss_weight": 0.05,
            "threshold": 0.5,
        }
        _, transitions, _, _ = _collect_episode(job)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(len(transitions[0]), 9)
        self.assertEqual(
            tuple(value.shape for value in transitions[0][0]), ((28,),) * 4
        )
        self.assertEqual(transitions[0][7].shape, (3, SEARCH_FEATURE_DIM))
        self.assertEqual(transitions[0][8].shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
