from pathlib import Path
import tempfile
import unittest

import torch

from chapter3_bser.experiments.phase1c_common import Phase1CTransitionMetadata, TransitionPhase
from chapter3_bser.experiments.phase1c_prrac.replay_adapter import PRRACReplayAdapter
from chapter3_bser.experiments.phase1c_prrac.train_phase1c_prrac import (
    _apply_transitions,
    _load_checkpoint,
    _save_checkpoint,
    _write_csv,
)
from chapter3_bser.experiments.phase1c_prrac.transition_protocol import PRRACTransitionMetadata
from chapter3_bser.models.prrac.prrac_maddpg import PRRACMADDPG
from chapter3_bser.models.prrac.stage_mapping import transition_phase_to_prrac_stage


def _config():
    architecture = {
        "num_stages": 3, "encoder_hidden_dim": 12, "expert_hidden_dim": 12,
        "critic_hidden_dim": 24, "router_temperature": 1.0,
        "gate_initial_mean": 0.75, "alignment_scale_init": 1.0,
    }
    loss = {
        "router_ce_coef": 0.05, "gate_conflict_coef": 0.01,
        "gate_entropy_coef": 0.001, "residual_action_reg": 0.01,
    }
    return {
        "seed": 3, "profile": "M20_MOVING_UNKNOWN_MULTI",
        "execution_runtime_revision": "dynamic_public_intercept_v2_1",
        "architecture": architecture, "loss": loss,
        "reward": {"schema": "bser.phase1c.execution_reward.v2"},
        "replay": {"schema": "bser.phase1c.phase_aware_replay.v1"},
    }


def _transitions():
    phases = [TransitionPhase.PRE_FOUND] * 4 + [TransitionPhase.POST_FOUND] * 4 + [TransitionPhase.HOLD] * 4
    rows = []
    previous = TransitionPhase.PRE_FOUND
    for index, phase in enumerate(phases):
        found = phase != TransitionPhase.PRE_FOUND
        hold = phase == TransitionPhase.HOLD
        base = Phase1CTransitionMetadata.build(
            episode_id=0, episode_index=0, step=index + 1,
            task_found=found, executor_target_assigned=found,
            contact=hold, full_hold=hold, hold_counter=int(hold), mission_complete=False,
        )
        metadata = PRRACTransitionMetadata(
            base, transition_phase_to_prrac_stage(previous), transition_phase_to_prrac_stage(phase)
        )
        obs = tuple(torch.randn(28) for _ in range(4))
        actions = torch.tanh(torch.randn(4, 3))
        rewards = torch.randn(4)
        next_obs = tuple(value + 0.01 for value in obs)
        rows.append((obs, actions, rewards, next_obs, (False,) * 4, (False,) * 4, metadata))
        previous = phase
    return rows


class PRRACTrainingSmokeTests(unittest.TestCase):
    def test_real_update_diagnostics_and_checkpoint_in_temporary_directory(self):
        torch.manual_seed(5)
        config = _config()
        learner = PRRACMADDPG(architecture=config["architecture"], loss=config["loss"])
        replay = PRRACReplayAdapter(64, generator_seed=4)
        router_before = next(learner.agents[0].actor.router.parameters()).detach().clone()
        expert_before = next(learner.agents[0].actor.residual_experts.parameters()).detach().clone()
        gate_before = next(learner.agents[0].actor.trust_gate_module.parameters()).detach().clone()
        critic_before = next(learner.agents[0].critic1.parameters()).detach().clone()
        update = _apply_transitions(
            learner, replay, _transitions(), {"episode_id": 0, "success": False},
            {"warmup_steps": 4, "update_frequency": 1, "batch_size": 4,
             "updates_per_train": 1, "policy_delay": 1},
            global_step=0, update_step=0, device="cpu",
        )
        self.assertGreater(update["replay_sample_count"], 0)
        self.assertGreater(update["optimizer_update_count"], 0)
        self.assertFalse(torch.equal(router_before, next(learner.agents[0].actor.router.parameters())))
        self.assertFalse(torch.equal(expert_before, next(learner.agents[0].actor.residual_experts.parameters())))
        self.assertFalse(torch.equal(gate_before, next(learner.agents[0].actor.trust_gate_module.parameters())))
        self.assertFalse(torch.equal(critic_before, next(learner.agents[0].critic1.parameters())))
        self.assertIsNotNone(update["diagnostics"]["gate_mean"])
        with tempfile.TemporaryDirectory() as directory:
            path = _save_checkpoint(
                learner, replay, Path(directory), config, 1,
                global_step=update["global_step"], update_step=update["update_step"],
                replay_sample_count=update["replay_sample_count"],
                optimizer_update_count=update["optimizer_update_count"],
                episode_rows=[{"episode": 1}], execution_rows=[{"episode": 1}],
                prrac_rows=[update["diagnostics"]],
            )
            restored = PRRACMADDPG(architecture=config["architecture"], loss=config["loss"])
            restored_replay = PRRACReplayAdapter(64, generator_seed=1)
            _load_checkpoint(path, restored, restored_replay, config)
            self.assertEqual(len(restored_replay), len(replay))
            diagnostics_path = Path(directory) / "prrac_diagnostics.csv"
            _write_csv(diagnostics_path, [update["diagnostics"]])
            self.assertTrue(diagnostics_path.is_file())
            self.assertGreater(diagnostics_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
