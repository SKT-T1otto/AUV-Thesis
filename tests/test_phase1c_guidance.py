import json
import unittest
from pathlib import Path

import numpy as np
import torch

from chapter3_bser.controllers.state_provider import OnlinePlanningStateProvider
from chapter3_bser.integration.control_context import BSERControlContextV1
from chapter3_bser.integration.guided_env import GuidedEnv
from chapter3_bser.integration.rmaddpg_bridge import (
    RMADDPGGuidanceBridge,
    get_tracking_targets,
)
from chapter3_bser.online.config import load_phase1b2_config
from chapter3_bser.online.controller import OnlineBSERController
from chapter3_bser.online.mission_context import OnlineMissionContext
from core.algorithms.maddpg import MADDPG
from core.config.ch3_config import build_ch3_config
from core.env.mission_env import MissionCoreEnv, environment_kwargs_from_config
from core.replay.ch3_buffer import CH3ReplayBuffer


class Phase1CGuidanceSmokeTest(unittest.TestCase):
    @staticmethod
    def _mission_context(env, state):
        return OnlineMissionContext.from_public_views(
            env.get_task_state(),
            env.get_search_execution_state(),
            state,
        )

    def test_phase1c_config_is_disabled_by_default(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "chapter3"
            / "bser_phase1c.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(config["enabled"])
        self.assertFalse(config["phase1c_enable_bser_guidance"])
        self.assertFalse(config["training_enabled"])

    def test_reset_step_guidance_and_replay_transition_are_aligned(self):
        np.random.seed(2729)
        torch.manual_seed(2729)
        env_config = build_ch3_config(
            "ch3_v3_full_reference", "M20_MOVING_UNKNOWN_MULTI"
        )
        base_env = MissionCoreEnv(
            **environment_kwargs_from_config(
                env_config,
                device="cpu",
                max_steps=4,
                return_numpy=False,
            )
        )
        env = GuidedEnv(base_env, enabled=True)
        try:
            reset_observations = env.reset()
            self.assertEqual(len(reset_observations), 4)

            provider = OnlinePlanningStateProvider(env, refresh_interval=20)
            state = provider.initialize()
            mission_context = self._mission_context(env, state)
            controller = OnlineBSERController(load_phase1b2_config())
            initial = controller.initialize(state, mission_context)
            bridge = RMADDPGGuidanceBridge()
            guidance = bridge.compile_guidance(
                initial.allocation,
                state,
                mission_context,
                decision_reason="INITIALIZE",
            )
            self.assertIsInstance(guidance, BSERControlContextV1)

            task_before = env.get_task_state()
            agents_before = env.get_agent_state()
            acceleration_before = env.unwrapped._agent_acc.detach().clone()
            env.install_guidance(guidance)
            observations = env.refresh_observation_after_guidance()
            self.assertEqual(task_before, env.get_task_state())
            self.assertEqual(agents_before.positions, env.get_agent_state().positions)
            self.assertEqual(agents_before.velocities, env.get_agent_state().velocities)
            self.assertTrue(
                torch.equal(acceleration_before, env.unwrapped._agent_acc)
            )

            for observation in observations:
                self.assertEqual(tuple(observation.shape), (28,))

            maddpg = MADDPG.init_from_env(env.unwrapped, hidden_dim=16)
            self.assertTrue(
                all(
                    params["num_in_pol"] == 28
                    and params["num_out_pol"] == 3
                    and params["num_in_critic"] == 124
                    for params in maddpg.agent_init_params
                )
            )

            actions = torch.zeros((4, 3), dtype=torch.float32)
            self.assertEqual(tuple(actions.shape), (4, 3))
            _, rewards, dones = env.step(actions)

            next_state = provider.snapshot(force=True)
            next_mission_context = self._mission_context(env, next_state)
            result = controller.step(next_state, next_mission_context)
            next_guidance = bridge.compile_guidance(
                result.allocation,
                next_state,
                next_mission_context,
                decision_reason=result.decision_reason,
            )
            env.install_guidance(next_guidance)
            next_observations = env.refresh_observation_after_guidance()

            targets = get_tracking_targets(next_guidance)
            public_agents = env.get_agent_state()
            for agent_id, observation in enumerate(next_observations):
                self.assertEqual(tuple(observation.shape), (28,))
                expected_delta = np.asarray(targets[agent_id]) - np.asarray(
                    public_agents.positions[agent_id]
                )
                np.testing.assert_allclose(
                    observation[6:9].detach().cpu().numpy(),
                    expected_delta,
                    rtol=0.0,
                    atol=1e-5,
                )

            replay = CH3ReplayBuffer(
                max_steps=4,
                num_agents=4,
                obs_dims=(28, 28, 28, 28),
                ac_dims=(3, 3, 3, 3),
                storage_device="cpu",
            )
            replay.push(
                observations,
                actions,
                rewards,
                next_observations,
                dones,
                [bool(next_mission_context.mission_complete)] * 4,
            )
            sample = replay.sample(1, norm_rews=False, device="cpu")
            sampled_obs, sampled_actions, _, sampled_next_obs = sample[:4]
            self.assertTrue(all(tuple(value.shape) == (1, 28) for value in sampled_obs))
            self.assertTrue(all(tuple(value.shape) == (1, 3) for value in sampled_actions))
            self.assertTrue(
                all(tuple(value.shape) == (1, 28) for value in sampled_next_obs)
            )
            for agent_id in range(4):
                torch.testing.assert_close(
                    sampled_next_obs[agent_id][0], next_observations[agent_id]
                )
        finally:
            env.close()

    def test_unknown_map_reset_defers_invalid_legacy_initial_endpoint(self):
        env_config = build_ch3_config(
            "ch3_v3_full_reference", "M20_MOVING_UNKNOWN_MULTI"
        )
        base_env = MissionCoreEnv(
            **environment_kwargs_from_config(
                env_config,
                device="cpu",
                max_steps=1,
                return_numpy=False,
            )
        )
        env = GuidedEnv(base_env, enabled=True)
        planner = env.unwrapped.map_module

        def invalid_endpoint(_point, role="searcher"):
            return {
                "reachable": False,
                "role": role,
                "failure_reason": "point_invalid",
            }

        planner.endpoint_status = invalid_endpoint
        try:
            observations = env.reset()
            self.assertEqual(len(observations), 4)
            self.assertEqual(env.initial_endpoint_fallback_count, 1)
            self.assertEqual(
                tuple(item["agent_id"] for item in env.last_initial_endpoint_failures),
                (0, 1, 2),
            )
            torch.testing.assert_close(
                env.unwrapped._search_waypoints,
                env.unwrapped._agent_pos[:3],
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
