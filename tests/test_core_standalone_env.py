import unittest

import torch

from core.config.ch3_config import build_ch3_config
from core.env import MissionCoreEnv, environment_kwargs_from_config
from core.env.uav_env import UAVEnv
from core.scenarios.generator import build_e0_manifests


class CoreStandaloneEnvironmentTests(unittest.TestCase):
    def test_facade_uses_core_and_preserves_contract(self):
        config = build_ch3_config("ch3_v3_full_reference", "M00_MOVING_CLEAR")
        env = MissionCoreEnv(**environment_kwargs_from_config(config, max_steps=1))
        try:
            self.assertEqual(env.implementation_source, "core")
            self.assertIsInstance(env.unwrapped, UAVEnv)
            self.assertEqual(env.unwrapped.__class__.__module__, "core.env.uav_env")
            scenario = build_e0_manifests()["M00_MOVING_CLEAR"]["scenarios"][0]
            obs = env.reset(scenario)
            self.assertEqual([tuple(item.shape) for item in obs], [(28,)] * 4)
            _, rewards, done = env.step(torch.zeros((4, 3), dtype=torch.float32))
            self.assertEqual(tuple(rewards.shape), (4,))
            self.assertEqual(len(done), 4)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
