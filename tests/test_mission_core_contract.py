import json
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]

from core.config.ch3_config import build_ch3_config
from core.env import MissionCoreEnv, environment_kwargs_from_config
from core.env.task_state import MappingStateView, MissionTaskState, TargetStateView


class MissionCoreContractTests(unittest.TestCase):
    def test_cpu_reset_step_and_public_views(self):
        config = build_ch3_config("ch3_v3_full_reference", "M00_MOVING_CLEAR")
        env = MissionCoreEnv(**environment_kwargs_from_config(config, max_steps=2))
        try:
            manifest = json.loads((ROOT / "configs/scenarios/e0_equivalence/M00_MOVING_CLEAR.json").read_text(encoding="utf-8"))
            observations = env.reset(manifest["scenarios"][0])
            self.assertEqual([tuple(item.shape) for item in observations], [(28,)] * 4)
            self.assertIsInstance(env.get_task_state(), MissionTaskState)
            self.assertIsInstance(env.get_target_state(), TargetStateView)
            self.assertIsInstance(env.get_mapping_state(), MappingStateView)
            new_obs, reward, done = env.step(torch.zeros((4, 3), dtype=torch.float32))
            self.assertEqual([tuple(item.shape) for item in new_obs], [(28,)] * 4)
            self.assertEqual(tuple(reward.shape), (4,))
            self.assertEqual(len(done), 4)
            self.assertEqual(env.action_range, (-1.0, 1.0))
            self.assertEqual(env.role_order, ("search_fast", "search_balanced", "search_precise", "executor"))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
