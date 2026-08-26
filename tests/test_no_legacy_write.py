import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

from core.env import MissionCoreEnv, environment_kwargs_from_config
from core.config.ch3_config import build_ch3_config
from core.scenarios.generator import build_e0_manifests


class LegacyReadOnlyTests(unittest.TestCase):
    def test_core_environment_does_not_write_outside_temporary_output(self):
        config = build_ch3_config("ch3_v3_full_reference", "M00_MOVING_CLEAR")
        env = MissionCoreEnv(**environment_kwargs_from_config(config, max_steps=1))
        with tempfile.TemporaryDirectory() as temporary:
            before = sorted(path.relative_to(temporary) for path in Path(temporary).rglob("*"))
            scenario = build_e0_manifests()["M00_MOVING_CLEAR"]["scenarios"][0]
            env.reset(scenario)
            env.step(torch.zeros((4, 3), dtype=torch.float32))
            after = sorted(path.relative_to(temporary) for path in Path(temporary).rglob("*"))
        env.close()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
