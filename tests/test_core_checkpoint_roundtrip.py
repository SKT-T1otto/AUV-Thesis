import tempfile
import unittest
from pathlib import Path

import torch

from core.algorithms.maddpg import MADDPG


class _Space:
    def __init__(self, shape):
        self.shape = shape


class _DummyEnv:
    num_agents = 4
    role_names = ["search_fast", "search_balanced", "search_precise", "executor"]
    observation_space = {f"agent_{i}": _Space((28,)) for i in range(4)}
    action_space = {f"agent_{i}": _Space((3,)) for i in range(4)}


class CoreCheckpointRoundtripTests(unittest.TestCase):
    def test_weights_only_checkpoint_roundtrip(self):
        model = MADDPG.init_from_env(_DummyEnv(), hidden_dim=16)
        observations = [torch.zeros((1, 28), dtype=torch.float32) for _ in range(4)]
        before = model.step(observations, explore=False)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "roundtrip.pt"
            model.save(path, metadata={"test": "core_roundtrip"})
            loaded = MADDPG.init_from_save(path, device="cpu")
            after = loaded.step(observations, explore=False)
        self.assertEqual(loaded.checkpoint_metadata, {"test": "core_roundtrip"})
        for left, right in zip(before, after):
            self.assertTrue(torch.equal(left, right))


if __name__ == "__main__":
    unittest.main()
