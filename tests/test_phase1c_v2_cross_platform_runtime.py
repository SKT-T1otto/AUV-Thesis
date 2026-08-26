from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields, is_dataclass
import inspect
import multiprocessing as mp
from collections.abc import Mapping
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from chapter3_bser.experiments.phase1c_bser_rmaddpg_v2 import train_phase1c_v2 as trainer


def _contains_tensor(value) -> bool:
    if torch.is_tensor(value):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return any(_contains_tensor(getattr(value, field.name)) for field in fields(value))
    return False


def _worker_job() -> dict:
    config = copy.deepcopy(trainer._load_config(trainer.DEFAULT_CONFIG))
    config["max_steps"] = 1
    config["rl"]["replay_size"] = 8
    maddpg, _ = trainer._build_learner(config)
    manifest = trainer.build_scenario_manifests(
        count=1,
        generator_seed=int(config["seed"]),
        split="train",
        profiles=(str(config["profile"]),),
    )[str(config["profile"])]
    return {
        "episode_index": 0,
        "scenario": manifest["scenarios"][0],
        "base_candidate": config["base_candidate"],
        "profile": config["profile"],
        "max_steps": 1,
        "rl": config["rl"],
        "reward": trainer._reward_config(config),
        "policy_states": trainer._policy_snapshot(maddpg),
    }


class Phase1CV2CrossPlatformRuntimeTests(unittest.TestCase):
    def test_training_executor_uses_spawn_context(self) -> None:
        source = inspect.getsource(trainer.run_training)
        self.assertIn('mp.get_context("spawn")', source)
        self.assertIn("mp_context=mp_context", source)
        self.assertNotIn("set_start_method", source)

    def test_policy_snapshot_contains_numpy_copies_only(self) -> None:
        maddpg = SimpleNamespace(
            agents=[SimpleNamespace(policy=torch.nn.Linear(3, 2)) for _ in range(4)]
        )
        snapshot = trainer._policy_snapshot(maddpg)
        self.assertFalse(_contains_tensor(snapshot))
        for state in snapshot:
            self.assertTrue(state)
            self.assertTrue(all(isinstance(value, np.ndarray) for value in state.values()))
            self.assertTrue(all(value.flags.owndata for value in state.values()))

    def test_worker_transition_payload_is_numpy_and_preserves_contracts(self) -> None:
        result = trainer._collect_episode(_worker_job())

        self.assertFalse(_contains_tensor(result))
        _, transitions, _ = result
        self.assertEqual(len(transitions), 1)
        observations, actions, rewards, next_observations, dones, success, _ = transitions[0]
        self.assertEqual(tuple(value.shape for value in observations), ((28,),) * 4)
        self.assertEqual(actions.shape, (4, 3))
        self.assertEqual(rewards.shape, (4,))
        self.assertEqual(tuple(value.shape for value in next_observations), ((28,),) * 4)
        self.assertEqual(len(dones), 4)
        self.assertEqual(len(success), 4)
        self.assertTrue(all(isinstance(value, np.ndarray) for value in observations))
        self.assertTrue(all(isinstance(value, np.ndarray) for value in next_observations))

    def test_spawn_executor_round_trip_contains_no_tensors(self) -> None:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
            result = executor.submit(trainer._collect_episode, _worker_job()).result(
                timeout=180
            )
        self.assertFalse(_contains_tensor(result))

    def test_cuda_seed_is_applied_when_cuda_is_available(self) -> None:
        with (
            mock.patch.object(trainer.torch, "manual_seed") as manual_seed,
            mock.patch.object(trainer.torch.cuda, "is_available", return_value=True),
            mock.patch.object(trainer.torch.cuda, "manual_seed_all") as manual_seed_all,
        ):
            trainer._seed_all(2729)
        manual_seed.assert_called_with(2729)
        manual_seed_all.assert_called_with(2729)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def test_explicit_cuda_requests_fail_fast_without_cuda(self) -> None:
        with mock.patch.object(trainer.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA device"):
                trainer._resolve_training_device("cuda")

    def test_training_summary_records_reproducibility_runtime(self) -> None:
        source = inspect.getsource(trainer.run_training)
        for field in (
            '"torch_version"',
            '"deterministic_algorithms_enabled"',
            '"cuda_available"',
            '"cuda_device"',
        ):
            self.assertIn(field, source)


if __name__ == "__main__":
    unittest.main()
