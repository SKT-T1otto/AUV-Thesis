from __future__ import annotations

import copy
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from multiprocessing.reduction import ForkingPickler
import traceback
import unittest

import numpy as np

from chapter3_bser.experiments.phase1c_prrac import train_phase1c_prrac as trainer
from chapter3_bser.models.prrac.prrac_maddpg import PRRACMADDPG


POOL_FAILURE_MARKERS = (
    "rebuild_storage_fd",
    "received 0 items of ancdata",
    "Too many open files",
    "BrokenProcessPool",
)


def _plain_metadata(value) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _plain_metadata(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_plain_metadata(item) for item in value)
    return False


def _worker_jobs(count: int = 2) -> list[dict]:
    config = copy.deepcopy(trainer._load_config(trainer.DEFAULT_CONFIG))
    config["max_steps"] = 1
    config["architecture"].update(
        encoder_hidden_dim=8,
        expert_hidden_dim=8,
        critic_hidden_dim=16,
    )
    actor = PRRACMADDPG(
        architecture=config["architecture"],
        loss=config["loss"],
        gamma=float(config["rl"]["gamma"]),
        tau=float(config["rl"]["tau"]),
        lr_actor=float(config["rl"]["lr_actor"]),
        lr_critic=float(config["rl"]["lr_critic"]),
    )
    snapshot = actor.policy_snapshot()
    scenarios = trainer.build_scenario_manifests(
        count=count,
        generator_seed=int(config["seed"]),
        split="train",
        profiles=(str(config["profile"]),),
    )[str(config["profile"])]["scenarios"]
    return [
        {
            "episode_index": index,
            "scenario": scenarios[index],
            "base_candidate": config["base_candidate"],
            "profile": config["profile"],
            "max_steps": 1,
            "rl": config["rl"],
            "reward": config["reward"],
            "architecture": config["architecture"],
            "loss": config["loss"],
            "execution_runtime": config["execution_runtime"],
            "policy_snapshot": snapshot,
        }
        for index in range(count)
    ]


class PRRACCrossPlatformRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.jobs = _worker_jobs()

    def assert_worker_payload(self, result) -> None:
        self.assertFalse(trainer._contains_tensor(result))
        _, transitions, execution, diagnostics = result
        self.assertTrue(_plain_metadata(execution))
        self.assertTrue(_plain_metadata(diagnostics))
        self.assertEqual(len(transitions), 1)
        observations, actions, rewards, next_observations, dones, success, metadata = (
            transitions[0]
        )
        self.assertEqual(tuple(value.shape for value in observations), ((28,),) * 4)
        self.assertEqual(actions.shape, (4, 3))
        self.assertEqual(rewards.shape, (4,))
        self.assertEqual(tuple(value.shape for value in next_observations), ((28,),) * 4)
        self.assertTrue(all(isinstance(value, np.ndarray) for value in observations))
        self.assertIsInstance(actions, np.ndarray)
        self.assertIsInstance(rewards, np.ndarray)
        self.assertTrue(all(isinstance(value, np.ndarray) for value in next_observations))
        self.assertTrue(all(type(value) is bool for value in dones))
        self.assertTrue(all(type(value) is bool for value in success))
        self.assertTrue(_plain_metadata(metadata))

    def test_linux_launcher_is_lf_thin_and_cpu_safe_by_default(self) -> None:
        path = trainer.ROOT / "scripts" / "linux" / "run_phase1c_prrac_train.sh"
        payload = path.read_bytes()
        text = payload.decode("utf-8")

        self.assertNotIn(b"\r", payload)
        for expected in (
            "set -euo pipefail",
            'CRK_CONDA_ENV="${CRK_CONDA_ENV:-AUV}"',
            "export MPLBACKEND=Agg",
            'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"',
            'export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"',
            'python -B -m chapter3_bser.experiments.phase1c_prrac.train_phase1c_prrac "$@"',
        ):
            self.assertIn(expected, text)
        self.assertNotIn("python -B - <<", text)

    def test_policy_snapshot_is_numpy_copy_and_spawn_pickleable(self) -> None:
        snapshot = self.jobs[0]["policy_snapshot"]
        self.assertFalse(trainer._contains_tensor(snapshot))
        for actor_state in snapshot:
            self.assertTrue(actor_state)
            self.assertTrue(
                all(isinstance(value, np.ndarray) for value in actor_state.values())
            )
            self.assertTrue(all(value.flags.owndata for value in actor_state.values()))
        restored = ForkingPickler.loads(ForkingPickler.dumps(snapshot))
        self.assertFalse(trainer._contains_tensor(restored))
        for actor_state in restored:
            self.assertTrue(
                all(isinstance(value, np.ndarray) for value in actor_state.values())
            )

    def test_spawn_single_worker_collects_tensor_free_payload(self) -> None:
        self.assertFalse(trainer._contains_tensor(self.jobs[0]))
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
            result = executor.submit(trainer._collect_episode, self.jobs[0]).result(
                timeout=240
            )
        self.assert_worker_payload(result)

    def test_spawn_two_worker_smoke_has_no_pool_transport_failure(self) -> None:
        context = mp.get_context("spawn")
        try:
            with ProcessPoolExecutor(max_workers=2, mp_context=context) as executor:
                futures = [
                    executor.submit(trainer._collect_episode, job) for job in self.jobs
                ]
                results = [future.result(timeout=240) for future in futures]
        except Exception:
            failure = traceback.format_exc()
            for marker in POOL_FAILURE_MARKERS:
                self.assertNotIn(marker, failure)
            raise
        self.assertEqual(len(results), 2)
        for result in results:
            self.assert_worker_payload(result)


if __name__ == "__main__":
    unittest.main()
