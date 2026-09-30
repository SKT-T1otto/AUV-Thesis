"""Bounded synthetic integration; never formal performance evidence."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.experiments.bser_effect_v1.runtime import make_runtime
from chapter3_bser.experiments.d2_v1.contract import enabled
from chapter3_bser.experiments.d2_v1.provenance import framework_sources
from chapter3_bser.experiments.hgr.runtime import MissionRuntime, collect_trajectory, continue_branch
from chapter3_bser.experiments.baselines.common.runtime import BaselineMissionRuntime
from chapter3_bser.models.hgr.policy import HandoffPolicy
from tools.ch3_baselines.bser_prior import ZeroResidualSource
from tools.ch3_baselines.registry import training_config, load_reference, task_conditions, validate_method_config

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def scene():
    value = read("tests/fixtures/hgr/handoff_manifest.json")["scenarios"][0]
    value.update(max_steps=400, scenario_id="D2_synthetic_search",
                 target_position=[18., 18., 6.], target_initial_position=[18., 18., 6.])
    return value


class Actions:
    def __init__(self, value=0.):
        self.value = value
        self.calls = 0

    def step(self, observations, *, explore):
        self.calls += 1
        return [torch.full((1, 3), self.value) for _ in observations]


def force_refresh(runtime):
    original = runtime.controller.inner.detector.detect
    runtime.controller.inner.detector.detect = lambda *a, **kw: replace(
        original(*a, **kw), events=(BSEREvent.PERIODIC_REFRESH,))


class D2Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_seal_defaults_and_planner_mismatch(self):
        sources = framework_sources()
        self.assertEqual(sources["historical_record_count"], 27)
        _, hgr = load_reference()
        self.assertTrue(enabled(hgr))
        self.assertFalse(enabled(read("configs/chapter3/hgr_train.json")))
        with self.assertRaises(ValueError):
            enabled(dict(planner_protocol="D2_typo"))
        for name in ("B2_direct_mc", "B3_direct_boundary"):
            spec, _, config = training_config(name)
            self.assertEqual(spec["planner"], "d2_v1")
            self.assertEqual(task_conditions(config), task_conditions(hgr))
            with self.assertRaises(ValueError):
                validate_method_config(spec, {**config, "planner_protocol": "legacy_bser"}, hgr)

    def test_original_d2_and_both_runtime_paths_match_with_zero_actions(self):
        old = read("configs/chapter3/hgr_train.json")
        config = {**old, "planner_protocol": "d2_v1"}
        records = []
        for mode in ("reference", "hgr", "baseline"):
            runtime = (make_runtime(old, scene(), arm="D2", seed=123) if mode == "reference" else
                       (MissionRuntime if mode == "hgr" else BaselineMissionRuntime)(config, scene(), seed=123))
            policy, model = ZeroResidualSource(), Actions()
            try:
                self.assertFalse(runtime.env.get_task_state().target_found)
                force_refresh(runtime)
                trace = []
                for _ in range(4):
                    if mode == "reference": runtime.advance()
                    elif mode == "hgr": runtime.advance(policy, deterministic=True)
                    else: runtime.advance(model, explore=False)
                    trace.append((np.asarray(runtime.env.get_agent_state().positions).copy(),
                                  np.asarray(runtime.env.get_agent_state().velocities).copy(),
                                  copy.deepcopy(runtime.controller.current_allocation),
                                  np.asarray([o.numpy() for o in runtime.observations]).copy()))
                self.assertTrue(runtime.controller.allocator.audit)
                records.append(trace)
            finally:
                runtime.close()
        for trace in records[1:]:
            for actual, expected in zip(trace, records[0]):
                for i in (0, 1, 3): np.testing.assert_array_equal(actual[i], expected[i])
                self.assertEqual(actual[2], expected[2])

    def test_learned_baseline_actions_reach_physics(self):
        config = training_config("B2_direct_mc")[2]
        runtime = BaselineMissionRuntime(config, scene(), seed=123)
        model = Actions(.2)
        try:
            row = runtime.advance(model, explore=False)
            self.assertEqual(model.calls, 1)
            np.testing.assert_allclose(row["actions"], .2)
            self.assertGreater(float(runtime.env.unwrapped._last_residual_acc.abs().max()), 0.)
        finally:
            runtime.close()

    def test_hgr_real_handoff_snapshot_preserves_d2_owners_and_continuation(self):
        config = read("tests/fixtures/hgr/integration_config.json")
        config["planner_protocol"] = "d2_v1"
        scenario = read(config["scenario_manifest"])["scenarios"][0]
        torch.manual_seed(9)
        policy = HandoffPolicy(config["policy"])
        result = collect_trajectory(config, scenario, policy, seed=321)
        self.assertIsNotNone(result["snapshot"])
        snapshot = result["snapshot"]
        restored = MissionRuntime.restore(snapshot)
        try:
            self.assertIs(restored.controller.owner, restored)
            self.assertIs(restored.controller.allocator.owner, restored)
            self.assertIs(restored.planning_views.runtime, restored)
            self.assertIs(restored.bridge.owner, restored)
            actual = []
            while not restored.terminal:
                actual.append(restored.advance(policy))
            expected = result["records"][snapshot.step:]
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                for key in ("observations", "actions", "next_observations", "team_reward"):
                    np.testing.assert_array_equal(a[key], b[key])
        finally:
            restored.close()
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        from tests.test_hgr_integration import spawned_continuation
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
            spawned = pool.submit(spawned_continuation, snapshot, config["policy"],
                                  policy.state_dict()).result(timeout=180)
        self.assertEqual(spawned["steps"], len(expected))
        for a, b in zip(spawned["records"], expected):
            for key in ("observations", "actions", "next_observations", "team_reward"):
                np.testing.assert_array_equal(a[key], b[key])

    def test_both_trainers_do_bounded_real_updates_on_d2(self):
        from tests.test_ch3_baseline_training import tiny_config, independent_calls_only
        from chapter3_bser.experiments.baselines.direct_mc.train import B2Trainer
        from chapter3_bser.experiments.baselines.direct_boundary.train import B3Trainer
        for name, trainer in (("B2_direct_mc", B2Trainer), ("B3_direct_boundary", B3Trainer)):
            config = tiny_config(name)
            config["planner_protocol"] = "d2_v1"
            with tempfile.TemporaryDirectory() as directory, independent_calls_only():
                instance = trainer(config, Path(directory) / "collision_terminal" / "smoke")
                summary = instance.run()
                self.assertGreater(summary["optimizer_updates"], 0)
                self.assertEqual(instance.episodes[0]["planner"]["planner_protocol"], "d2_v1")


if __name__ == "__main__":
    unittest.main()
