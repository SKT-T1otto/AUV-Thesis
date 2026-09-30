"""Only eight physical fixture steps; no real development episodes or training."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from chapter3_bser.experiments.safe_search_v1.run_paired import read_json, load_dependencies, execute_episode
from chapter3_bser.experiments.bser_effect_v1.runtime import make_runtime
from chapter3_bser.events.event_types import BSEREvent

ROOT=Path(__file__).resolve().parents[1]


class BoundedTests(unittest.TestCase):
    def test_all_arms_use_real_zero_residual_physics_and_observed_trace(self):
        torch.set_num_threads(1)
        config=read_json(ROOT/"configs/chapter3/hgr_train.json")
        scenario=read_json(ROOT/"tests/fixtures/hgr/handoff_manifest.json")["scenarios"][0]
        scenario.update(max_steps=400,scenario_id="D_bounded_fixture",target_position=[18.,18.,6.],target_initial_position=[18.,18.,6.])
        for arm in ("D0","D1","D2","D3"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                instances=[]
                def factory():
                    instance=make_runtime(copy.deepcopy(config),copy.deepcopy(scenario),arm=arm,seed=123)
                    instances.append(instance)
                    self.assertFalse(instance.env.get_task_state().target_found)
                    self.assertEqual([tuple(x.shape) for x in instance.observations],[(28,)]*4)
                    original=instance.env.step
                    def step(actions,*args,**kwargs):
                        np.testing.assert_array_equal(actions.cpu().numpy(),np.zeros((4,3)))
                        return original(actions,*args,**kwargs)
                    instance.env.step=step
                    # Exercise an eligible planning event within the two-step
                    # synthetic test without waiting 20 physical steps.
                    detect=instance.controller.inner.detector.detect
                    def forced_event(*args,**kwargs):
                        return replace(detect(*args,**kwargs),events=(BSEREvent.PERIODIC_REFRESH,))
                    instance.controller.inner.detector.detect=forced_event
                    return instance
                deps=load_dependencies(config,scenario,0,123,"B1_bser_prior",search_coverage=True)
                _,row=execute_episode(factory,Path(directory),steps=2,observed=True,deps=deps,full_episodes=False)
                self.assertEqual(row["physical_steps"],2)
                self.assertEqual(row["controller"]["optimizer_update_count"],0)
                self.assertEqual(row["controller"]["physical_residual_acceleration_max_abs"],0)
                self.assertEqual(row["controller"]["bser_effect_v1"]["arm"],arm)
                self.assertTrue(row["search_coverage"]["available"])
                runtime=instances[0]
                self.assertEqual(runtime.env.unwrapped.max_steps,400)
                self.assertEqual(runtime.bridge.path_tracker.threshold,.75)
                self.assertTrue(any(x["kind"]=="decision" for x in runtime.controller.allocator.audit))
                if arm in ("D0","D1"):
                    self.assertEqual(runtime.controller.current_allocation.executor_assignment.target_region,runtime.controller.allocator.anchor)


if __name__ == "__main__":
    unittest.main()
