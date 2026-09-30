"""Eight physical steps on a synthetic fixture; no development-set evaluation."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from chapter3_bser.experiments.safe_search_v1.run_paired import read_json
from chapter3_bser.experiments.safe_search_v2.runtime import make_runtime
from chapter3_bser.experiments.safe_search_v2.motion import Vehicle

ROOT = Path(__file__).resolve().parents[1]


class BoundedRuntimeTests(unittest.TestCase):
    def test_both_parent_and_repaired_runtime_contracts_and_nominal_prediction(self):
        torch.set_num_threads(1)
        config = read_json(ROOT/"configs/chapter3/hgr_train.json")
        scenario = read_json(ROOT/"tests/fixtures/hgr/handoff_manifest.json")["scenarios"][0]
        scenario = copy.deepcopy(scenario)
        scenario.update(max_steps=400, scenario_id="found_v2_interface_fixture",
                        target_position=[18.,18.,6.], target_initial_position=[18.,18.,6.])
        for baseline in ("B0_search_prior", "B1_bser_prior"):
            for arm in ("R0", "R4"):
                with self.subTest(baseline=baseline, arm=arm):
                    runtime = make_runtime(copy.deepcopy(config), copy.deepcopy(scenario),
                                           baseline=baseline, arm=arm, seed=123, episode_id=0)
                    try:
                        self.assertFalse(runtime.env.get_task_state().target_found)
                        self.assertEqual([tuple(o.shape) for o in runtime.observations], [(28,)]*4)
                        before = runtime.env.get_agent_state()
                        item = runtime.guidance.assignment_for(0)
                        target = item.hold_position if item.hold_state else item.tracking_waypoint
                        vehicle = Vehicle.from_constants(runtime.env.unwrapped, 0)
                        p, v = vehicle.step(np.asarray(before.positions[0]), np.asarray(before.velocities[0]),
                                            np.asarray(target), np.array([.18,0.,0.]))
                        def constant_flow(points):
                            flow = torch.zeros_like(points)
                            flow[...,0] = .18
                            return flow
                        # A test-only controlled disturbance verifies forecast
                        # discretization against real unchanged physics.
                        with patch.object(runtime.env.unwrapped, "_flow_at", side_effect=constant_flow), \
                             patch.object(runtime.env, "step", wraps=runtime.env.step) as physical_step:
                            runtime.advance()
                            after = runtime.env.get_agent_state()
                            np.testing.assert_allclose(after.positions[0], p, atol=2e-6, rtol=0)
                            np.testing.assert_allclose(after.velocities[0], v, atol=2e-6, rtol=0)
                            runtime.advance()
                            for call in physical_step.call_args_list:
                                np.testing.assert_array_equal(call.args[0].cpu().numpy(), np.zeros((4,3)))
                        self.assertEqual(runtime.step, 2)
                        diagnostics = runtime.controller_diagnostics()
                        self.assertEqual(diagnostics["physical_residual_acceleration_max_abs"], 0.)
                        self.assertEqual(diagnostics["optimizer_update_count"], 0)
                        self.assertEqual(runtime.env.unwrapped.max_steps, 400)
                        self.assertEqual([tuple(o.shape) for o in runtime.observations], [(28,)]*4)
                    finally:
                        runtime.close()


if __name__ == "__main__":
    unittest.main()
