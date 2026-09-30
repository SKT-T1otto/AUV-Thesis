"""Bounded synthetic checks only. No full evaluation episode or training."""
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
from chapter3_bser.experiments.bser_effect_v1.runtime import make_runtime as d_runtime
from chapter3_bser.experiments.bser_final_v1.runtime import make_runtime
from chapter3_bser.experiments.bser_final_v1.options import ARMS, SETTINGS
from chapter3_bser.experiments.bser_final_v1.scoring import allocation_key
from chapter3_bser.experiments.safe_search_v1.run_paired import load_dependencies, execute_episode

ROOT=Path(__file__).resolve().parents[1]


def inputs():
    config=json.loads((ROOT/"configs/chapter3/hgr_train.json").read_text(encoding="utf-8"))
    scenario=json.loads((ROOT/"tests/fixtures/hgr/handoff_manifest.json").read_text(encoding="utf-8"))["scenarios"][0]
    scenario.update(max_steps=400,scenario_id="F_bounded_fixture",target_position=[18.,18.,6.],target_initial_position=[18.,18.,6.])
    return config,scenario


def force_event(runtime):
    original=runtime.controller.inner.detector.detect
    runtime.controller.inner.detector.detect=lambda *a,**k:replace(original(*a,**k),events=(BSEREvent.PERIODIC_REFRESH,))


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_f0_delegates_exact_d2_physical_and_guidance_behavior(self):
        config,scenario=inputs()
        records=[]
        for make,arm in ((d_runtime,"D2"),(make_runtime,"F0")):
            runtime=make(copy.deepcopy(config),copy.deepcopy(scenario),arm=arm,seed=123)
            try:
                force_event(runtime)
                values=[]
                for _ in range(3):
                    values.append((allocation_key(runtime.controller.current_allocation),runtime.guidance,
                        np.asarray(runtime.env.get_agent_state().positions).tolist(),
                        [np.asarray(x).tolist() for x in runtime.observations]))
                    runtime.advance()
                records.append(values)
            finally:runtime.close()
        self.assertEqual(records[0],records[1])

    def test_all_six_interventions_reach_real_event_path_with_zero_actions(self):
        config,scenario=inputs()
        for arm in list(ARMS)[1:]:
            with self.subTest(arm=arm),tempfile.TemporaryDirectory() as directory:
                instances=[]
                def factory():
                    runtime=make_runtime(copy.deepcopy(config),copy.deepcopy(scenario),arm=arm,seed=123)
                    instances.append(runtime)
                    self.assertEqual([tuple(x.shape) for x in runtime.observations],[(28,)]*4)
                    step=runtime.env.step
                    def checked(actions,*a,**k):
                        np.testing.assert_array_equal(actions.cpu().numpy(),np.zeros((4,3)))
                        return step(actions,*a,**k)
                    runtime.env.step=checked
                    force_event(runtime)
                    return runtime
                deps=load_dependencies(config,scenario,0,123,"B1_bser_prior",search_coverage=True)
                _,row=execute_episode(factory,Path(directory),steps=2,observed=True,deps=deps,full_episodes=False)
                runtime=instances[0]
                self.assertEqual(row["physical_steps"],2)
                self.assertEqual(row["controller"]["optimizer_update_count"],0)
                self.assertEqual(runtime.bridge.path_tracker.threshold,.75)
                self.assertTrue(any(r["kind"]=="decision" for r in runtime.controller.allocator.audit))
                self.assertGreater(len(runtime.controller.allocator.final_audit),1)
                for record in runtime.controller.allocator.final_audit:
                    self.assertLessEqual(record["counts"]["response_queries"],80)
                    self.assertLessEqual(record["counts"]["forecasts"],24)
                    self.assertLessEqual(record["combinations"],256)
                    if not record["selected_is_reference"]:
                        self.assertIsNone(record["fallback_reason"])

    def test_budget_fallback_and_preview_do_not_mutate_live_controller_or_rng(self):
        config,scenario=inputs()
        runtime=make_runtime(config,scenario,arm="F1",seed=123)
        try:
            allocator=runtime.controller.allocator
            before=allocation_key(runtime.controller.current_allocation)
            tracking=copy.deepcopy(vars(runtime.bridge.path_tracker))
            rng=torch.get_rng_state().clone()
            with patch.dict(SETTINGS,max_forecasts=0):
                proposed=allocator.allocate(runtime.state,trigger_reason="TEST_BUDGET")
            self.assertEqual(allocator.final_audit[-1]["fallback_reason"],"forecast_budget")
            self.assertEqual(allocator.final_audit[-1]["reference_ids"],[x.candidate_id for x in proposed.search_assignments])
            self.assertEqual(before,allocation_key(runtime.controller.current_allocation))
            self.assertEqual(tracking,vars(runtime.bridge.path_tracker))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            # Proposals and model previews never step the simulator.
            self.assertEqual(runtime.step,0)
        finally:runtime.close()


if __name__=="__main__":
    unittest.main()
