"""Counterexamples for temporal coverage, ablations and bounded fallback."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from chapter3_bser.experiments.bser_final_v1.options import ARMS, SETTINGS, parse_arms
from chapter3_bser.experiments.bser_final_v1.scoring import PlanScore, Scorer, connected_times
from chapter3_bser.experiments.bser_final_v1.forecast import Incomparable, vehicle_constants
from chapter3_bser.experiments.bser_final_v1.solver import guard, choose
from chapter3_bser.experiments.safe_search_v1.public_geometry import PublicGeometry, filter_planning_state
from core.mapping.travel_cost_service import TravelCostService
from core.mapping.planning_state import planning_state_sha256
from tests.test_safe_search_geometry import fixture
from tests.test_bser_effect import candidate, standby


def score(full=1., near=1., response=1., risks=None):
    return PlanScore(full, near, response, risks or {0:(0.,1.,1.), 3:(0.,1.,1.)})


class GuardTests(unittest.TestCase):
    def test_search_tradeoff_risk_compensation_and_each_ablation(self):
        reference = score()
        sacrificed = score(full=.9, near=.9, response=1.2)
        self.assertEqual(guard(sacrificed, reference, ARMS["F1"]), "full_search_regression")
        self.assertIsNone(guard(sacrificed, reference, ARMS["F3"]))
        risky = score(response=1.2, risks={0:(0.,1.1,1.),3:(0.,0.,0.)})
        self.assertEqual(guard(risky, reference, ARMS["F1"]), "risk_regression_agent_0")
        self.assertIsNone(guard(risky, reference, ARMS["F5"]))
        self.assertEqual(guard(score(response=1.001),reference,ARMS["F1"]),"below_minimum_gain")
        self.assertIsNone(guard(score(near=1.04,response=1.),reference,ARMS["F1"]))
        self.assertEqual(guard(score(near=1.,response=2.),reference,ARMS["F2"]),"below_minimum_gain")
        self.assertIsNone(guard(score(near=1.04,response=None),reference,ARMS["F2"]))

    def test_enumeration_finds_joint_improvement_coordinate_greedy_misses(self):
        groups = [(candidate(i,f"{i}_a",1),candidate(i,f"{i}_b",2)) for i in (0,1)]
        ys = (standby("y",0),)
        def evaluate(items,y):
            b = sum(c.candidate_id.endswith("b") for c in items)
            return ((tuple(c.candidate_id for c in items),score(response={0:1.,1:.9,2:1.3}[b])),None)
        full,_,count = choose(groups,ys,score(),evaluate,ARMS["F1"])
        greedy,_,_ = choose(groups,ys,score(),evaluate,ARMS["F6"])
        self.assertEqual(full[0],("0_b","1_b"))
        self.assertIsNone(greedy)
        self.assertEqual(count,4)

    def test_deterministic_budget_aborts_instead_of_returning_prefix_winner(self):
        groups=[(candidate(0,"a",1),candidate(0,"b",2))]
        with patch.dict(SETTINGS,max_combinations=1),self.assertRaisesRegex(Incomparable,"combination_budget"):
            choose(groups,(standby("y",0),),score(),lambda cs,y:((cs,score(response=2)),None),ARMS["F1"])


    def test_coordinate_greedy_uses_supplied_d2_seed_and_rejects_out_of_pool_seed(self):
        groups=[(candidate(i,f"{i}_a",1),candidate(i,f"{i}_b",2)) for i in (0,1)]
        seed=tuple(g[1] for g in groups)
        visited=[]
        def evaluate(items,y):
            visited.append(tuple(c.candidate_id for c in items))
            return ((items,score(response=1.2)),None)
        choose(groups,(standby("y",0),),score(),evaluate,ARMS["F6"],seed_items=seed)
        self.assertEqual(visited[0],("0_a","1_b"))
        with self.assertRaisesRegex(ValueError,"same candidate"):
            choose(groups,(standby("y",0),),score(),evaluate,ARMS["F6"],
                   seed_items=(candidate(0,"missing",1),seed[1]))


class TemporalTests(unittest.TestCase):
    def scorer(self, mode, executor_weights):
        h=SETTINGS["prediction_steps"]
        instance=Scorer.__new__(Scorer)
        instance.snapshot_compatible=True
        instance.scores={};instance.state=SimpleNamespace(searcher_ids=(0,),executor_id=3)
        instance.context=SimpleNamespace(belief=np.ones(1));instance.arm=dict(ARMS["F1"],response=mode)
        instance.counts={"combination_scores":0};instance.timings={"score_seconds":0.}
        forecast=SimpleNamespace(cumulative_detection=np.array([[0.]]+[[.5]]*h),risk=(0.,0.,0.),positions=np.zeros((h+1,3)))
        instance.forecast=lambda allocation,i:forecast
        instance.full=lambda allocation:.8
        instance.coverage=lambda allocation:np.array([.8])
        instance.response_weights=lambda positions:np.asarray(executor_weights[:len(positions)]).reshape(-1,1)
        allocation=SimpleNamespace(search_assignments=(),executor_assignment=SimpleNamespace(target_region=(0,0,0)))
        return instance,allocation

    def test_repeat_observation_not_independent_and_response_uses_discovery_time(self):
        h=SETTINGS["prediction_steps"]
        instance,allocation=self.scorer("dynamic",[.2]+[1.]*(h-1))
        result=instance.score(allocation)
        self.assertAlmostEqual(result.near,.5)
        self.assertAlmostEqual(result.response,.1)
        # Later arrival of executor at a good standby cannot retroactively improve
        # response to a detection that happened at the first predicted step.
        static,allocation=self.scorer("static",[1.]*h)
        self.assertAlmostEqual(static.score(allocation).response,.8)

    def test_existing_coverage_is_not_new_near_term_mass(self):
        instance,allocation=self.scorer("sequential",[])
        f=instance.forecast(allocation,0)
        f.cumulative_detection[:]=.5
        self.assertEqual(instance.score(allocation).near,0.)

    def test_stale_public_snapshot_is_incomparable(self):
        instance,allocation=self.scorer("sequential",[])
        instance.snapshot_compatible=False
        with self.assertRaisesRegex(Incomparable,"stale_public_planning_snapshot"):
            instance.score(allocation)


class PublicEndpointTests(unittest.TestCase):
    def test_verified_continuous_connection_without_mutating_snapshot(self):
        state=filter_planning_state(fixture((4,)))
        point=(.2,.3,1.)
        before=planning_state_sha256(state)
        self.assertTrue(np.isinf(TravelCostService(state).travel_times_from(point,state.agents[3])).all())
        times,tested=connected_times(state,point,PublicGeometry(state),limit=128)
        self.assertTrue(np.isfinite(times).any())
        self.assertGreater(tested,0)
        self.assertEqual(planning_state_sha256(state),before)
        with self.assertRaisesRegex(Incomparable,"outside_public"):
            connected_times(state,(1.5,1.5,1),PublicGeometry(state),limit=128)

    def test_connector_budget_is_not_reported_as_physical_unreachability(self):
        state=filter_planning_state(fixture())
        with self.assertRaisesRegex(Incomparable,"budget_limited"):
            connected_times(state,(.2,.3,1),PublicGeometry(state),limit=0)

    def test_role_specific_prior_strength(self):
        spec=dict(v_xy_max=2.,v_z_max=1.,a_xy_max=1.,a_z_max=1.,drag_xy=.1,drag_z=.1,buoyancy_bias=0.)
        env=SimpleNamespace(agent_specs=[spec]*4,use_residual_prior=True,dt=.2,prior_kv_xy=1.,prior_kv_z=1.,
            prior_slow_radius_xy=1.,prior_slow_radius_z=1.,prior_strength_search=.5,prior_strength_executor=.9)
        self.assertEqual(vehicle_constants(env,0).strength,.5)
        self.assertEqual(vehicle_constants(env,3).strength,.9)


if __name__=="__main__":
    unittest.main()
