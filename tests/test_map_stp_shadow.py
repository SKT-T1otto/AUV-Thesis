"""Scientific-contract tests for the isolated, non-installing shadow screen."""
from dataclasses import replace
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from chapter3_bser.experiments.map_stp_shadow_v1.io import (
    digest, load_snapshot, plain, read, save_snapshot, state_from_dict,
)
from chapter3_bser.experiments.map_stp_shadow_v1.routes import Route, route_pool
from chapter3_bser.experiments.map_stp_shadow_v1.run import config, summarize
from chapter3_bser.experiments.map_stp_shadow_v1.scoring import (
    compare_case, first_values, hypotheses, marginal_utility, swept_events, team_scores,
)
from core.env.target_motion import swept_relative_min_distance
from core.mapping.planning_state import planning_state_sha256
from tests.bser_test_utils import synthetic_state


class ScientificContracts(unittest.TestCase):
    def test_vector_detector_equals_shared_scalar_including_zero_relative_motion(self):
        rng=np.random.default_rng(17)
        agent=rng.normal(size=(8,3)); targets=rng.normal(size=(13,8,3))
        targets[0]=agent+[2.,0.,0.]
        actual=swept_events(agent,targets,2.1)
        expected=np.array([[swept_relative_min_distance(agent[k],agent[k+1],x[k],x[k+1])[0]<=2.1
                            for x in targets] for k in range(7)])
        np.testing.assert_array_equal(actual,expected)

    def test_shared_eps_zero_denominator_branch_and_closed_radius(self):
        agent=np.zeros((2,3)); targets=np.array([[[2.1,0.,0.],[2.1,0.,0.]]])
        self.assertTrue(swept_events(agent,targets,2.1)[0,0])
        self.assertFalse(swept_events(agent,targets,2.09)[0,0])

    def test_equal_marginals_do_not_identify_first_detection_probability(self):
        repeated=np.array([[1,0],[1,0]],dtype=bool)
        disjoint=np.array([[1,0],[0,1]],dtype=bool)
        self.assertEqual(marginal_utility(repeated,.2,0)[0],.75)
        self.assertEqual(marginal_utility(disjoint,.2,0)[0],.75)
        self.assertEqual(first_values(repeated,.2,0).mean(),.5)
        self.assertEqual(first_values(disjoint,.2,0).mean(),1.)

    def test_repeated_detection_is_counted_once_at_earliest_step(self):
        events=np.array([[0,0],[1,0],[1,1]],dtype=bool)
        np.testing.assert_allclose(first_values(events,.2,.1),np.exp(-.1*.2*np.array([2,3])))

    def test_team_identical_routes_do_not_duplicate_benefit(self):
        e=np.array([[0,1],[1,1]],dtype=bool)
        one=team_scores([e],[e],[2.],.2,.04)
        two=team_scores([e,e],[e,e],[2.,2.],.2,.04)
        self.assertEqual(one,two)

    def test_heldout_evaluation_can_reject_selection_winner(self):
        route=lambda name:Route(name,'test',np.zeros((2,3)),np.array([0,1]),1.,1.,1.)
        selection=[np.array([[0,0],[1,1]],bool),np.array([[1,1],[1,1]],bool)]
        validation=[np.ones((2,2),bool),np.zeros((2,2),bool)]
        settings=dict(particles=2,discount_per_second=.1,bootstrap_repetitions=10)
        result=compare_case([[route('a'),route('b')]],[selection],[selection],
                            [validation],[validation],.2,settings,1)
        self.assertEqual(result['choices']['correlated_first'],[1])
        self.assertEqual(result['choices']['static'],[0])
        self.assertLess(result['comparisons']['static']['validation_delta'],0)

    def test_routes_preserve_same_endpoints_existing_costs_and_snapshot(self):
        state=synthetic_state(); before=planning_state_sha256(state)
        settings=dict(planner_clearance=0.,max_cost_ratio=1.4,via_limit=12,routes_per_agent=4)
        routes,counts=route_pool(state,0,state.agents[1].position,settings)
        self.assertTrue(routes); self.assertLessEqual(len(routes),4)
        self.assertEqual(routes[0].source,'unchanged_astar_query')
        for route in routes:
            np.testing.assert_allclose(route.points[0],state.agents[0].position)
            np.testing.assert_allclose(route.points[-1],state.agents[1].position)
            self.assertLessEqual(route.planning_cost,routes[0].planning_cost*1.4+1e-9)
        self.assertEqual(before,planning_state_sha256(state))
        self.assertGreater(counts['queries'],0)

    def test_public_state_json_roundtrip_preserves_identity_and_readonly_arrays(self):
        original=synthetic_state(); restored=state_from_dict(plain(original))
        self.assertEqual(planning_state_sha256(original),planning_state_sha256(restored))
        self.assertFalse(restored.target_belief.probabilities.flags.writeable)
        payload=dict(schema='ch3.map_stp.public_snapshot.v1',identity={},state=original,
                     public={},goals={},vehicles={},state_sha256=planning_state_sha256(original))
        with tempfile.TemporaryDirectory(dir='runs') as directory:
            path=Path(directory)/'state.json.gz'; save_snapshot(path,payload)
            self.assertEqual(planning_state_sha256(load_snapshot(path)['state']),planning_state_sha256(original))
            altered=json.loads(gzip.decompress(path.read_bytes())); altered['hidden_target']=[1,2,3]
            path.write_bytes(gzip.compress(json.dumps(altered).encode()))
            with self.assertRaises(ValueError): load_snapshot(path)

    def test_config_cannot_enable_training_or_unbounded_collection(self):
        c=read('configs/chapter3/map_stp_shadow_v1.json')
        with tempfile.TemporaryDirectory(dir='runs') as directory:
            path=Path(directory)/'config.json'
            for override in (dict(training=True),dict(max_collection_steps=400),dict(particles=20000)):
                path.write_text(json.dumps(dict(c,**override)),encoding='utf-8')
                with self.assertRaises(ValueError):config(path)

    def test_hypotheses_are_repeatable_use_only_shared_motion_and_condition_nondetection(self):
        state=synthetic_state()
        # Small test geometry has huge real sensor radii; use explicit public test radii.
        agents=tuple(replace(a,sensor_radius=.05) if a.agent_id<3 else a for a in state.agents)
        state=replace(state,agents=agents)
        settings=dict(particles=32,horizon_steps=4,target_speed_prior=[.25,.55])
        public=dict(dt=.2,bounds=[3.,3.,2.],target_clearance=0.,max_reflections=4,
                    detect_eps_bias=.1,remaining_steps=20)
        from core.env.target_motion import simulate_target_trajectory
        with patch('chapter3_bser.experiments.map_stp_shadow_v1.scoring.simulate_target_trajectory',
                   wraps=simulate_target_trajectory) as motion:
            a,identity=hypotheses(state,settings,public,27)
            self.assertEqual(motion.call_count,32)
        b,other=hypotheses(state,settings,public,27)
        np.testing.assert_array_equal(a,b); self.assertEqual(identity,other)
        for agent in state.agents[:3]:
            self.assertTrue(np.all(np.linalg.norm(a[:,0]-agent.position,axis=1)>.15))
        self.assertFalse(identity['posterior_claim'])

    def test_insufficient_opportunity_cannot_pass_continuation_gate(self):
        c=read('configs/chapter3/map_stp_shadow_v1.json')
        self.assertEqual(summarize([],c)['decision'],'insufficient_route_opportunity')


if __name__=='__main__': unittest.main()
