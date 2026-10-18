import copy
import hashlib
import json
import unittest
import numpy as np
import torch
from chapter3_bser.experiments.baselines.common.model import (
    _contract_environment,BoundaryConditionedPolicy,BoundaryConditionedMADDPG,
)
from chapter3_bser.experiments.d2_repair_v2.model import build_model
from chapter3_bser.experiments.d2_repair_v2.runtime import RepairedMissionRuntime
from tests.d2_v2_test_utils import ROOT,config,sample,state,delta
from tests.d2_v2_scenarios import scenario

def parent_model(c,boundary):
    path=ROOT/'tests/fixtures/maddpg_before_repair_v2.py.txt'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='fbf983ec992613a45c10a2ced45725c17b2a412414d64167cb02b33ab8692a91'
    namespace={}; exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),namespace)
    Parent=namespace['MADDPG']
    class Boundary(Parent):
        _bind=staticmethod(BoundaryConditionedMADDPG._bind)
        def _build_target_actions(self,obs):
            self._bind(self.target_policies,obs); return super()._build_target_actions(obs)
        def _prepare_sample(self,s):
            batch=super()._prepare_sample(s); self._bind(self.policies,batch['obs']); return batch
    rl=c['rl']; cls=Boundary if boundary else Parent
    m=cls.init_from_env(_contract_environment(),gamma=rl['gamma'],tau=rl['tau'],lr_actor=rl['lr_actor'],
        lr_critic=rl['lr_critic'],hidden_dim=rl['hidden_dim'],residual_action_reg=0.)
    if boundary:
        for a in m.agents:
            a.policy=BoundaryConditionedPolicy(a.policy,c.get('boundary_encoder',{}).get('hidden_dim',32))
            a.target_policy=copy.deepcopy(a.policy)
            a.policy_optimizer=torch.optim.Adam(a.policy.parameters(),lr=rl['lr_actor'])
    return m

class ActionSource:
    def step(self,obs,explore=False): return [torch.full((1,3),.25) for _ in range(4)]

class IndependentSwitches(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old=torch.get_num_threads(); cls.addClassCleanup(torch.set_num_threads,old); torch.set_num_threads(1)

    def test_all_off_exact_parent_network_update_and_original_hyperparameters(self):
        for label in ('b2','b3'):
            c=config(label); old=json.loads((ROOT/f'configs/chapter3/d2_repair_v1/{label}_train.json').read_text(encoding='utf-8'))
            self.assertEqual(c['rl'],old['rl']); self.assertEqual(c['reward'],old['reward'])
            c['repairs']={k:False for k in c['repairs']}; c.pop('waypoint_reward_timing')
            torch.manual_seed(129); parent=parent_model(c,label=='b3')
            torch.manual_seed(129); child=build_model(c)
            s=sample()
            for i in range(4):
                a=parent.update(s,i); b=child.update(s,i)
                self.assertEqual(a[:2],b[:2])
            parent.update_all_targets(); child.update_all_targets()
            for a,b in zip(parent.agents,child.agents):
                for name in ('policy','critic1','critic2','target_policy','target_critic1','target_critic2'):
                    self.assertEqual(delta(state(getattr(a,name)),getattr(b,name)),0)

    def test_freeze_only_keeps_real_rewards_and_team_done(self):
        traces=[]
        for enabled in (False,True):
            c=config(); c['max_steps']=4
            c['repairs']=dict(waypoint_arrival=False,inactive_actor=enabled,hold_residual=False); c.pop('waypoint_reward_timing')
            rt=RepairedMissionRuntime(c,scenario('near',4),seed=31729); rows=[]
            try:
                while not rt.terminal: rows.append(rt.advance(ActionSource(),explore=False))
                traces.append(rows)
            finally: rt.close()
        for a,b in zip(*traces):
            np.testing.assert_array_equal(a['next_observations'],b['next_observations'])
            np.testing.assert_array_equal(a['rewards'],b['rewards']); self.assertEqual(a['dones'],b['dones'])
        np.testing.assert_array_equal(traces[1][2]['actions'][:3],0)
        self.assertFalse(any(traces[1][2]['dones']))

    def test_hold_only_changes_only_requested_normalized_residuals(self):
        results=[]
        for enabled in (False,True):
            c=config(); c['max_steps']=4
            c['repairs']=dict(waypoint_arrival=False,inactive_actor=False,hold_residual=enabled); c.pop('waypoint_reward_timing')
            rt=RepairedMissionRuntime(c,scenario('collision',4),seed=31729)
            try:
                requested=np.array([not x.reachable or x.hold_state for x in rt.guidance.agent_assignments])
                self.assertTrue(requested.any())
                row=rt.advance(ActionSource(),explore=False); results.append((row,requested))
            finally: rt.close()
        old,_=results[0]; new,mask=results[1]
        np.testing.assert_array_equal(new['actions'][mask],0)
        np.testing.assert_array_equal(new['actions'][~mask],old['actions'][~mask])
        np.testing.assert_array_equal(old['proposed_actions'],new['proposed_actions'])
