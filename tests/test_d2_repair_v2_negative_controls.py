"""Prove that repair oracles reject real, hash-pinned historical defects.

All historical code executes only in disposable test objects. No production
module is rewritten, and no checkpoint or formal training is used here.
"""
import copy
from functools import lru_cache
import hashlib
import json
import unittest

import numpy as np
import torch
from torch import nn

from core.algorithms.maddpg import MADDPG
from core.replay.ch3_buffer import CH3ReplayBuffer
from chapter3_bser.experiments.baselines.common.model import build_model as legacy_model
from chapter3_bser.experiments.d2_repair_v1.replay import ExecutionReplay as V1Replay
from chapter3_bser.experiments.d2_repair_v2.model import build_model
from chapter3_bser.experiments.d2_repair_v2.replay import ExecutionReplay
from chapter3_bser.experiments.d2_repair_v2.runtime import RepairedMissionRuntime
from tests.d2_v2_test_utils import ROOT,config,sample,mutate,state,state_digest,delta,evidence_dir
from tests.d2_v2_scenarios import scenario
from tests.test_d2_repair_v2_ablation import parent_model
from tests.test_waypoint_reward_timing import trace

FIXTURES={
    "pre_v1":("maddpg_before_repair.py.txt","a574285b33d9a82971cf8016ce93e79f1c90b07cca5236edf5deb7fcb9b6d907"),
    "replay":("ch3_buffer_before_repair_v2.py.txt","e22412e43e589962582c2a3ce727b167a6689e60223f6d64fd71ef1e9e5cfa42"),
}


@lru_cache(maxsize=None)
def historical_class(kind):
    filename,expected=FIXTURES[kind]
    path=ROOT/'tests/fixtures'/filename
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError("historical negative-control fixture changed: "+filename)
    namespace={"__name__":"tests.frozen_negative_control_"+kind}
    exec(compile(raw,str(path),'exec'),namespace)
    return namespace['CH3ReplayBuffer' if kind=='replay' else 'MADDPG']


def raw_model(cls,*,local=False):
    rl=config()['rl']
    return cls([dict(num_in_pol=28,num_out_pol=3,num_in_critic=31 if local else 124)]*4,
        ['DDPG' if local else 'MADDPG']*4,gamma=rl['gamma'],tau=rl['tau'],
        lr_actor=rl['lr_actor'],lr_critic=rl['lr_critic'],hidden_dim=rl['hidden_dim'],residual_action_reg=0.)


def replay(cls,capacity):
    return cls(capacity,4,[28]*4,[3]*4,alpha=0.,beta_start=0.)


def push(buffer,k):
    buffer.push([torch.full((28,),float(k)) for _ in range(4)],
        [torch.zeros(3) for _ in range(4)],[float(k)]*4,
        [torch.full((28,),float(k+1)) for _ in range(4)],[False]*4,[False]*4)


class Constant(nn.Module):
    def __init__(self,value): super().__init__(); self.value=value
    def forward(self,x): return torch.full((len(x),1),self.value)


class KnownAction:
    def step(self,observations,explore=False): return [torch.full((1,3),.37) for _ in range(4)]


class NegativeControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threads=torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads,threads)
        torch.set_num_threads(1)
        cls.observed=[]

    @classmethod
    def tearDownClass(cls):
        (evidence_dir()/'negative_controls.json').write_text(
            json.dumps(cls.observed,indent=2)+'\n',encoding='utf-8')

    def setUp(self): torch.manual_seed(61729)

    def record(self,defect,**witness):
        self.observed.append(dict(defect=defect,old_control_detected=True,
                                  current_repair_passed=True,**witness))

    def test_batch_broadcast_oracle_rejects_frozen_parent(self):
        for label in ('b2','b3'):
            old=parent_model(config(label),label=='b3'); current=build_model(config(label))
            def oracle(model):
                bad=mutate(sample(7),2,[torch.ones(1) for _ in range(4)])
                bad=mutate(bad,5,torch.ones(1))
                before=state_digest(model.training_state_dict())
                with self.assertRaises(ValueError): model.update(bad,0)
                self.assertEqual(before,state_digest(model.training_state_dict()))
            oracle(current)
            with self.assertRaises(AssertionError): oracle(old)
            self.record('batch_broadcast',algorithm=label,batch=7,bad_reward_and_weight_batch=1)

    def test_capacity_restore_oracle_rejects_old_loader(self):
        def objects(cls):
            source=replay(cls,3); target=replay(cls,5)
            for i in range(5): push(source,i); push(target,-9876)
            return source,target
        source,current=objects(CH3ReplayBuffer)
        before=state_digest(current.state_dict())
        with self.assertRaises(ValueError): current.load_state_dict(source.state_dict())
        self.assertEqual(before,state_digest(current.state_dict()))
        source,old=objects(historical_class('replay'))
        def oracle():
            with self.assertRaises(ValueError): old.load_state_dict(source.state_dict())
        with self.assertRaises(AssertionError): oracle()
        push(old,5)
        ids=old.sample(4,norm_rews=False)[0][0][:,0].tolist()
        self.assertIn(-9876.,ids)
        self.record('capacity_restore',old_sampled_unwritten_sentinel=True,current_rejected_atomically=True)

    def test_execution_push_oracle_rejects_v1_bypass(self):
        def oracle(buffer):
            with self.assertRaises(TypeError): push(buffer,0)
            self.assertEqual(len(buffer),0)
        current=ExecutionReplay(3); old=V1Replay(3)
        oracle(current)
        with self.assertRaises(AssertionError): oracle(old)
        self.assertEqual(len(old),1)
        self.record('execution_push_bypass',v1_accepted_untagged_transition=True)

    def test_cpu_snapshot_oracle_rejects_parent_alias(self):
        for label in ('b2','b3'):
            def oracle(model):
                saved=model.training_state_dict(); before=state_digest(saved)
                with torch.no_grad(): next(model.agents[0].policy.parameters()).add_(.125)
                self.assertEqual(before,state_digest(saved))
            oracle(build_model(config(label)))
            with self.assertRaises(AssertionError): oracle(parent_model(config(label),label=='b3'))
            self.record('cpu_snapshot_alias',algorithm=label)

    def test_step_count_oracle_rejects_parent_zip_truncation(self):
        # This defect is the raw shared-core API. B3 has a separate joint
        # boundary-context guard, so do not confuse that guard with core.step.
        def oracle(model):
            with self.assertRaises(ValueError): model.step([torch.zeros(28)]*3)
        oracle(build_model(config()))
        old=parent_model(config(),False)
        with self.assertRaises(AssertionError): oracle(old)
        self.assertEqual(len(old.step([torch.zeros(28)]*3)),3)
        self.record('step_zip_truncation',scope='raw core API')

    def test_critic_only_flag_oracle_rejects_parent_stale_flag(self):
        for label in ('b2','b3'):
            def oracle(model):
                model.update(sample(),0)
                self.assertTrue(model.last_actor_updated)
                model.update_critic_only(sample(),0)
                self.assertFalse(model.last_actor_updated)
            oracle(build_model(config(label)))
            with self.assertRaises(AssertionError): oracle(parent_model(config(label),label=='b3'))
            self.record('critic_only_stale_flag',algorithm=label)

    def test_waypoint_bonus_oracle_rejects_legacy_timing(self):
        old=trace('normal',False,True); current=trace('normal',True,True)
        self.assertEqual(current[0]['count'],1)
        bonus=current[0]['bonus']; self.assertGreater(bonus,0)
        expected=old[0]['components']['reward_coverage']+torch.tensor([bonus,0.,0.,0.])
        def oracle(rows):
            torch.testing.assert_close(rows[0]['components']['reward_coverage'],expected,rtol=0,atol=0)
        oracle(current)
        with self.assertRaises(AssertionError): oracle(old)
        self.record('waypoint_timing',arrival_step=1,expected_increment=bonus,real_environment=True)

    def test_frozen_actor_oracle_rejects_legacy_training(self):
        for label in ('b2','b3'):
            def oracle(model):
                model.update(sample(),0)  # Give Adam nonzero moments first.
                a=model.agents[0]; policy=state(a.policy); critic=state(a.critic1)
                opt=state_digest(a.policy_optimizer.state_dict()); batch=sample(frozen=True)
                self.assertFalse(bool(batch[4][0].any()))
                model.update(batch,0)
                self.assertEqual(delta(policy,a.policy),0)
                self.assertEqual(opt,state_digest(a.policy_optimizer.state_dict()))
                self.assertGreater(delta(critic,a.critic1),0)
            oracle(build_model(config(label)))
            old=legacy_model(config(label))
            with self.assertRaises(AssertionError): oracle(old)
            self.record('inactive_actor',algorithm=label,adam_warmed=True,team_terminal=False)

    def test_hold_action_oracle_rejects_disabled_gate(self):
        rows=[]
        for enabled in (False,True):
            c=config(); c.update(max_steps=4)
            c['repairs']=dict(waypoint_arrival=False,inactive_actor=False,hold_residual=enabled)
            c.pop('waypoint_reward_timing')
            runtime=RepairedMissionRuntime(c,scenario('collision',4),seed=61729)
            try:
                mask=np.array([not a.reachable or a.hold_state for a in runtime.guidance.agent_assignments])
                self.assertTrue(mask.any())
                row=runtime.advance(KnownAction(),explore=False)
                rows.append((row,mask))
            finally: runtime.close()
        def oracle(item):
            row,mask=item; np.testing.assert_array_equal(row['actions'][mask],0)
        oracle(rows[1])
        with self.assertRaises(AssertionError): oracle(rows[0])
        self.record('hold_residual',real_environment=True)

    def test_terminal_nan_oracle_rejects_pre_v1_target(self):
        current=build_model(config()); a=current.agents[0]
        a.target_critic1=Constant(float('nan')); a.target_critic2=Constant(float('inf'))
        args=(a,torch.zeros(1,124),torch.tensor([-2.]),torch.ones(1))
        def oracle(value):
            self.assertTrue(bool(torch.isfinite(value).all()))
            self.assertEqual(value.item(),-2.)
        oracle(current._compute_target_q(*args))
        with self.assertRaises(AssertionError):
            oracle(historical_class('pre_v1')._compute_target_q(current,*args))
        self.record('terminal_nan_mask',terminal_reward=-2.)

    def test_mutable_sample_oracle_rejects_pre_v1_cache(self):
        def oracle(model):
            batch=sample()
            model._prepare_sample(batch)
            batch[0][0][:,0].add_(3.)
            actual=model._prepare_sample(batch)['joint_vf_in'][:,:112]
            torch.testing.assert_close(actual,torch.cat(batch[0],dim=1),rtol=0,atol=0)
        oracle(build_model(config()))
        with self.assertRaises(AssertionError): oracle(raw_model(historical_class('pre_v1')))
        self.record('mutable_sample_cache',same_object_mutated=True)

    def test_local_ddpg_oracle_rejects_pre_v1_joint_input(self):
        def oracle(model):
            try:
                result=model.update(sample(),0)
            except RuntimeError as exc:
                self.fail('valid 31D local Critic path raised: '+str(exc))
            self.assertTrue(np.isfinite(result[:2]).all())
        oracle(raw_model(MADDPG,local=True))
        with self.assertRaises(AssertionError): oracle(raw_model(historical_class('pre_v1'),local=True))
        self.record('local_ddpg_input',critic_dim=31,not_used_by_b2_b3=True)

    def test_nonfinite_gradient_oracle_rejects_pre_v1_optimizer_write(self):
        def oracle(model):
            a=model.agents[0]; before=state_digest([a.critic1.state_dict(),a.critic2.state_dict()])
            handle=next(a.critic2.parameters()).register_hook(lambda g:g*float('nan'))
            try:
                with self.assertRaises(RuntimeError): model.update(sample(),0)
            finally: handle.remove()
            self.assertEqual(before,state_digest([a.critic1.state_dict(),a.critic2.state_dict()]))
        oracle(build_model(config()))
        old=raw_model(historical_class('pre_v1'))
        with self.assertRaises(AssertionError): oracle(old)
        self.assertTrue(any(not bool(torch.isfinite(p).all()) for p in old.agents[0].critic2.parameters()))
        self.record('nonfinite_optimizer_write',old_critic_became_nonfinite=True)


if __name__=='__main__': unittest.main(verbosity=2)
