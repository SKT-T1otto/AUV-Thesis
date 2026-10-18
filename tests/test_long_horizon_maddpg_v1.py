"""Bounded learning/discount contracts, never formal performance experiments."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from chapter3_bser.experiments.long_horizon_maddpg_v1 import cli, config, provenance
from chapter3_bser.experiments.long_horizon_maddpg_v1.model import FieldScale, build_model
from chapter3_bser.experiments.long_horizon_maddpg_v1.runtime import MissionRuntime

ROOT=Path(__file__).resolve().parents[1]


def settings():
    c=json.loads((ROOT/'configs/chapter3/long_horizon_maddpg_v1.json').read_text())
    c['rl'].update(hidden_dim=16,batch_size=4,replay_size=32,warmup_steps=0,
                   critic_warmup_updates=0,actor_delay=1)
    return config.validate(c)


def sample(finished=False):
    values=torch.randn(8,4,28)
    values[...,20]=0.
    values[...,22:26]=torch.eye(4).unsqueeze(0)
    values[...,26]=1.
    values[...,27]=0.
    if finished:values[:,0,20]=1.
    return ([values[:,i].clone() for i in range(4)],
            [torch.randn(8,3).clamp(-1,1) for _ in range(4)],
            [torch.full((8,),.25) for _ in range(4)],
            [values[:,i].clone() for i in range(4)],
            [torch.zeros(8) for _ in range(4)],torch.ones(8),torch.arange(8),torch.zeros(8,dtype=torch.bool))


def cloned(net):
    return {k:v.clone() for k,v in net.state_dict().items()}


class Constant(nn.Module):
    def __init__(self,value):
        super().__init__();self.value=value
    def forward(self,x):return torch.full((x.shape[0],1),self.value,dtype=x.dtype,device=x.device)


class LearningContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):torch.manual_seed(2729)

    def test_discount_retains_delayed_discovery(self):
        c=settings();g=c['rl']['gamma']
        self.assertAlmostEqual(g**400,.5,places=12)
        self.assertGreater(g**150,.77)
        self.assertGreater(g**150/(.95**150),1600)
        self.assertTrue(config.discount_report(c)['changed_optimization_objective'])

    def test_inconsistent_gamma_and_insufficient_retention_rejected(self):
        for change in ('gamma','retention'):
            c=settings()
            if change=='gamma':c['rl']['gamma']=.95
            else:c['discount']['retention']=.4
            with self.assertRaises(ValueError):config.validate(c)
        for retention in (0,1,float('nan')):
            with self.assertRaises(ValueError):config.discount_from_retention(400,retention)

    def test_contracts_and_historical_weight_identity_not_relabelled(self):
        for name,value in (('observation_dim',29),('critic_input_dim',125),('method','ch3_baseline_direct_mc')):
            c=settings();c[name]=value
            with self.assertRaises(ValueError):config.validate(c)
        c=settings();c['reward']['terminal_success_bonus']=20.
        with self.assertRaises(ValueError):config.validate(c)

    def test_shapes_initialization_and_target_equality(self):
        c=json.loads((ROOT/'configs/chapter3/long_horizon_maddpg_v1.json').read_text())
        m=build_model(c)
        for a in m.agents:
            self.assertEqual((a.policy.fc1.in_features,a.policy.fc3.out_features,a.critic1.fc1.in_features),(28,3,124))
            self.assertEqual((a.policy.fc1.out_features,a.policy.fc2.out_features),(128,128))
            self.assertEqual((a.critic1.fc1.out_features,a.critic1.fc2.out_features),(256,256))
            for original,target in ((a.policy,a.target_policy),(a.critic1,a.target_critic1),(a.critic2,a.target_critic2)):
                for k,v in original.state_dict().items():torch.testing.assert_close(v,target.state_dict()[k],rtol=0,atol=0)
                self.assertTrue(all(not p.requires_grad for p in target.parameters()))
            self.assertLess(float(a.policy(torch.randn(32,28)).detach().abs().max()),.05)

    def test_field_scaling_preserves_flags_and_action_units(self):
        x=torch.ones(2,28)
        scaled=FieldScale()(x)
        torch.testing.assert_close(scaled[:,20:28],x[:,20:28])
        self.assertAlmostEqual(float(scaled[0,0]),.05,places=7)
        joint=FieldScale(joint=True)(torch.ones(2,124))
        torch.testing.assert_close(joint[:,112:],torch.ones(2,12))
        with self.assertRaises(ValueError):FieldScale()(torch.ones(2,29))

    def test_train_eval_have_no_dropout_or_batchnorm_drift(self):
        for mode in ('field_scale','legacy_layernorm'):
            c=settings();c['rl']['input_scaling']=mode
            m=build_model(c);x=sample()[0]
            m.prep_training(device='cpu');a=m.step(x,explore=False)
            m.prep_rollouts(device='cpu');b=m.step(x,explore=False)
            for left,right in zip(a,b):torch.testing.assert_close(left,right,rtol=0,atol=0)

    def test_long_horizon_target_not_clipped_at_ten(self):
        m=build_model(settings());a=m.agents[0]
        a.target_critic1=Constant(20.);a.target_critic2=Constant(30.)
        target=m._compute_target_q(a,torch.zeros(2,124),torch.tensor([-2.,.25]),torch.tensor([1.,0.]))
        self.assertEqual(float(target[0]),-2.)
        self.assertGreater(float(target[1]),20.)
        self.assertAlmostEqual(float(target[1]),.25+m.gamma*20,places=5)

    def test_terminal_target_does_not_propagate_nan_and_nonterminal_fails(self):
        m=build_model(settings());a=m.agents[0]
        a.target_critic1=Constant(float('nan'));a.target_critic2=Constant(float('nan'))
        torch.testing.assert_close(m._compute_target_q(a,torch.zeros(1,124),torch.tensor([-2.]),torch.ones(1)),torch.tensor([[-2.]]))
        with self.assertRaises(FloatingPointError):m._compute_target_q(a,torch.zeros(1,124),torch.zeros(1),torch.zeros(1))

    def test_inactive_actor_does_not_drift_after_adam_has_momentum(self):
        m=build_model(settings())
        m.update(sample(),0);self.assertEqual(m.actor_updates[0],1)
        before=cloned(m.agents[0].policy)
        m.update(sample(finished=True),0)
        self.assertFalse(m.last_update['actor_updated'])
        self.assertEqual(m.actor_updates[0],1)
        for k,v in before.items():torch.testing.assert_close(v,m.agents[0].policy.state_dict()[k],rtol=0,atol=0)

    def test_frozen_actions_canonicalized_without_cutting_team_bootstrap(self):
        m=build_model(settings());s=sample(finished=True)
        b=m._prepare_sample(s)
        self.assertEqual(int(torch.count_nonzero(b['acs'][0])),0)
        self.assertEqual(int(torch.count_nonzero(b['next_acs'][0])),0)
        self.assertEqual(int(torch.count_nonzero(b['dones'][0])),0)
        self.assertTrue(torch.equal(m.step(s[0],explore=True)[0],torch.zeros(8,3)))
        a=m.agents[0];a.target_critic1=Constant(1.);a.target_critic2=Constant(1.)
        self.assertGreater(float(m._compute_target_q(a,b['target_vf_in'],b['rews'][0],b['dones'][0]).mean()),1.)

    def test_critic_only_warmup_then_actor_delay(self):
        c=settings();c['rl'].update(critic_warmup_updates=2,actor_delay=2)
        m=build_model(c)
        updated=[]
        for _ in range(4):
            m.update(sample(),0);updated.append(m.last_update['actor_updated']);m.update_all_targets()
        self.assertEqual(updated,[False,False,False,True])

    def test_gradients_are_finite_and_other_actors_and_critics_not_leaked(self):
        m=build_model(settings());others=[cloned(a.policy) for a in m.agents[1:]]
        m.update(sample(),0)
        for i,a in enumerate(m.agents[1:]):
            for k,v in others[i].items():torch.testing.assert_close(v,a.policy.state_dict()[k],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in m.agents[0].critic1.parameters()))
        self.assertTrue(all(p.grad is None for a in m.agents for p in a.target_policy.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.agents[0].policy.parameters()))

    def test_actor_gradient_maximizes_known_q(self):
        m=build_model(settings());s=sample();a=m.agents[0]
        hook=a.critic1.register_forward_hook(lambda module,args,out:out*0+args[0][:,112:113])
        before=m.step(s[0])[0][:,0].mean()
        try:m.update(s,0)
        finally:hook.remove()
        self.assertGreater(float(m.step(s[0])[0][:,0].mean()),float(before))

    def test_target_polyak_formula(self):
        c=settings();c['rl']['tau']=.25
        m=build_model(c);a=m.agents[0]
        with torch.no_grad():
            for p in a.policy.parameters():p.fill_(2.)
            for p in a.target_policy.parameters():p.fill_(0.)
            for p in a.critic1.parameters():p.fill_(4.)
            for p in a.target_critic1.parameters():p.fill_(0.)
        m.update_all_targets()
        self.assertTrue(all(torch.equal(p,torch.full_like(p,.5)) for p in a.target_policy.parameters()))
        self.assertTrue(all(torch.equal(p,torch.ones_like(p)) for p in a.target_critic1.parameters()))

    def test_noise_schedule_and_no_noise_evaluation(self):
        m=build_model(settings());base=list(m.base_sigmas)
        for t,factor in ((0,1.),(50000,.625),(100000,.25),(200000,.25)):
            self.assertEqual(m.exploration_at(t),[v*factor for v in base])
        obs=sample()[0]
        with patch('core.algorithms.noise.OUNoise.sample',side_effect=AssertionError('eval noise')):
            m.step(obs,explore=False)

    def test_checkpoint_state_roundtrip_and_settings_mismatch(self):
        c=settings();m=build_model(c);s=sample();m.update(s,0);m.update_all_targets()
        state=copy.deepcopy(m.training_state_dict());other=build_model(c);other.load_training_state_dict(state)
        self.assertEqual(cli.state_digest(state),cli.state_digest(other.training_state_dict()))
        for a,b in zip(m.step(s[0]),other.step(s[0])):torch.testing.assert_close(a,b,rtol=0,atol=0)
        state['repair']['actor_delay']=99
        with self.assertRaises(ValueError):other.load_training_state_dict(state)

    def test_whole_400_step_backwards_event_signal(self):
        m=build_model(settings());a=m.agents[0]
        # Exercise the real target implementation for a delayed terminal signal.
        value=torch.tensor([[1.]])
        for _ in range(400):
            a.target_critic1=Constant(float(value));a.target_critic2=Constant(float(value))
            value=m._compute_target_q(a,torch.zeros(1,124),torch.zeros(1),torch.zeros(1))
        # 400 FP32 multiplications accumulate about 6e-6 absolute roundoff.
        torch.testing.assert_close(value,torch.tensor([[.5]]),rtol=0,atol=1e-5)

    def test_original_learning_and_reward_files_remain_frozen(self):
        manifest=json.loads((ROOT/'docs/provenance/d2_performance_v1_evolution.json').read_text())
        expected=manifest['profiles']['git_clone_preserved']['files']
        for name in ('core/algorithms/maddpg.py','core/algorithms/agents.py','core/algorithms/networks.py',
                     'core/env/uav_env.py','chapter3_bser/experiments/baselines/common/model.py',
                     'chapter3_bser/experiments/baselines/common/train.py'):
            from tests.d2_repair_v2_review import successor
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),successor(name,expected[name]))


class RuntimeAndGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_zero_action_d2_rollout_matches_old_runtime_and_rewards(self):
        from chapter3_bser.experiments.baselines.common.runtime import BaselineMissionRuntime
        old=json.loads((ROOT/'configs/chapter3/d2_v1/direct_mc_train.json').read_text())
        new=settings()
        scene=json.loads((ROOT/'tests/fixtures/hgr/handoff_manifest.json').read_text())['scenarios'][0]
        scene['max_steps']=400
        class Zero:
            def step(self,obs,explore=False):return [torch.zeros(1,3) for _ in range(4)]
        traces=[]
        for klass,cfg in ((BaselineMissionRuntime,old),(MissionRuntime,new)):
            runtime=klass(cfg,scene,seed=12729)
            rows=[]
            try:
                for _ in range(8):
                    if runtime.terminal:break
                    row=runtime.advance(Zero(),explore=False)
                    rows.append((row['observations'],row['actions'],row['rewards'],row['next_observations'],row['dones']))
                self.assertEqual(runtime.env.reward_accounting.gamma,cfg['rl']['gamma'])
            finally:runtime.close()
            traces.append(rows)
        self.assertEqual(len(traces[0]),len(traces[1]))
        for a,b in zip(*traces):
            for x,y in zip(a,b):np.testing.assert_array_equal(np.asarray(x),np.asarray(y))

    def test_preview_never_builds_model_or_runtime(self):
        scene=json.loads((ROOT/'tests/fixtures/hgr/handoff_manifest.json').read_text())['scenarios'][0]
        scene['max_steps']=400
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            p=root/'train.json';p.write_text(json.dumps(dict(scenarios=[scene])))
            out=root/'long_horizon_maddpg_v1/collision_terminal/train'
            with patch.object(cli,'build_model',side_effect=AssertionError('model')), \
                 patch.object(cli,'MissionRuntime',side_effect=AssertionError('runtime')):
                result=cli.train(settings(),p,out,4,execute=False)
            self.assertFalse(result['executed']);self.assertFalse(out.exists())

    def test_manifest_disjointness_and_output_guards(self):
        a=[dict(id='a',seed=1,content='x')]
        b=[dict(id='b',seed=2,content='y')]
        cli.check_disjoint(a,b)
        for key in ('id','seed','content'):
            changed=copy.deepcopy(b);changed[0][key]=a[0][key]
            with self.assertRaises(ValueError):cli.check_disjoint(a,changed)
        with self.assertRaises(ValueError):cli.new_output(ROOT/'outputs/old_baseline')

    def test_source_gate_rejects_foreign_additions_without_weakening_parent(self):
        parent=json.loads((ROOT/provenance.PARENT).read_text())
        files=parent['profiles']['windows_existing']['files']
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name in set(files)|provenance.ADDED|{provenance.PARENT,provenance.SEAL}:
                dst=root/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dst)
            self.assertTrue(provenance.preflight(root)['ready'])
            foreign=root/'chapter3_bser/unreviewed.py';foreign.write_text('')
            result=provenance.preflight(root)
            self.assertFalse(result['ready']);self.assertEqual(result['unreviewed_additions'],['chapter3_bser/unreviewed.py'])
            original=root/'core/algorithms/maddpg.py';original.write_bytes(original.read_bytes()+b'\n# changed\n')
            with self.assertRaisesRegex(ValueError,'frozen parent'):provenance.preflight(root)


if __name__=='__main__':unittest.main()
