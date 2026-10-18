import copy
import unittest
import torch
from chapter3_bser.experiments.d2_repair_v2.model import build_model
from chapter3_bser.experiments.baselines.common.model import boundary_features
from tests.d2_v2_test_utils import config,sample,state,delta,state_digest

class ActorMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old=torch.get_num_threads(); cls.addClassCleanup(torch.set_num_threads,old); torch.set_num_threads(1)

    def test_frozen_adam_state_unchanged_critic_still_learns(self):
        for label in ("b2","b3"):
            m=build_model(config(label)); m.update(sample(),0); m.update_all_targets()
            a=m.agents[0]; policy=state(a.policy); critic=state(a.critic1)
            optimizer=state_digest(a.policy_optimizer.state_dict()); counts=list(m.actor_updates)
            s=sample(frozen=True); self.assertFalse(bool(s[4][0].any()))
            m.update(s,0)
            self.assertFalse(m.last_actor_updated); self.assertEqual(policy.keys(),state(a.policy).keys())
            self.assertEqual(delta(policy,a.policy),0); self.assertGreater(delta(critic,a.critic1),0)
            self.assertEqual(optimizer,state_digest(a.policy_optimizer.state_dict())); self.assertEqual(counts,m.actor_updates)

    def test_known_q_direction_mask_gradient_and_parameter_ownership(self):
        for label in ("b2","b3"):
            torch.manual_seed(1301); m=build_model(config(label))
            torch.manual_seed(1301); reference=build_model(config(label))
            s=sample(); s.enabled[4:,0]=0
            target=state(m.agents[0].target_policy); others=[state(a.policy) for a in m.agents[1:]]
            before=float(m.step(s[0])[0][:,0].mean())
            hook=m.agents[0].critic1.register_forward_hook(lambda n,args,out:out*0+args[0][:,112:113])
            try: m.update(s,0)
            finally: hook.remove()
            reference._prepare_sample(s)
            policy=reference.agents[0].policy
            loss=-policy(s[0][0])[:4,0].mean()
            reference.agents[0].policy_optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),.2,error_if_nonfinite=True)
            reference.agents[0].policy_optimizer.step()
            self.assertLess(delta(state(policy),m.agents[0].policy),2e-7)
            self.assertGreater(float(m.step(s[0])[0][:,0].mean()),before)
            self.assertEqual(delta(target,m.agents[0].target_policy),0)
            self.assertTrue(all(delta(x,a.policy)==0 for x,a in zip(others,m.agents[1:])))
            if label=="b3":
                params=set(m.agents[0].policy.parameters())
                optimizer=set(p for g in m.agents[0].policy_optimizer.param_groups for p in g["params"])
                self.assertEqual(params,optimizer)
                features=boundary_features(torch.stack(s[3],dim=1))
                for i,p in enumerate(m.target_policies): torch.testing.assert_close(p._boundary_context,features[:,i])

    def test_soft_update_matches_scalar_formula_for_every_parameter(self):
        for label in ("b2","b3"):
            m=build_model(config(label)); s=sample()
            for i in range(4): m.update(s,i)
            pairs=[]
            for a in m.agents:
                for t,o in ((a.target_policy,a.policy),(a.target_critic1,a.critic1),(a.target_critic2,a.critic2)):
                    pairs += [(p,p.detach().clone(),q.detach().clone()) for p,q in zip(t.parameters(),o.parameters())]
            m.update_all_targets()
            for actual,old,online in pairs:
                torch.testing.assert_close(actual,old*.995+online*.005,rtol=2e-6,atol=6e-8)

    def test_optimizers_own_only_their_declared_parameters(self):
        for label in ('b2','b3'):
            m=build_model(config(label))
            sets=[]
            for a in m.agents:
                for net,opt in ((a.policy,a.policy_optimizer),(a.critic1,a.critic1_optimizer),(a.critic2,a.critic2_optimizer)):
                    parameters=set(net.parameters())
                    self.assertEqual(parameters,set(p for group in opt.param_groups for p in group['params']))
                    self.assertTrue(all(not parameters.intersection(other) for other in sets))
                    sets.append(parameters)
