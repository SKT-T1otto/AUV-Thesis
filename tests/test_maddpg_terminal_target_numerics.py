import unittest
import torch
from torch import nn
from chapter3_bser.experiments.d2_repair_v2.model import build_model
from tests.d2_v2_test_utils import config,sample,state,delta,state_digest

class Constant(nn.Module):
    def __init__(self,value): super().__init__(); self.value=value
    def forward(self,x): return torch.full((len(x),1),self.value)

class NumericOracles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old=torch.get_num_threads(); cls.addClassCleanup(torch.set_num_threads,old); torch.set_num_threads(1)

    def test_bellman_terminal_nan_and_live_nan(self):
        for label in ("b2","b3"):
            m=build_model(config(label)); ag=m.agents[0]
            ag.target_critic1=Constant(2.); ag.target_critic2=Constant(3.)
            actual=m._compute_target_q(ag,torch.zeros(3,124),torch.tensor([.2,-2.,.3]),torch.tensor([0.,1.,0.]))
            expected=torch.tensor([[.2+.95*2],[-2.],[.3+.95*2]])
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            ag.target_critic1=Constant(float("nan")); ag.target_critic2=Constant(float("inf"))
            self.assertEqual(m._compute_target_q(ag,torch.zeros(1,124),torch.tensor([-2.]),torch.ones(1)).item(),-2.)
            before=state_digest(m.training_state_dict())
            with self.assertRaises(FloatingPointError): m.update(sample(),0)
            self.assertEqual(before,state_digest(m.training_state_dict()))

    def test_independent_huber_loss_and_parameter_derivative(self):
        m=build_model(config()); ag=m.agents[0]; s=sample(4,terminal=True)
        for part in (s[0],s[1],s[3]):
            for value in part: value.zero_()
        for value in s[2]: value.fill_(.25)
        # A known scalar Q on a disposable model makes the derivative observable.
        for name in ("critic1","critic2"):
            net=nn.Linear(124,1); nn.init.zeros_(net.weight); nn.init.zeros_(net.bias)
            setattr(ag,name,net); setattr(ag,name+"_optimizer",torch.optim.SGD(net.parameters(),lr=.1))
        loss,_=m.update_critic_only(s,0)
        residual=-.25
        huber=.5*residual**2 if abs(residual)<1 else abs(residual)-.5
        self.assertAlmostEqual(loss,2*huber,places=7)
        for net in (ag.critic1,ag.critic2):
            self.assertAlmostEqual(net.bias.item(),0.-.1*residual,places=7)
            self.assertEqual(net.weight.abs().max().item(),0.)

    def test_nonfinite_losses_and_gradients_fail_before_corresponding_steps(self):
        m=build_model(config()); ag=m.agents[0]
        before=[state(ag.critic1),state(ag.critic2)]
        handle=next(ag.critic2.parameters()).register_hook(lambda g:g*float("nan"))
        try:
            with self.assertRaises(RuntimeError): m.update(sample(),0)
        finally: handle.remove()
        self.assertEqual(delta(before[0],ag.critic1),0); self.assertEqual(delta(before[1],ag.critic2),0)
        s=sample(); handle=ag.critic1.register_forward_hook(lambda n,a,o:o*float("nan"))
        try:
            with self.assertRaises(FloatingPointError): m.update(s,0)
        finally: handle.remove()
        before=state(ag.policy); opt=state_digest(ag.policy_optimizer.state_dict())
        handle=next(ag.policy.parameters()).register_hook(lambda g:g*float("nan"))
        try:
            with self.assertRaises(RuntimeError): m.update(sample(),0)
        finally: handle.remove()
        self.assertEqual(delta(before,ag.policy),0); self.assertEqual(opt,state_digest(ag.policy_optimizer.state_dict()))
