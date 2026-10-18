import unittest
import torch
from chapter3_bser.experiments.d2_repair_v2.replay import ExecutionReplay
from chapter3_bser.experiments.d2_repair_v2.model import build_model
from tests.d2_v2_test_utils import config,sample,state_digest

def transition(k):
    obs=torch.zeros(4,28); obs[:,0]=k
    gate=torch.tensor([bool(k%2),True,True,True])
    return dict(observations=obs,next_observations=obs+1,actions=torch.zeros(4,3),rewards=torch.zeros(4),
                dones=[False]*4,action_enabled=gate,next_action_enabled=~gate)

class ExecutionGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old=torch.get_num_threads(); cls.addClassCleanup(torch.set_num_threads,old); torch.set_num_threads(1)

    def test_public_bypass_is_rejected_and_ring_slots_are_complete(self):
        r=ExecutionReplay(3)
        with self.assertRaises(TypeError): r.push([],[],[],[],[],[])
        self.assertEqual(len(r),0)
        for k in range(5): r.push_transition(transition(k))
        s=r.sample(3)
        self.assertEqual(sorted(s[0][0][:,0].tolist()),[2.,3.,4.])
        self.assertTrue(torch.equal(s.enabled[:,0],s[0][0][:,0].remainder(2).bool()))
        self.assertTrue(torch.equal(s.next_enabled,~s.enabled))
        r.gate_initialized[1]=False
        with self.assertRaises(ValueError): r.sample(3)
        with self.assertRaises(NotImplementedError): r.state_dict()

    def test_bad_transition_never_partially_writes(self):
        r=ExecutionReplay(3); r.push_transition(transition(1))
        for key,value in (("action_enabled",torch.ones(3)),("next_action_enabled",torch.ones(4)*2),
                          ("rewards",torch.ones(1)),("observations",torch.zeros(4,27)),
                          ("actions",torch.ones(4,3)*2),("dones",[True,False,False,False])):
            bad=transition(2); bad[key]=value
            old=(len(r),r.next_idx,r.enabled.clone(),r.gate_initialized.clone())
            with self.assertRaises(ValueError): r.push_transition(bad)
            self.assertEqual((len(r),r.next_idx),old[:2]); self.assertTrue(torch.equal(old[2],r.enabled)); self.assertTrue(torch.equal(old[3],r.gate_initialized))

    def test_invalid_sample_gates_rejected_before_optimizer(self):
        for label in ("b2","b3"):
            m=build_model(config(label))
            bads=[sample(),sample(),sample(),sample(frozen=True),sample(terminal=True)]
            bads[0].enabled=torch.ones(8,3)
            bads[1].next_enabled.fill_(2)
            bads[2].enabled[:,0]=0; bads[2][1][0].fill_(.1)
            bads[3].enabled[:,0]=1
            bads[4].next_enabled.fill_(1)
            for s in bads:
                before=state_digest(m.training_state_dict())
                with self.assertRaises(ValueError): m.update(s,0)
                self.assertEqual(before,state_digest(m.training_state_dict()))
