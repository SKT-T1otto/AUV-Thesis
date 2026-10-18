import copy
import unittest
import torch
from core.replay.ch3_buffer import CH3ReplayBuffer
from tests.d2_v2_test_utils import state_digest

def replay(capacity):
    return CH3ReplayBuffer(capacity,4,[28]*4,[3]*4,alpha=0.,beta_start=0.)

def push(r,k):
    r.push([torch.full((28,),float(k))]*4,[torch.zeros(3)]*4,[k]*4,
           [torch.full((28,),float(k+1))]*4,[False]*4,[False]*4)

def whole(r):
    fields=("obs_buffs","ac_buffs","rew_buffs","next_obs_buffs","done_buffs")
    return state_digest(dict(header=r.state_dict(),**{k:[v.clone() for v in getattr(r,k)] for k in fields}))

class ReplayRestore(unittest.TestCase):
    def test_same_capacity_ring_and_sampling_continue(self):
        r=replay(3)
        for i in range(5): push(r,i)
        saved=r.state_dict(); other=replay(3); other.load_state_dict(saved)
        self.assertEqual(state_digest(saved),state_digest(other.state_dict()))
        push(other,5)
        self.assertEqual(sorted(other.sample(3,norm_rews=False)[0][0][:,0].tolist()),[3.,4.,5.])
        other.update_priorities(torch.tensor([0,1]),torch.tensor([.5,.8]))
        self.assertTrue(bool((other.priorities>0).all()))

    def test_invalid_restore_is_atomic_including_unused_slots(self):
        source=replay(3)
        for i in range(5): push(source,i)
        target=replay(5)
        for i in range(5): push(target,100+i)
        before=whole(target)
        with self.assertRaises(ValueError): target.load_state_dict(source.state_dict())
        self.assertEqual(before,whole(target))
        target=replay(3)
        for i in range(3): push(target,100+i)
        saved=source.state_dict()
        cases=[]
        for key,value in (("next_idx",3),("filled_i",4),("frame",float("nan")),("next_idx",True)):
            bad=copy.deepcopy(saved); bad[key]=value; cases.append(bad)
        for field in ("obs_buffs","ac_buffs","rew_buffs","next_obs_buffs","done_buffs"):
            bad=copy.deepcopy(saved); bad[field][-1]=bad[field][-1][:-1]; cases.append(bad)
            bad=copy.deepcopy(saved); bad[field]=bad[field][:-1]; cases.append(bad)
        for field in ("success_buffs","priorities"):
            bad=copy.deepcopy(saved); bad[field]=bad[field][:-1]; cases.append(bad)
        bad=copy.deepcopy(saved); bad["filled_i"]=2; bad["next_idx"]=0; cases.append(bad)
        for bad in cases:
            before=whole(target)
            with self.assertRaises(ValueError): target.load_state_dict(bad)
            self.assertEqual(before,whole(target))
