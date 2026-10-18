import copy
import unittest
import torch
from chapter3_bser.experiments.d2_repair_v2.model import build_model
from core.algorithms.maddpg import MADDPG
from tests.d2_v2_test_utils import config,sample,mutate,state_digest


class SampleContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old=torch.get_num_threads(); cls.addClassCleanup(torch.set_num_threads,old); torch.set_num_threads(1)

    def test_broadcast_and_all_malformed_fields_are_rejected_before_write(self):
        for label in ("b2","b3"):
            m=build_model(config(label)); s=sample(7)
            cases=[mutate(s,2,[torch.ones(1) for _ in range(4)]),mutate(s,5,torch.ones(1)),
                   mutate(s,0,s[0][:3]),mutate(s,1,[torch.zeros(7,2)]*4),
                   mutate(s,3,[torch.zeros(6,28)]*4),mutate(s,4,[torch.ones(7)*.5]*4),
                   mutate(s,5,torch.ones(7)*-1),mutate(s,5,torch.zeros(7)),
                   mutate(s,6,torch.arange(7).float()),mutate(s,6,torch.arange(6)),
                   mutate(s,7,torch.ones(7)*2),mutate(s,2,[torch.full((7,),float("inf"))]*4),
                   sample(0)]
            for bad in cases:
                before=state_digest(m.training_state_dict())
                with self.subTest(label=label,shapes=[str(type(x)) for x in bad]):
                    with self.assertRaises((ValueError,FloatingPointError)): m.update(bad,0)
                    self.assertEqual(before,state_digest(m.training_state_dict()))

    def test_column_vectors_and_snapshot_are_independent(self):
        for label in ("b2","b3"):
            m=build_model(config(label)); s=sample()
            m.update(s,0)
            m.update_all_targets()
            saved=m.training_state_dict(); old=state_digest(saved)
            values=list(s); values[2]=[v[:,None] for v in s[2]]; values[4]=[v[:,None] for v in s[4]]; values[5]=s[5][:,None]
            from chapter3_bser.experiments.d2_repair_v2.replay import ExecutionSample
            m.update(ExecutionSample(values,s.enabled,s.next_enabled),0)
            self.assertEqual(old,state_digest(saved))
            clone=build_model(config(label)); clone.load_training_state_dict(saved)
            self.assertEqual(old,state_digest(clone.training_state_dict()))
            self.assertIsNot(saved["init_dict"],m.init_dict)

    def test_core_rollout_count_and_critic_only_flag(self):
        m=build_model(config())
        for count in (0,3,5):
            with self.assertRaises(ValueError): MADDPG.step(m,[torch.zeros(28)]*count)
        s=sample(); m.update(s,0); self.assertTrue(m.last_actor_updated)
        before=state_digest(m.agents[0].policy_optimizer.state_dict())
        m.update_critic_only(s,0)
        self.assertFalse(m.last_actor_updated)
        self.assertEqual(before,state_digest(m.agents[0].policy_optimizer.state_dict()))
