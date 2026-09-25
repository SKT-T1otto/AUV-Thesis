"""Compare fixed synthetic numerical fixtures directly to the audited Git base.

Loads source TEXT only; does not construct any real runtime or read checkpoints.
"""
import copy
import io
import json
from pathlib import Path
import subprocess
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from chapter3_bser.models.hgr import policy as current_policy, estimator as current_estimator
from chapter3_bser.models.hgr.phase1 import REVISION, STREAM_REVISION, PREDICTOR_REVISION, behavior_identity
from chapter3_bser.models.hgr.stable_predictor import StablePredictor
from chapter3_bser.experiments.hgr import train
from tests.test_hgr_phase1 import config, ROOT

BASE = "da4a8a64cba26b2adf7d248fd8486e517422628c"


def base_module(relative, package):
    source = subprocess.run(
        ["git","-c","safe.directory="+ROOT.as_posix(),"show",BASE+":"+relative],
        cwd=ROOT,check=True,capture_output=True).stdout.decode("utf-8")
    module=types.ModuleType(package+".phase1_frozen_fixture")
    module.__package__=package
    module.__file__=str(ROOT/relative)
    exec(compile(source,str(ROOT/relative)+"@"+BASE,"exec"),module.__dict__)
    return module


class LegacyFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.old_policy=base_module("chapter3_bser/models/hgr/policy.py","chapter3_bser.models.hgr")
        cls.old_estimator=base_module("chapter3_bser/models/hgr/estimator.py","chapter3_bser.models.hgr")
        cls.old_train=base_module("chapter3_bser/experiments/hgr/train.py","chapter3_bser.experiments.hgr")

    def test_T8_old_policy_density_action_gradient_and_rng_fixture_exact(self):
        torch.manual_seed(61)
        old=self.old_policy.HandoffPolicy(dict(actor=dict(hidden_dim=8,expert_hidden_dim=8)))
        new=current_policy.HandoffPolicy(old.config); new.load_state_dict(old.state_dict(),strict=True)
        observations=np.linspace(-.5,.5,112,dtype=np.float32).reshape(4,28)
        for suffix,active in ((False,[True]*4),(False,[True,False,True,True]),(True,[False]*3+[True])):
            a,z=old.actions(observations,suffix=suffix,active=active,generator=torch.Generator().manual_seed(51))
            b,w=new.actions(observations,suffix=suffix,active=active,generator=torch.Generator().manual_seed(51))
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            for x,y in zip(z,w):
                if x is None: self.assertIsNone(y)
                else: np.testing.assert_array_equal(x,y)
            record=dict(observations=observations,latents=z,suffix=suffix)
            torch.testing.assert_close(old.score_log_prob(record),new.score_log_prob(record),rtol=0,atol=0)
        a,z=old.actions(observations,suffix=False,active=[True]*4,generator=torch.Generator().manual_seed(62))
        records=[dict(observations=observations,latents=z,suffix=False),
                 dict(observations=observations,latents=[None]*4,suffix=True)]
        batch=[dict(records=records,rewards=[.2,1.],tau=1),dict(records=records[:1],rewards=[-.5],tau=None)]
        for method in ("hgr","direct_boundary_corrected","stochastic_direct_mc"):
            draws=[] if method=="stochastic_direct_mc" else [(0,.5,.3),(0,.5,-.2),(1,.5,0.)]
            before=[]
            for implementation,policy in ((self.old_estimator,old),(current_estimator,new)):
                losses=implementation.prefix_losses(policy,batch,[.7,0.],draws,gamma=.95,method=method)
                before.append(implementation.gradient_vector(sum(losses.values()),list(policy.theta_minus.parameters())))
            torch.testing.assert_close(*before,rtol=0,atol=0)

    def test_T8_old_full_synthetic_cycle_and_stream_sequence_exact(self):
        cfg=json.loads((ROOT/"configs/chapter3/hgr_train.json").read_text(encoding="utf-8"))
        cfg.update(suffix_training_episodes_per_cycle=2,pilot_prefix_episodes_per_cycle=2,
                   correction_draws_per_cycle=3,ablation="predictor_zero")
        result=[]
        for implementation in (self.old_train,train):
            torch.manual_seed(7)
            t=implementation.Trainer.__new__(implementation.Trainer)
            t.config=copy.deepcopy(cfg); t.phase1=None; t.named_counts={}; t.behavior_contract=None
            t.policy=current_policy.HandoffPolicy(dict(actor=dict(hidden_dim=8,expert_hidden_dim=8)))
            t.predictor=current_estimator.BoundaryPredictor(153,8)
            t.prefix_optimizer=torch.optim.SGD(t.policy.theta_minus.parameters(),lr=.001)
            t.suffix_optimizer=torch.optim.SGD(t.policy.phi.parameters(),lr=.001)
            t.costs={k:0 for k in (*train.COST_FIELDS,"snapshot_restore_count","predictor_updates","prefix_actor_updates","suffix_actor_updates")}
            t.completed_main=t.cycle=t.stream_counter=0; t.cycles=[]; t.episodes=[]; t.branches=[]
            t.output=Path("synthetic_unused_output"); t.index_rng=np.random.default_rng(7)
            streams=[]
            def collect(purpose,policy,*,stop=False):
                stream,seed=t.stream(purpose); streams.append((stream,seed))
                observations=np.zeros((4,28),dtype=np.float32)
                _,latents=policy.actions(observations,suffix=False,active=[True]*4,generator=torch.Generator().manual_seed(seed))
                return dict(records=[dict(t=0,suffix=False,observations=observations,latents=latents)],
                            rewards=[1.],tau=None,dataset_id=stream)
            t.collect=collect
            with patch.object(implementation,"write_json"),patch.object(implementation,"continue_branch",side_effect=AssertionError("real branch")):
                row=t.run_cycle(2)
            row.pop("wall_seconds")
            result.append((row,streams,torch.get_rng_state().clone(),current_policy.weights_hash(t.policy)))
        self.assertEqual(result[0][:2],result[1][:2])
        self.assertTrue(torch.equal(result[0][2],result[1][2]))
        self.assertEqual(result[0][3],result[1][3])
        # Legacy predictor_zero still collected pilots.
        self.assertEqual(len(result[1][1]),6)

    def test_T8_revision_constructed_predictor_memory_roundtrip_and_mismatch_rejection(self):
        cfg=config(); cfg["phase1"].update(predictor_mode="ridge",**{"lambda":.5})
        model=StablePredictor(153,cfg["phase1"])
        model.fit([np.zeros(153),np.ones(153)],[1.,2.])
        data=dict(config=cfg,phase1_revision=REVISION,predictor_revision=PREDICTOR_REVISION,
                  random_source_revision=STREAM_REVISION,named_counts={"1/main":2},predictor=model.state_dict())
        stream=io.BytesIO(); torch.save(data,stream); stream.seek(0)
        payload=torch.load(stream,weights_only=True)
        loaded=train.phase1_predictor_from_state(payload)
        self.assertEqual(behavior_identity(model,{}).exact,behavior_identity(loaded,{}).exact)
        for key,value in (("predictor_revision","old"),("random_source_revision","old"),
                          ("named_counts",{"1/main":-1})):
            changed=copy.deepcopy(payload); changed[key]=value
            with self.assertRaises(ValueError): train.phase1_predictor_from_state(changed)
        changed=copy.deepcopy(payload); changed["predictor"].pop("scale")
        with self.assertRaises(RuntimeError): train.phase1_predictor_from_state(changed)


if __name__=="__main__":
    unittest.main()
