"""Exercise production branch and cycle orchestration against synthetic doubles."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from chapter3_bser.models.hgr.policy import HandoffPolicy, weights_hash
from chapter3_bser.models.hgr.phase1 import (
    PairNoise, behavior_identity, runtime_contract, NoUpdateProof, phase1_options,
    isolated_global_rng,
)
from chapter3_bser.models.hgr.stable_predictor import StablePredictor
from chapter3_bser.experiments.hgr.runtime import continue_branch, seed_innovations
from chapter3_bser.experiments.hgr import train, phase1_acceptance as acceptance
from tests.test_hgr_phase1 import config, fake_trainer, LinearScore, trajectory


class SyntheticRuntime:
    """Finite deterministic state transition with explicit independent innovations."""
    def __init__(self, seed):
        self.config = config()
        self.step = 5
        self.stop = 8
        self.action_rng = torch.Generator().manual_seed(seed)
        self.observations = np.zeros((4,28),dtype=np.float32)
        self.env = SimpleNamespace(get_episode_result=lambda:dict(termination_reason="synthetic_terminal"))
        seed_innovations(seed,cpu_only=True)
        self.closed = False

    @property
    def terminal(self):
        return self.step >= self.stop

    def features(self):
        return np.zeros(153,dtype=np.float32)

    def advance(self, policy, *, standard_noise):
        t=self.step
        actions,latents=policy.actions(self.observations,suffix=True,active=[False]*3+[True],
                                      generator=self.action_rng,standard_noise=standard_noise)
        innovation=random.random()+float(np.random.rand())+float(torch.rand(()))
        self.observations[:, :3] += actions.numpy()+innovation*.01
        self.step += 1
        if float(policy.phi.log_std[0].detach()) > -2.:
            self.stop = 7
        return dict(t=t, observations=np.zeros((4,28)),next_observations=self.observations.copy(),
                    actions=actions.numpy(), latents=latents, suffix=True,
                    team_reward=float(actions[3].sum())+innovation, dones=[self.terminal]*4,
                    public_positions=self.observations[:,:3].copy())

    def close(self):
        self.closed=True


class SyntheticRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_T5_real_branch_function_with_synthetic_restore_relative_discount_and_early_stop(self):
        torch.manual_seed(14)
        old=HandoffPolicy(dict(actor=dict(hidden_dim=8,expert_hidden_dim=8)))
        new=copy.deepcopy(old)
        with torch.no_grad(): new.phi.log_std.fill_(-1.9)
        snapshot=SimpleNamespace(step=5,max_steps=400,sha256="synthetic_snapshot_not_real",
                                 policy_identity={"phase1_boundary_features_sha256":
                                     hashlib.sha256(np.zeros(153,dtype=np.float32).tobytes()).hexdigest()})
        runtimes=[]
        def restore(snapshot,*,innovation_seed):
            value=SyntheticRuntime(innovation_seed); runtimes.append(value); return value
        noise=PairNoise.make(15,1,"test",0,"old","policy_crn")
        new_noise=PairNoise.make(15,1,"test",0,"new","policy_crn")
        saved=torch.get_rng_state().clone()
        with patch("chapter3_bser.experiments.hgr.runtime.MissionRuntime.restore",side_effect=restore):
            a=continue_branch(snapshot,old,pair_noise=noise,keep_records=True)
            b=continue_branch(snapshot,new,pair_noise=new_noise,keep_records=True)
            repeated=continue_branch(snapshot,old,pair_noise=noise,keep_records=True)
        self.assertTrue(torch.equal(saved,torch.get_rng_state()))
        self.assertEqual(a["steps"],3); self.assertEqual(b["steps"],2)
        self.assertEqual(a["terminal_step"],8); self.assertEqual(b["terminal_step"],7)
        self.assertEqual(a["trajectory_trace_sha256"],repeated["trajectory_trace_sha256"])
        self.assertEqual(a["G_plus"],sum(.95**i*r["team_reward"] for i,r in enumerate(a["records"])))
        self.assertTrue(all(r.closed for r in runtimes))
        self.assertEqual(a["pair_id"],b["pair_id"])
        self.assertNotEqual(a["trace_id"],b["trace_id"])

    def test_T3_invalid_or_changed_bypass_proofs(self):
        policy=LinearScore(); t=fake_trainer(policy); old=copy.deepcopy(policy)
        data=[trajectory(policy,t.behavior_contract,tau=1)]
        proof=NoUpdateProof.create(old.phi,policy.phi,t.behavior_contract)
        for field,value in [("trajectory_complete",False),("theta_behavior_sha256","changed"),
                            ("suffix_behavior_sha256","changed"),("consumed",True)]:
            changed=copy.deepcopy(data); changed[0][field]=value
            with self.assertRaises(ValueError): proof.validate(policy,changed,[0.],[],"hgr","none")
        with torch.no_grad(): policy.phi.weight.add_(.01)
        with self.assertRaisesRegex(ValueError,"identity changed"):
            proof.validate(policy,data,[0.],[],"hgr","none")

    def test_T7_pilots_independent_and_predictor_frozen_before_formal_labels(self):
        t=fake_trainer(LinearScore()); old=copy.deepcopy(t.policy)
        with torch.no_grad(): t.policy.phi.weight.add_(.1)
        t.phase1["predictor_mode"]="ridge"; t.phase1["lambda"]=.5
        t.config["pilot_prefix_episodes_per_cycle"]=2
        t.predictor=StablePredictor(153,t.phase1)
        main=[trajectory(t.policy,t.behavior_contract,tau=1)]
        pilots=[trajectory(old,t.behavior_contract,tau=1) for _ in range(2)]
        for i,p in enumerate(pilots):
            p["dataset_id"]=f"pilot_{i}"; p["features"]=np.ones(153)*i
        t.collect=Mock(side_effect=pilots)
        snapshots=[]
        def label(trajectory,old,new,purpose,*,query_index):
            if purpose=="pilot": return (1.+query_index,{})
            snapshots.append(behavior_identity(t.predictor,t.behavior_contract))
            return (1000.+query_index,{})
        t.paired_label=Mock(side_effect=label)
        values,draws,ids,fit,proof,audit=t.phase1_estimates(main,old,"hgr")
        self.assertIsNone(proof); self.assertEqual(fit["samples"],2)
        self.assertEqual(fit["dataset_ids"],["pilot_0","pilot_1"])
        self.assertEqual(fit["mse_scope"],"pilot_training_only"); self.assertIsNone(fit["independent_error"])
        self.assertTrue(all(s.exact==snapshots[0].exact for s in snapshots))
        self.assertEqual(fit["lambda_value"],.5)
        self.assertEqual(len(draws),8); self.assertIsInstance(values,tuple)
        self.assertEqual([c.args[0] for c in t.collect.call_args_list],["pilot","pilot"])

    def test_T8_one_synthetic_cycle_wires_bypass_costs_and_single_updates(self):
        t=train.Trainer.__new__(train.Trainer)
        t.config=config(); t.config.update(main_prefix_batch_size=2,suffix_training_episodes_per_cycle=2)
        t.phase1=phase1_options(t.config); t.behavior_contract=runtime_contract(t.config)
        torch.manual_seed(12)
        t.policy=HandoffPolicy(dict(actor=dict(hidden_dim=8,expert_hidden_dim=8)))
        t.predictor=StablePredictor(153,t.phase1)
        t.prefix_optimizer=torch.optim.SGD(t.policy.theta_minus.parameters(),lr=.001)
        t.suffix_optimizer=torch.optim.SGD(t.policy.phi.parameters(),lr=.001)
        t.costs={k:0 for k in (*train.COST_FIELDS,"snapshot_restore_count","predictor_updates",
                              "prefix_actor_updates","suffix_actor_updates")}
        t.completed_main=t.cycle=0; t.cycles=[]; t.episodes=[]; t.branches=[]
        t.output=Path("synthetic_unused_output")
        calls=[]
        def collect(purpose,policy,*,stop=False):
            calls.append(purpose)
            obs=np.zeros((4,28),dtype=np.float32)
            _,latent=policy.actions(obs,suffix=False,active=[True]*4,generator=torch.Generator().manual_seed(6))
            records=[dict(t=0,suffix=False,observations=obs,latents=latent,dones=[True]*4)]
            return dict(records=records,rewards=[1.],tau=None,features=None,dataset_id=f"{purpose}_{len(calls)}",
                        consumed=False,trajectory_complete=True,theta_hash=weights_hash(policy.theta_minus),
                        theta_behavior_sha256=behavior_identity(policy.theta_minus,t.behavior_contract).sha256,
                        suffix_behavior_sha256=behavior_identity(policy.phi,t.behavior_contract).sha256)
        t.collect=collect
        with patch.object(train,"write_json") as write, patch.object(train,"continue_branch",side_effect=AssertionError("branch")):
            row=t.run_cycle(2)
        self.assertEqual(calls,["suffix_training"]*2+["main"]*2)
        self.assertEqual(row["actual_total_environment_steps"],4)
        self.assertEqual(row["costs"]["prefix_actor_updates"],1)
        self.assertEqual(row["costs"]["suffix_actor_updates"],1)
        self.assertEqual(row["phase1"]["K_actual"],0); self.assertEqual(row["delta_g_norm"],0.)
        self.assertEqual(row["completed_main_trajectories"],2)
        self.assertEqual(write.call_count,3)

    def test_manual_runner_not_exercised_is_bounded_and_does_not_retry(self):
        scenarios=[dict(scenario_id=f"predeclared_{i}") for i in range(2)]
        returned=dict(records=[None]*3,tau=None,snapshot=None,summary={})
        manifest={"M20_MOVING_UNKNOWN_MULTI":{"scenarios":scenarios}}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(acceptance,"build_scenario_manifests",return_value=manifest), \
                patch.object(acceptance,"collect_trajectory",return_value=returned) as collect, \
                patch.object(acceptance,"continue_branch",side_effect=AssertionError("no handoff branch")):
            output=Path(directory)/"collision_terminal"/"new"
            status=acceptance.run(config(),output)
            report=json.loads((output/"acceptance.json").read_text(encoding="utf-8"))
            self.assertEqual(status,2); self.assertEqual(collect.call_count,2)
            self.assertEqual(report["status"],"NOT_EXERCISED")
            self.assertEqual(report["recorded_environment_steps"],6)
            self.assertEqual(report["plan"]["maximum_environment_steps"],5600)
            with self.assertRaises(FileExistsError): acceptance.run(config(),output)

    def test_T7_invalid_formal_label_fails_and_leaves_failure_diagnostic(self):
        t=fake_trainer(LinearScore()); t.branches=[]
        fake=dict(snapshot=SimpleNamespace(sha256="synthetic"),tau=1,
                  dataset_id="synthetic",features=np.zeros(153))
        t.branch=Mock(return_value=dict(G_plus=float("nan")))
        with self.assertRaisesRegex(ValueError,"nonfinite raw paired label"):
            t.paired_label(fake,t.policy,t.policy,"correction",query_index=0)
        self.assertEqual(t.branches,[])
        with tempfile.TemporaryDirectory() as directory:
            t.output=Path(directory)
            t._run_cycle=Mock(side_effect=ValueError("invalid raw predictor features"))
            with self.assertRaisesRegex(ValueError,"invalid raw"):
                t.run_cycle(2)
            diagnostic=json.loads((t.output/"phase1_failure.json").read_text(encoding="utf-8"))
            self.assertEqual(diagnostic["status"],"FAIL")
            self.assertIn("invalid raw predictor features",diagnostic["error"])


if __name__=="__main__":
    unittest.main()
