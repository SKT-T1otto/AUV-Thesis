"""Phase1 automatic gate: exact finite enumeration and synthetic data ONLY.

No real environment construction, snapshot restore, Trainer.run or checkpoint load.
"""
import copy
from fractions import Fraction as F
import io
import itertools
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

from chapter3_bser.models.hgr.phase1 import (
    behavior_identity, identical_behavior, runtime_contract, NoUpdateProof,
    PairNoise, named_seed, isolated_global_rng, audit_local_generators,
    phase1_options, PREDICTOR_REVISION,
)
from chapter3_bser.models.hgr.stable_predictor import StablePredictor
from chapter3_bser.models.hgr.policy import HandoffPolicy, TanhGaussianPolicy, weights_hash
from chapter3_bser.models.hgr.estimator import prefix_losses, gradient_vector, update_prefix
from chapter3_bser.experiments.hgr.train import Trainer, COST_FIELDS, validate_config
from chapter3_bser.experiments.hgr.provenance import fresh_source_identity, require_source_match

ROOT = Path(__file__).resolve().parents[1]
TOL = 2e-12


def config():
    return json.loads((ROOT / "configs/chapter3/hgr_phase1_zero.json").read_text(encoding="utf-8"))


def predictor(mode="ridge", shrink=.5):
    options = config()["phase1"]
    options.update(predictor_mode=mode, **{"lambda": shrink})
    return StablePredictor(153, options)


class LinearScore(nn.Module):
    """Fixed conditional scores, not an environment or a policy training task."""
    def __init__(self):
        super().__init__()
        self.theta_minus = nn.Linear(2, 1, bias=False).double()
        self.phi = nn.Linear(1, 1, bias=False).double()

    def score_log_prob(self, record):
        return (self.theta_minus.weight.reshape(2) * torch.tensor(list(map(float,record["v"])), dtype=torch.float64)).sum()


def trajectory(policy, contract, v=(1., 2.), tau=None):
    records = [dict(t=0, v=v, suffix=False, dones=[False]*4),
               dict(t=1, v=(0., 0.), suffix=tau is not None, dones=[True]*4)]
    return dict(records=records, rewards=[1., 2.], tau=tau, features=np.zeros(153),
                consumed=False, trajectory_complete=True, dataset_id="synthetic",
                theta_hash=weights_hash(policy.theta_minus),
                theta_behavior_sha256=behavior_identity(policy.theta_minus, contract).sha256,
                suffix_behavior_sha256=behavior_identity(policy.phi, contract).sha256)


def fake_trainer(policy):
    t = Trainer.__new__(Trainer)
    t.config = config(); t.phase1 = phase1_options(t.config)
    t.behavior_contract = runtime_contract(t.config)
    t.policy = policy; t.predictor = predictor("zero", 0)
    t.cycle = 0; t.stream_counter = 0; t.named_counts = {}
    t.costs = {k: 0 for k in (*COST_FIELDS, "predictor_updates")}
    return t


def synthetic_path(policy, noise, *, stop=5, active=(False, False, False, True)):
    """Known fully specified dynamics; all innovations supplied, no runtime."""
    table = noise.table(2, 9)
    env = np.random.default_rng(noise.environment_seed)
    state = np.zeros((4, 28), dtype=np.float32)
    rows = []
    for absolute_step in range(2, stop):
        actions, latents = policy.actions(state, suffix=True, active=active,
                                         generator=torch.Generator(), standard_noise=table[absolute_step-2])
        state[:, :3] += actions.numpy() + env.normal(0, .01, (4, 3)).astype(np.float32)
        reward = float(state[3, :3].sum())
        rows.append((absolute_step, state.copy(), actions.numpy(), reward, absolute_step == stop-1))
    return rows


def spawn_synthetic(queue):
    torch.set_num_threads(1)
    torch.manual_seed(99)
    actor = HandoffPolicy(dict(actor=dict(hidden_dim=8, expert_hidden_dim=8)))
    noise = PairNoise.make(11, 2, "synthetic", 7, "new", "policy_crn")
    rows = synthetic_path(actor, noise)
    queue.put([(t, s.tolist(), a.tolist(), r, done) for t, s, a, r, done in rows])


class Phase1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        # Any accidental use of a real simulator makes this automatic suite fail.
        cls.guards = [patch("chapter3_bser.experiments.hgr.runtime.MissionRuntime.__init__",
                            side_effect=AssertionError("real runtime forbidden in automatic gate")),
                      patch("chapter3_bser.experiments.hgr.runtime.MissionRuntime.restore",
                            side_effect=AssertionError("real restore forbidden in automatic gate"))]
        for guard in cls.guards:
            guard.start()

    @classmethod
    def tearDownClass(cls):
        for guard in reversed(cls.guards):
            guard.stop()

    def test_T1_exact_fixed_K_mean_full_covariance_and_float_implementation(self):
        c = [(F(1), F(2)), (F(-2), F(1)), (F(0), F(0))]
        distributions = [[(F(-1), F(1,2)), (F(3), F(1,2))],
                         [(F(0), F(1,2)), (F(2), F(1,2))], [(F(0), F(1))]]
        mu = [sum(y*p for y,p in d) for d in distributions]
        var = [sum(p*(y-m)**2 for y,p in d) for d,m in zip(distributions, mu)]
        policy = LinearScore()
        batch = [dict(records=[dict(v=v)], rewards=[0.], tau=1 if i<2 else None) for i,v in enumerate(c)]
        params = list(policy.theta_minus.parameters())
        duplicate_different_labels_seen = False
        for q, k, shrink in itertools.product(
                [(F(1,3),)*3, (F(1,6), F(1,3), F(1,2))], [1, 2], [F(0), F(1,2), F(1)]):
            f = [shrink*x for x in (5, -3, 0)]
            residual = [m-p for m,p in zip(mu,f)]
            b = [sum(c[i][a]*residual[i] for i in range(3)) for a in range(2)]
            truth = [sum(c[i][a]*mu[i] for i in range(3))/3 for a in range(2)]
            expected_cov = [[(sum(c[i][a]*c[i][d]*(var[i]+residual[i]**2)/q[i]
                                  for i in range(3))-b[a]*b[d])/(9*k)
                             for d in range(2)] for a in range(2)]
            atoms = [(i,y,q[i]*p) for i in range(3) for y,p in distributions[i]]
            mean = [F(0),F(0)]; second = [[F(0),F(0)],[F(0),F(0)]]
            for draws in itertools.product(atoms, repeat=k):
                probability = math.prod(d[2] for d in draws)
                value = [sum(c[i][a]*f[i] for i in range(3))/3
                         +sum(c[i][a]*(y-f[i])/q[i] for i,y,_ in draws)/(3*k) for a in range(2)]
                for a in range(2):
                    mean[a] += probability*value[a]
                    for d in range(2):
                        second[a][d] += probability*value[a]*value[d]
                if k == 2 and draws[0][0] == draws[1][0] and draws[0][1] != draws[1][1]:
                    duplicate_different_labels_seen = True
                losses = prefix_losses(policy, batch, list(map(float,f)),
                                       [(i,float(q[i]),float(y)) for i,y,_ in draws], gamma=1.)
                actual = -gradient_vector(losses["prediction"]+losses["correction"], params).numpy()
                np.testing.assert_allclose(actual, list(map(float,value)), atol=TOL, rtol=0)
            self.assertEqual(mean, truth)
            self.assertEqual([[second[a][d]-mean[a]*mean[d] for d in range(2)] for a in range(2)], expected_cov)
        self.assertTrue(duplicate_different_labels_seen)

    def test_T2_multistep_stopping_time_MDP_and_theta_suffix_counterexample(self):
        class ProbabilityPolicy(LinearScore):
            def __init__(self):
                super().__init__()
                with torch.no_grad():
                    self.theta_minus.weight.copy_(torch.tensor([[.4,.6]],dtype=torch.float64))
            def score_log_prob(self, record):
                if record["a"] is None:
                    return self.theta_minus.weight.sum()*0
                p = self.theta_minus.weight.reshape(2)[record["t"]%2]
                return p.log() if record["a"] else (1-p).log()
        policy = ProbabilityPolicy(); params = list(policy.theta_minus.parameters())
        gamma = F(19,20); probs=(F(2,5), F(3,5))
        paths = [(a,b) for a,b in itertools.product((0,1),repeat=2) if a==0]
        paths += [(1,b,0) for b in (0,1)]
        paths += [(1,b,1,d) for b,d in itertools.product((0,1),repeat=2)]
        exact = [F(0),F(0)]; estimate=np.zeros(2); truth_loss=0.
        counter_objective=0.; counter_estimate=np.zeros(2)
        for actions in paths:
            no_handoff = actions == (0,0)
            tau = None if no_handoff else len(actions)
            probability = math.prod(probs[t%2] if a else 1-probs[t%2] for t,a in enumerate(actions))
            scores = [sum((F(1)/probs[j] if a else -F(1)/(1-probs[j]))
                          for t,a in enumerate(actions) if t%2==j) for j in range(2)]
            prefix = [F(1,10)*(2*a-1) for a in actions]
            old_tail = [F(-1,2)] if no_handoff else [F(1,5)]
            new_tail = old_tail if no_handoff else ([F(3,5),F(-1,10)] if tau<4 else [F(4,5)])
            old_rewards = prefix+old_tail
            old_value=sum(gamma**t*r for t,r in enumerate(old_tail))
            new_value=sum(gamma**t*r for t,r in enumerate(new_tail))
            new_return=sum(gamma**t*r for t,r in enumerate(prefix+new_tail))
            for j in range(2):
                exact[j]+=probability*scores[j]*new_return
            records=[dict(a=a,t=t) for t,a in enumerate(actions)]+[dict(a=None,t=len(actions))]
            batch=[dict(records=records,rewards=list(map(float,old_rewards)),tau=tau)]
            label=0 if no_handoff else float(new_value-old_value)
            losses=prefix_losses(policy,batch,[0. if no_handoff else 7.],[(0,1.,label)],gamma=float(gamma))
            grad=-gradient_vector(sum(losses.values()),params).numpy()
            estimate+=float(probability)*grad
            p_tensor=policy.theta_minus.weight.reshape(2)
            path_probability=torch.stack([p_tensor[t%2] if a else 1-p_tensor[t%2] for t,a in enumerate(actions)]).prod()
            truth_loss=truth_loss+path_probability*float(new_return)
            # Deliberately illegal post-handoff theta-dependent reward exposes missing derivative.
            extra=0 if no_handoff else float(gamma**tau)*p_tensor[0]
            counter_objective=counter_objective+path_probability*(float(new_return)+extra)
            counter_estimate+=float(probability)*(grad+np.array(list(map(float,scores)))*float(extra.detach() if torch.is_tensor(extra) else extra))
        np.testing.assert_allclose(estimate,list(map(float,exact)),atol=TOL,rtol=0)
        np.testing.assert_allclose(estimate,gradient_vector(truth_loss,params).numpy(),atol=TOL,rtol=0)
        counter_truth=gradient_vector(counter_objective,params).numpy()
        self.assertGreater(abs(counter_truth[0]-counter_estimate[0]),.1)
        self.assertAlmostEqual(counter_truth[1],counter_estimate[1],places=12)

    def test_T3_verified_bypass_calls_nothing_and_keeps_g0_denominator_consumption(self):
        policy=LinearScore(); trainer=fake_trainer(policy); old=copy.deepcopy(policy)
        # Residual predictor state cannot leak through the bypass.
        trainer.predictor.coefficient.fill_(9); trainer.predictor.fitted.fill_(True)
        trainer.collect=Mock(side_effect=AssertionError("pilot called"))
        trainer.paired_label=Mock(side_effect=AssertionError("branch called"))
        trainer.predictor.fit=Mock(side_effect=AssertionError("fit called"))
        trainer.predictor.forward=Mock(side_effect=AssertionError("inference called"))
        main=[trajectory(policy,trainer.behavior_contract,tau=1),
              trajectory(policy,trainer.behavior_contract,v=(-1.,0.),tau=None)]
        predictions,draws,_,fit,proof,audit=trainer.phase1_estimates(main,old,"hgr")
        self.assertEqual(audit["K_actual"],0); self.assertEqual(audit["skip_reason"],"identical_suffix_behavior")
        for callable_ in (trainer.collect,trainer.paired_label,trainer.predictor.fit,trainer.predictor.forward):
            callable_.assert_not_called()
        expected=-gradient_vector(sum(prefix_losses(policy,main,[0.,0.],[],gamma=.95,
                                                    method="stochastic_direct_mc").values()),list(policy.theta_minus.parameters()))
        before=policy.theta_minus.weight.detach().clone(); phi=weights_hash(policy.phi)
        info=update_prefix(policy,torch.optim.SGD(policy.theta_minus.parameters(),lr=.01),
                           main,predictions,draws,gamma=.95,method="hgr",bypass=proof)
        torch.testing.assert_close((policy.theta_minus.weight-before).reshape(-1),.01*expected,atol=1e-15,rtol=0)
        self.assertEqual(info["delta_g_norm"],0.); self.assertGreater(info["g0_norm"],0.)
        self.assertEqual(weights_hash(policy.phi),phi)
        with self.assertRaisesRegex(ValueError,"stale prefix"):
            update_prefix(policy,torch.optim.SGD(policy.theta_minus.parameters(),lr=.01),
                          main,predictions,draws,gamma=.95,method="hgr",bypass=proof)
        for method in ("hgr","direct_boundary_corrected"):
            with self.assertRaises(ValueError):
                prefix_losses(policy,main,predictions,[],gamma=.95,method=method)
        with self.assertRaises(ValueError):
            prefix_losses(policy,main,predictions,[],gamma=.95,method="direct_boundary_corrected",bypass=proof)

    def test_T3_zero_nonbypass_and_direct_boundary_keep_fixed_K(self):
        policy=LinearScore(); trainer=fake_trainer(policy); old=copy.deepcopy(policy)
        trainer.collect=Mock(side_effect=AssertionError("zero mode pilot"))
        trainer.paired_label=Mock(return_value=(2.,{}))
        main=[trajectory(policy,trainer.behavior_contract,tau=None)]
        main[0]["tau"]=None
        values=trainer.phase1_estimates(main,old,"direct_boundary_corrected")
        self.assertIsNone(values[4]); self.assertEqual(len(values[1]),trainer.config["correction_draws_per_cycle"])
        self.assertTrue(all(draw==(0,1.,0.) for draw in values[1]))
        trainer.collect.assert_not_called(); trainer.paired_label.assert_not_called()
        main[0]["tau"]=1
        trainer.phase1_estimates(main,old,"direct_boundary_corrected")
        self.assertEqual(trainer.paired_label.call_count,trainer.config["correction_draws_per_cycle"])
        self.assertEqual([c.kwargs["query_index"] for c in trainer.paired_label.call_args_list],list(range(8)))

    def test_T4_actual_behavior_identity_rejects_each_change(self):
        actor=TanhGaussianPolicy(dict(hidden_dim=8,expert_hidden_dim=8))
        actor.register_buffer("audit_buffer",torch.ones(2))
        baseline=behavior_identity(actor,{})
        mutations=[lambda a: next(a.mean_actor.parameters()).add_(.01),
                   lambda a: a.audit_buffer.add_(1), lambda a: a.log_std.add_(.1),
                   lambda a: setattr(a,"mean_epsilon",2e-6),
                   lambda a: setattr(a.mean_actor.router,"temperature",2.),
                   lambda a: a.eval()]
        for mutate in mutations:
            clone=copy.deepcopy(actor)
            with torch.no_grad(): mutate(clone)
            self.assertFalse(identical_behavior(baseline,behavior_identity(clone,{})))
            with self.assertRaises(ValueError): NoUpdateProof.create(actor,clone,{})
        # Probe on zero input can agree while unused-buffer state differs.
        clone=copy.deepcopy(actor); clone.audit_buffer.add_(1)
        torch.testing.assert_close(actor.deterministic_action(torch.zeros(28)),
                                   clone.deterministic_action(torch.zeros(28)),atol=0,rtol=0)
        self.assertFalse(identical_behavior(baseline,behavior_identity(clone,{})))
        optimizer=torch.optim.SGD(actor.parameters(),lr=.1)
        optimizer.param_groups[0]["lr"]=.02
        self.assertTrue(identical_behavior(baseline,behavior_identity(actor,{})))
        # An actual parameter change invisible to this one probe is still rejected.
        linear=nn.Linear(2,1,bias=False); modified=copy.deepcopy(linear)
        with torch.no_grad(): modified.weight[0,0].add_(1.)
        torch.testing.assert_close(linear(torch.tensor([0.,1.])),modified(torch.tensor([0.,1.])),atol=0,rtol=0)
        self.assertFalse(identical_behavior(behavior_identity(linear,{}),behavior_identity(modified,{})))

    def test_T5_coupled_synthetic_paths_swap_stopping_and_addressing(self):
        torch.manual_seed(99)
        actor=HandoffPolicy(dict(actor=dict(hidden_dim=8,expert_hidden_dim=8)))
        changed=copy.deepcopy(actor)
        with torch.no_grad(): changed.phi.log_std.add_(.2)
        old_noise=PairNoise.make(11,2,"synthetic",7,"old","policy_crn")
        new_noise=PairNoise.make(11,2,"synthetic",7,"new","policy_crn")
        torch.testing.assert_close(old_noise.table(2,9),new_noise.table(2,9),atol=0,rtol=0)
        self.assertNotEqual(old_noise.environment_seed,new_noise.environment_seed)
        self.assertEqual(old_noise.public()["coupling_scope"],"policy_only_environment_independent")
        def primitive(rows):
            return [(t,s.tolist(),a.tolist(),r,done) for t,s,a,r,done in rows]
        first=synthetic_path(actor,old_noise)
        self.assertEqual(primitive(first),primitive(synthetic_path(actor,old_noise)))
        # Swap policies AND their complete addressed source, so the same two terms exchange.
        value=lambda p,noise: sum(row[3] for row in synthetic_path(p,noise))
        delta=value(changed,new_noise)-value(actor,old_noise)
        swapped=value(actor,old_noise)-value(changed,new_noise)
        self.assertEqual(delta,-swapped)
        short=synthetic_path(actor,old_noise,stop=3)
        np.testing.assert_array_equal(first[0][1],short[0][1])
        table=old_noise.table(2,9)
        obs=np.zeros((4,28),dtype=np.float32)
        _,latents=actor.actions(obs,suffix=False,active=[False,True,False,True],
                              generator=torch.Generator(),standard_noise=table[3])
        _,full=actor.actions(obs,suffix=False,active=[True]*4,
                           generator=torch.Generator(),standard_noise=table[3])
        np.testing.assert_array_equal(latents[3],full[3])
        self.assertNotEqual(old_noise.pair_id,PairNoise.make(11,2,"synthetic",8,"old","policy_crn").pair_id)
        independent=PairNoise.make(11,2,"synthetic",7,"new","independent")
        self.assertNotEqual(old_noise.policy_seed,independent.policy_seed)

    def test_T6_named_streams_global_isolation_and_spawn(self):
        a=fake_trainer(LinearScore()); b=fake_trainer(LinearScore())
        self.assertEqual(a.stream("main/trajectory"),b.stream("main/trajectory"))
        for _ in range(20): a.stream("pilot/trajectory")
        for purpose in ("suffix_training/trajectory","main/trajectory","formal_selection"):
            self.assertEqual(a.stream(purpose),b.stream(purpose))
        a.cycle=b.cycle=1
        self.assertEqual(a.stream("main/trajectory"),b.stream("main/trajectory"))
        random.seed(9); np.random.seed(9); torch.manual_seed(9)
        saved=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
        with isolated_global_rng():
            random.seed(81); np.random.seed(81); torch.manual_seed(81)
            random.random(); np.random.rand(); torch.rand(7)
        self.assertEqual(random.getstate(),saved[0])
        self.assertEqual(np.random.get_state()[0],saved[1][0])
        np.testing.assert_array_equal(np.random.get_state()[1],saved[1][1])
        self.assertEqual(np.random.get_state()[2:],saved[1][2:])
        self.assertTrue(torch.equal(torch.get_rng_state(),saved[2]))
        local=Mock(); local.action_rng=torch.Generator(); local.bad=np.random.default_rng(1)
        with self.assertRaisesRegex(ValueError,"unreviewed"): audit_local_generators(local)
        context=mp.get_context("spawn"); queue=context.Queue(); process=context.Process(target=spawn_synthetic,args=(queue,))
        process.start()
        try:
            child=queue.get(timeout=45); process.join(timeout=5)
            self.assertEqual(process.exitcode,0)
        finally:
            if process.is_alive(): process.terminate(); process.join()
            queue.close()
        class Capture:
            def put(self,value): self.value=value
        target=Capture(); spawn_synthetic(target)
        self.assertEqual(child,target.value)

    def test_T7_predictor_small_samples_extremes_freeze_and_serialization(self):
        model=predictor()
        for x,y,reason in [([],[],"insufficient_samples"),([np.ones(153)],[3.],"insufficient_samples"),
                           ([np.ones(153)]*2,[0.,0.],"all_zero_labels"),
                           ([np.full(153,1e308),np.full(153,-1e308)],[1.,2.],"numerical_solve_failure")]:
            info=model.fit(x,y)
            self.assertEqual(info["fallback_reason"],reason)
            self.assertEqual(float(model(np.ones(153))),0.)
        for x in ([np.ones(153)]*2,[np.zeros(153),np.ones(153)],[np.full(153,1e100),np.full(153,2e100)]):
            info=model.fit(x,[1.,2.],dataset_ids=["a","b"])
            self.assertIsNone(info["fallback_reason"]); self.assertGreater(info["scale_min"],0.)
            self.assertTrue(math.isfinite(float(model(np.ones(153)))))
        model.fit([],[])
        self.assertFalse(bool(model.fitted)); self.assertEqual(float(model(np.ones(153))),0.)
        for x,y in [([np.full(153,np.nan)],[1.]),([np.zeros(153)],[np.inf]),([np.zeros(152)],[1.])]:
            with self.assertRaises(ValueError): model.fit(x,y)
        model.fit([np.zeros(153),np.ones(153)],[1.,2.])
        batch=[dict(tau=1,features=np.ones(153)),dict(tau=None)]
        frozen,reason=model.freeze_predictions(batch)
        identity=behavior_identity(model,{})
        self.assertIsNone(reason); self.assertIsInstance(frozen,tuple)
        # Draw/label observations do not call fit or change shrink/normalization.
        list(itertools.product((0,1),repeat=2))
        self.assertTrue(identical_behavior(identity,behavior_identity(model,{})))
        memory=io.BytesIO()
        torch.save(dict(revision=PREDICTOR_REVISION,state=model.state_dict()),memory)
        memory.seek(0); data=torch.load(memory,weights_only=True)
        restored=predictor(); restored.load_state_dict(data["state"],strict=True)
        self.assertEqual(data["revision"],PREDICTOR_REVISION)
        self.assertTrue(identical_behavior(identity,behavior_identity(restored,{})))
        policy=LinearScore(); c={}; main=[trajectory(policy,c,tau=1)]
        model.shrink=0.
        values,_=model.freeze_predictions(main)
        for f in (values,(0.,)):
            losses=prefix_losses(policy,main,f,[(0,1.,3.)],gamma=.95)
            grad=gradient_vector(sum(losses.values()),list(policy.theta_minus.parameters()))
            if f is values: expected=grad
            else: torch.testing.assert_close(grad,expected,rtol=0,atol=0)

    def test_T8_legacy_numeric_path_source_and_cost_accounting(self):
        legacy=json.loads((ROOT/"configs/chapter3/hgr_train.json").read_text(encoding="utf-8"))
        self.assertIsNone(phase1_options(validate_config(legacy)))
        self.assertIsNotNone(phase1_options(validate_config(config())))
        actor=TanhGaussianPolicy(dict(hidden_dim=8,expert_hidden_dim=8))
        obs=torch.linspace(-.3,.3,28)
        generator=torch.Generator().manual_seed(12)
        actual,latent=actor.sample_action(obs,generator=generator)
        mu,std=actor.distribution_parameters(obs)
        expected=mu+std*torch.randn(mu.shape,generator=torch.Generator().manual_seed(12),dtype=mu.dtype)
        torch.testing.assert_close(latent,expected,atol=0,rtol=0)
        torch.testing.assert_close(actual,expected.tanh(),atol=0,rtol=0)
        source=fresh_source_identity(); changed=copy.deepcopy(source)
        key=next(iter(changed["files"])); changed["files"][key]="0"*64
        with self.assertRaisesRegex(ValueError,"hash mismatch"): require_source_match(changed,source)
        # Branch accounting uses actual lengths returned by the continuation, not H.
        trainer=fake_trainer(LinearScore()); trainer.costs["snapshot_restore_count"]=0
        fake=dict(snapshot=object()); noise=PairNoise.make(1,1,"audit",0,"new","policy_crn")
        with patch("chapter3_bser.experiments.hgr.train.continue_branch",return_value={"steps":7}):
            trainer.branch(fake,trainer.policy,"correction_new_suffix",pair_noise=noise)
        self.assertEqual(trainer.costs["correction_new_suffix_steps"],7)
        self.assertEqual(trainer.costs["snapshot_restore_count"],1)
        self.assertEqual(sum(trainer.costs[k] for k in COST_FIELDS),7)


if __name__ == "__main__":
    unittest.main()
