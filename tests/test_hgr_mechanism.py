"""Independent finite-task and density checks for the production estimators."""
import copy
import itertools
import math
import unittest

import numpy as np
import torch
from torch import nn

from chapter3_bser.models.hgr.policy import HandoffPolicy, TanhGaussianPolicy, weights_hash
from chapter3_bser.models.hgr.estimator import prefix_losses, gradient_vector, reward_to_go, update_prefix, update_suffix


class FinitePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta_minus = nn.Linear(2, 1, bias=False).double()
        with torch.no_grad():
            self.theta_minus.weight.copy_(torch.tensor([[-0.4, 0.35]]))

    def score_log_prob(self, record):
        logits = self.theta_minus.weight.reshape(2)
        if record["joint"] is None:
            return logits.sum() * 0
        return torch.distributions.Bernoulli(logits=logits).log_prob(torch.tensor(record["joint"], dtype=torch.float64)).sum()


def finite_records():
    result = []
    for joint, tau, old, new in (((0,0), None, 0., 0.), ((0,1), 2, .8, .6),
                                 ((1,0), 4, .2, .9), ((1,1), 3, .5, .85)):
        length = 5 if tau is None else tau + 1
        rewards = [0.] * length
        rewards[0] = .13 * joint[0] - .08 * joint[1]
        rewards[-1] = -.4 if tau is None else old
        new_rewards = list(rewards)
        if tau is not None:
            new_rewards[-1] = new
        records = [dict(joint=joint if t == 0 else None) for t in range(length)]
        result.append(dict(records=records, rewards=rewards, new_rewards=new_rewards,
                           tau=tau, old=old, new=new, joint=joint))
    return result


class HGRMechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_enumeration_new_gradient_and_finite_difference(self):
        for gamma in (.9, 1.):
            policy = FinitePolicy(); data = finite_records(); params = list(policy.theta_minus.parameters())
            p = torch.sigmoid(params[0]).detach().numpy().reshape(2)
            def objective(theta, new):
                prob = 1 / (1 + np.exp(-theta))
                return sum(np.prod([prob[i] if a else 1-prob[i] for i,a in enumerate(d["joint"])]) * reward_to_go(d["new_rewards"] if new else d["rewards"], gamma)[0] for d in data)
            theta = params[0].detach().numpy().reshape(2)
            true = np.array([(objective(theta+np.eye(2)[i]*1e-5, True)-objective(theta-np.eye(2)[i]*1e-5, True))/2e-5 for i in range(2)])
            estimated = np.zeros(2)
            for d in data:
                probability = np.prod([p[i] if a else 1-p[i] for i,a in enumerate(d["joint"])])
                label = d["new"]-d["old"]
                losses = prefix_losses(policy, [d], [7.3 if d["tau"] else 0.], [(0,1.,label)], gamma=gamma)
                estimated += probability * (-gradient_vector(sum(losses.values()), params)).numpy()
            np.testing.assert_allclose(estimated, true, atol=2e-10)

    def test_biased_predictor_correction_and_nonuniform_sampling(self):
        policy = FinitePolicy(); data = finite_records(); params = list(policy.theta_minus.parameters())
        predictions = [0., 8., -5., 3.]
        q = [.1, .2, .3, .4]
        correct = np.zeros(2); removed_correction = np.zeros(2); removed_prediction = np.zeros(2)
        for i, probability in enumerate(q):
            draws = [(i, probability, data[i]["new"]-data[i]["old"])]
            for mode, target in (("none", correct), ("remove_correction", removed_correction), ("remove_prediction", removed_prediction)):
                losses = prefix_losses(policy, data, predictions, draws, gamma=.9, ablation=mode)
                target += probability * (-gradient_vector(sum(losses.values()), params)).numpy()
        direct = copy.deepcopy(data)
        for d in direct:
            d["rewards"] = d["new_rewards"]
        truth = -gradient_vector(sum(prefix_losses(policy, direct, [0.]*4, [], gamma=.9, method="stochastic_direct_mc").values()), params).numpy()
        np.testing.assert_allclose(correct, truth, atol=1e-12)
        self.assertGreater(np.linalg.norm(removed_correction-truth), .01)
        self.assertGreater(np.linalg.norm(removed_prediction-truth), .01)
        # N includes the no-handoff failure; redrawing only eligible records is wrong.
        wrong = -gradient_vector(sum(prefix_losses(policy, direct[1:], [0.]*3, [], gamma=.9, method="stochastic_direct_mc").values()), params).numpy()
        self.assertGreater(np.linalg.norm(wrong-truth), .01)

    def test_old_plus_difference_cancels_and_direct_boundary_keeps_prefix_rewards(self):
        policy = FinitePolicy(); data = finite_records(); params = list(policy.theta_minus.parameters())
        labels = [d["new"]-d["old"] for d in data]
        hgr = prefix_losses(policy, data, [0., .5, -.7, .9], [(i,.25,v) for i,v in enumerate(labels)], gamma=.9)
        boundary = prefix_losses(policy, data, [0., -.3, .4, .8], [(i,.25,d["new"]) for i,d in enumerate(data)], gamma=.9, method="direct_boundary_corrected")
        direct = copy.deepcopy(data)
        for d in direct:
            d["rewards"] = d["new_rewards"]
        mc = prefix_losses(policy, direct, [0.]*4, [], gamma=.9, method="stochastic_direct_mc")
        expected = gradient_vector(sum(mc.values()), params)
        torch.testing.assert_close(gradient_vector(sum(hgr.values()), params), expected)
        torch.testing.assert_close(gradient_vector(sum(boundary.values()), params), expected)
        self.assertGreater(float(gradient_vector(boundary["old"], params).norm()), .01)
        self.assertGreater(float(gradient_vector(hgr["old"], params).norm()), .01)

    def test_repeated_draws_and_no_handoff_keep_sampling_multiplicity(self):
        policy = FinitePolicy(); data = finite_records(); params = list(policy.theta_minus.parameters())
        draws = [(1,.25,-.2),(1,.25,.3),(0,.25,0.)]
        losses = prefix_losses(policy, data, [0., .8, .2, .5], draws, gamma=.9)
        u = .9**2 * policy.score_log_prob(data[1]["records"][0])
        expected = -u*((-.2-.8)+(.3-.8))/3
        torch.testing.assert_close(gradient_vector(losses["correction"], params), gradient_vector(expected, params))
        before = params[0].detach().clone()
        info = update_prefix(policy, torch.optim.SGD(params, lr=.01), data, [0.,.8,.2,.5], draws, gamma=.9, method="hgr")
        self.assertGreater(info["correction_norm"], 0)
        self.assertFalse(torch.equal(before, params[0]))
        with self.assertRaisesRegex(ValueError,"stale prefix"):
            update_prefix(policy, torch.optim.SGD(params,lr=.01), data, [0.,.8,.2,.5], draws, gamma=.9, method="hgr")

    def test_equal_suffix_policies_have_zero_increment_mean_but_nonzero_trial_noise(self):
        # Independent Bernoulli continuation trials: same policy does not imply
        # paired samples are equal. Enumerate all outcomes instead of a flaky MC tolerance.
        p = .7
        outcomes = [(new-old, (p if new else 1-p)*(p if old else 1-p)) for old,new in itertools.product((0,1),repeat=2)]
        self.assertAlmostEqual(sum(delta*prob for delta,prob in outcomes), 0.)
        self.assertGreater(sum(delta**2*prob for delta,prob in outcomes), 0.)

    def test_tanh_density_jacobian_fixed_latent_score_and_saturation(self):
        torch.manual_seed(7)
        actor = TanhGaussianPolicy(dict(hidden_dim=8, expert_hidden_dim=8)).double()
        obs = torch.linspace(-.7,.7,28,dtype=torch.float64).reshape(1,28)
        mu, std = actor.distribution_parameters(obs)
        latent = (mu + .3*std).requires_grad_(True)
        latent.retain_grad()
        score = actor.log_prob(obs, latent)
        expected = (torch.distributions.Normal(mu,std).log_prob(latent.detach())-torch.log1p(-latent.detach().tanh().square())).sum(-1)
        torch.testing.assert_close(score, expected)
        score.backward()
        self.assertIsNone(latent.grad)
        # Compare score derivative with an independently differenced fixed-sample density.
        original = actor.log_std.detach().clone(); eps=1e-5
        analytical = actor.log_std.grad.detach().clone()
        numerical = []
        for i in range(3):
            with torch.no_grad():
                actor.log_std.copy_(original); actor.log_std[i] += eps
            plus = float(actor.log_prob(obs, latent).detach())
            with torch.no_grad():
                actor.log_std.copy_(original); actor.log_std[i] -= eps
            minus = float(actor.log_prob(obs, latent).detach()); numerical.append((plus-minus)/(2*eps))
        torch.testing.assert_close(analytical, torch.tensor(numerical,dtype=torch.float64))
        self.assertTrue(torch.isfinite(actor.log_prob(obs, torch.tensor([[100.,-100.,30.]],dtype=torch.float64))).all())
        action,z=actor.sample_action(obs); self.assertTrue(bool((action.abs()<=1).all())); self.assertEqual(z.shape,(1,3))

    def test_prefix_standby_suffix_parameter_and_buffer_isolation(self):
        policy = HandoffPolicy(dict(actor=dict(hidden_dim=8,expert_hidden_dim=8)))
        observations = np.zeros((4,28),dtype=np.float32)
        first,_=policy.actions(observations,suffix=False,active=[True]*4,generator=torch.Generator().manual_seed(9))
        prefix_hash = weights_hash(policy.theta_minus)
        records=[]
        actions,latents=policy.actions(observations,suffix=True,active=[False]*3+[True],generator=torch.Generator().manual_seed(11))
        records.append(dict(observations=observations,latents=latents,suffix=True))
        update_suffix(policy,torch.optim.SGD(policy.phi.parameters(),lr=.01),[dict(records=records,rewards=[1.])],.95)
        self.assertEqual(prefix_hash,weights_hash(policy.theta_minus))
        second,_=policy.actions(observations,suffix=False,active=[True]*4,generator=torch.Generator().manual_seed(9))
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        self.assertEqual(latents[:3],[None]*3)
        self.assertTrue(torch.equal(actions[:3],torch.zeros(3,3)))
        policy.assert_isolated()

    def test_explicit_prrac_mean_mapping_is_finite_complete_and_independent(self):
        from chapter3_bser.models.prrac.phase_routed_actor import PhaseRoutedResidualActor
        actors=[PhaseRoutedResidualActor(hidden_dim=8,expert_hidden_dim=8) for _ in range(4)]
        policy=HandoffPolicy(dict(actor=dict(hidden_dim=8,expert_hidden_dim=8)))
        mapping=policy.import_prrac_means(actors)
        observations=torch.randn(4,28)
        for i,actor in enumerate(actors):
            expected=actor(observations).gated_residual_action
            actual=policy.theta_minus[i].deterministic_action(observations)
            torch.testing.assert_close(actual,expected,atol=mapping['epsilon'],rtol=0)
        torch.testing.assert_close(policy.phi.deterministic_action(observations),actors[3](observations).gated_residual_action,atol=mapping['epsilon'],rtol=0)
        policy.assert_isolated()
        hashes=[weights_hash(block) for block in policy.theta_minus]
        incompatible=list(actors); incompatible[3]=PhaseRoutedResidualActor(hidden_dim=9,expert_hidden_dim=8)
        with self.assertRaisesRegex(ValueError,'mapping keys/shapes'):
            policy.import_prrac_means(incompatible)
        self.assertEqual(hashes,[weights_hash(block) for block in policy.theta_minus])


if __name__ == "__main__":
    unittest.main()
