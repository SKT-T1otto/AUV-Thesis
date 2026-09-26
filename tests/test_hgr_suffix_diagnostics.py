"""Small differentiable fixtures; no simulator, Trainer, or checkpoint loading."""
import copy
import json
import pickle
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from chapter3_bser.models.hgr.estimator import update_suffix
from chapter3_bser.models.hgr.phase1 import behavior_identity
from scripts.hgr_suffix_diagnostics import diagnose_suffix_update


class TinyGaussian(nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(3, dtype=dtype))
        self.mean = nn.Linear(1, 3, bias=False, dtype=dtype)
        with torch.no_grad():
            self.mean.weight.zero_()

    def distribution_parameters(self, observation):
        observation = torch.as_tensor(observation, dtype=self.log_std.dtype).reshape(1, 1)
        mu = self.mean(observation)
        return mu, self.log_std.exp().expand_as(mu)

    def log_prob(self, observation, latent):
        mu, std = self.distribution_parameters(observation)
        z = torch.as_tensor(latent, dtype=mu.dtype).detach()
        return (-.5 * ((z - mu) / std).square() - std.log()).sum()


class TinyPolicy(nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.theta_minus = nn.Linear(1, 1, bias=False, dtype=dtype)
        self.phi = TinyGaussian(dtype)
        with torch.no_grad():
            self.theta_minus.weight.fill_(.25)

    def score_log_prob(self, record, *, suffix=False):
        assert suffix
        zero = self.phi.log_std.sum() * 0
        if not record["suffix"] or record["latents"][3] is None:
            return zero
        return zero + self.phi.log_prob(record["observations"][3], record["latents"][3])


def trajectory(*, tau=0, noise=1., reward=1.):
    length = 2 if tau is None else tau + 1
    records = []
    for t in range(length):
        suffix = tau is not None and t >= tau
        records.append(dict(t=t, suffix=suffix, observations=[np.array([1.]) for _ in range(4)],
                            latents=[None, None, None, np.full(3, noise) if suffix else None]))
    return dict(records=records, rewards=[0.] * (length - 1) + [reward], tau=tau,
                consumed=False, trajectory_complete=True)


def run(policy, trajectories, gamma=.95, lr=.001):
    return diagnose_suffix_update(policy, trajectories, gamma=gamma, lr=lr, contract={"gamma": gamma})


def test_exactly_one_original_update_and_exact_same_phi():
    policy = TinyPolicy()
    data = [trajectory(tau=0, noise=.5), trajectory(tau=3, noise=1.5, reward=-.7)]
    expected = copy.deepcopy(policy)
    loss = update_suffix(expected, torch.optim.SGD(expected.phi.parameters(), lr=.013), data, .95)
    with patch("scripts.hgr_suffix_diagnostics.update_suffix", wraps=update_suffix) as original:
        actual, report, vectors = run(policy, data, lr=.013)
    assert original.call_count == 1
    assert report["suffix_loss"] == loss
    for name, value in expected.state_dict().items():
        assert torch.equal(value, actual.state_dict()[name])
    assert report["sgd_update"]["native_step_bitwise_match"]
    expected_g = -np.concatenate([p.grad.detach().numpy().reshape(-1) for p in expected.phi.parameters()])
    np.testing.assert_array_equal(vectors["g_phi"], expected_g)
    json.dumps(report, allow_nan=False)


def test_no_handoff_is_legal_zero_update():
    policy = TinyPolicy()
    new, report, vectors = run(policy, [trajectory(tau=None), trajectory(tau=None)])
    assert report["status"] == "NOT_EXERCISED"
    assert report["handoff_count"] == report["effective_suffix_score_steps"] == 0
    assert report["sgd_update"]["changed_parameter_elements"] == 0
    assert not report["behavior"]["phi_identity_changed"]
    assert report["gradient"]["cancellation_ratio"] is None
    assert report["behavior"]["kl_mean"] is None
    assert all(np.count_nonzero(array) == 0 for array in vectors.values())
    for name, value in policy.state_dict().items():
        assert torch.equal(value, new.state_dict()[name])


def test_discount_decomposition_preserves_full_batch_denominator():
    policy = TinyPolicy()
    _, report, vectors = run(policy, [trajectory(tau=2), trajectory(tau=None)], gamma=.5)
    np.testing.assert_allclose(vectors["trajectory_discounted_h"][0], .25 * vectors["trajectory_h"][0])
    np.testing.assert_allclose(vectors["g_phi"], .125 * vectors["trajectory_h"][0])
    assert report["trajectory_rows"][0]["gamma_tau"] == .25
    assert report["gradient"]["cancellation_ratio"] == 1.
    assert not report["diagnostic_discount_removal_used_for_update"]


def test_opposed_trajectory_scores_cancel_without_false_gradient_block_claim():
    _, report, vectors = run(TinyPolicy(), [trajectory(noise=1.), trajectory(noise=-1.)])
    assert np.linalg.norm(vectors["trajectory_h"][0]) > 0
    np.testing.assert_array_equal(vectors["trajectory_h"][0], -vectors["trajectory_h"][1])
    assert report["gradient"]["g_phi_norm"] == 0
    assert report["gradient"]["cancellation_ratio"] == 0
    assert report["status"] == "EXERCISED"


def test_theta_input_policy_trajectories_and_rng_unchanged():
    policy = TinyPolicy()
    policy.requires_grad_(False)
    data = [trajectory(tau=2, noise=.7)]
    before_data = pickle.dumps(data)
    before = behavior_identity(policy, {"gamma": .95})
    theta = behavior_identity(policy.theta_minus, {"gamma": .95})
    rng = torch.get_rng_state().clone()
    updated, report, _ = run(policy, data)
    assert behavior_identity(policy, {"gamma": .95}).exact == before.exact
    assert behavior_identity(updated.theta_minus, {"gamma": .95}).exact == theta.exact
    assert all(not p.requires_grad for p in policy.parameters())
    assert all(p.requires_grad for p in updated.phi.parameters())
    assert pickle.dumps(data) == before_data
    assert torch.equal(rng, torch.get_rng_state())
    assert report["behavior"]["phi_identity_changed"]
    assert report["behavior"]["kl_mean"] > 0
    assert report["behavior"]["max_abs_shared_noise_action_change"] > 0


def test_report_uses_recorded_observations_and_zero_kl_for_equal_forward_behavior():
    policy = TinyPolicy()
    _, report, _ = run(policy, [trajectory(noise=1.), trajectory(noise=-1.)])
    assert report["behavior"]["observation_count"] == 2
    assert report["behavior"]["observation_source"] == "recorded_pre_update_suffix_trajectories"
    assert report["behavior"]["kl_mean"] == 0
    assert report["behavior"]["max_abs_shared_noise_action_change"] == 0


def test_float32_rounding_is_observed_without_changing_sgd():
    policy = TinyPolicy(dtype=torch.float32)
    with torch.no_grad():
        policy.phi.mean.weight.fill_(1.)
    # z=mu+std: nonzero score, but the proposed mean step is below float32 ULP.
    _, report, vectors = run(policy, [trajectory(noise=2.)], lr=1e-12)
    assert report["gradient"]["g_phi_norm"] > 0
    assert report["sgd_update"]["delta_phi_norm"] == 0
    assert report["sgd_update"]["rounding_residual_norm"] > 0
    assert report["sgd_update"]["native_step_bitwise_match"]
    np.testing.assert_array_equal(vectors["rounding_residual"], -vectors["ideal_delta_phi"])


@pytest.mark.parametrize("field", ["reward", "observation", "latent", "policy"])
def test_nonfinite_inputs_rejected_before_update(field):
    policy, data = TinyPolicy(), [trajectory()]
    if field == "reward":
        data[0]["rewards"][0] = float("nan")
    elif field == "observation":
        data[0]["records"][0]["observations"][3][0] = float("inf")
    elif field == "latent":
        data[0]["records"][0]["latents"][3][0] = float("nan")
    else:
        with torch.no_grad():
            policy.phi.log_std[0] = float("nan")
    with patch("scripts.hgr_suffix_diagnostics.update_suffix", side_effect=AssertionError("must reject first")):
        with pytest.raises(ValueError, match="nonfinite"):
            run(policy, data)


def test_contract_discount_mismatch_rejected():
    with pytest.raises(ValueError, match="contract"):
        diagnose_suffix_update(TinyPolicy(), [trajectory()], gamma=.95, lr=.001, contract={"gamma": .9})


def test_inconsistent_handoff_mask_rejected():
    data = [trajectory(tau=1)]
    data[0]["tau"] = None
    with pytest.raises(ValueError, match="stopping-time"):
        run(TinyPolicy(), data)


@pytest.mark.parametrize("key", ["theta_behavior_sha256", "suffix_behavior_sha256"])
def test_stale_trajectory_behavior_identity_rejected(key):
    data = [trajectory()]
    data[0][key] = "different-policy"
    with patch("scripts.hgr_suffix_diagnostics.update_suffix", side_effect=AssertionError("must reject first")):
        with pytest.raises(ValueError, match="stale suffix trajectory"):
            run(TinyPolicy(), data)
