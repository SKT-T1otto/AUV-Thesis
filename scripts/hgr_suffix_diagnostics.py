"""Observe one unchanged HGR suffix SGD update without constructing a runtime.

The undiscounted-at-handoff decomposition is diagnostic only.  The sole update
is the existing ``update_suffix`` with its full-batch denominator and absolute
discount.  No Trainer, environment, random stream, or checkpoint is used here.
"""
from __future__ import annotations

import copy
import json
import math

import numpy as np
import torch

from chapter3_bser.models.hgr.estimator import reward_to_go, update_suffix
from chapter3_bser.models.hgr.phase1 import behavior_identity, identical_behavior

SCHEMA = "hgr.suffix_update_diagnostics.v1"


def _flat(parameters):
    return np.concatenate([p.detach().cpu().reshape(-1).double().numpy() for p in parameters])


def _gradient_array(parameters, gradients):
    return np.concatenate([
        np.zeros(p.numel(), dtype=np.float64) if g is None
        else g.detach().cpu().reshape(-1).double().numpy()
        for p, g in zip(parameters, gradients)
    ])


def _norm(array):
    return float(np.linalg.norm(array))


def _finite(value, label):
    if not bool(torch.isfinite(torch.as_tensor(value)).all()):
        raise ValueError(f"nonfinite {label}")


def _validate_trajectories(trajectories):
    if not trajectories:
        raise ValueError("suffix diagnostic requires a nonempty full batch")
    for trajectory in trajectories:
        records, rewards = trajectory["records"], trajectory["rewards"]
        if not records or len(records) != len(rewards):
            raise ValueError("inconsistent suffix trajectory records/rewards")
        if trajectory.get("consumed", False) or trajectory.get("trajectory_complete") is False:
            raise ValueError("suffix diagnostic requires fresh complete trajectories")
        _finite(rewards, "trajectory rewards")
        tau = trajectory["tau"]
        if tau is not None and (isinstance(tau, bool) or not isinstance(tau, int)
                                or not 0 <= tau < len(records)):
            raise ValueError("invalid suffix handoff tau")
        for t, record in enumerate(records):
            if record.get("t", t) != t or bool(record["suffix"]) != (tau is not None and t >= tau):
                raise ValueError("suffix stopping-time/record clock mismatch")
            if len(record["latents"]) != 4 or len(record["observations"]) != 4:
                raise ValueError("suffix diagnostic requires four-agent records")
            for observation in record["observations"]:
                _finite(observation, "record observations")
            for latent in record["latents"]:
                if latent is not None:
                    _finite(latent, "record latents")
            if record["suffix"] and any(z is not None for z in record["latents"][:3]):
                raise ValueError("suffix record contains an active searcher")


def _group_gradients(named_parameters):
    result = {}
    for group in ("mean_network", "log_std"):
        selected = [(name, p) for name, p in named_parameters
                    if (name == "log_std" or name.endswith(".log_std")) == (group == "log_std")]
        parameters = [p for _, p in selected]
        gradients = [p.grad for p in parameters]
        vector = _gradient_array(parameters, gradients) if parameters else np.empty(0)
        result[group] = dict(
            gradient_norm=_norm(vector), parameter_tensors=len(parameters),
            parameter_elements=sum(p.numel() for p in parameters),
            none_gradient_tensors=sum(g is None for g in gradients),
            zero_gradient_tensors=sum(g is not None and bool((g == 0).all()) for g in gradients),
            finite_gradient_tensors=sum(g is not None and bool(torch.isfinite(g).all()) for g in gradients),
            nonzero_gradient_elements=int(np.count_nonzero(vector)),
        )
    return result


def _behavior_change(old, new, trajectories):
    """Native forward parameters, then float64-only numerical diagnostics."""
    kls, mean_changes, std_changes, action_changes = [], [], [], []
    saturated_old_mean = saturated_new_mean = total_mean_elements = 0
    with torch.no_grad():
        for trajectory in trajectories:
            for record in trajectory["records"]:
                latent = record["latents"][3]
                if not record["suffix"] or latent is None:
                    continue
                obs = record["observations"][3]
                mu0, sd0 = old.phi.distribution_parameters(obs)
                mu1, sd1 = new.phi.distribution_parameters(obs)
                for name, value in (("old mean", mu0), ("new mean", mu1),
                                    ("old std", sd0), ("new std", sd1)):
                    _finite(value, name)
                mu0, sd0, mu1, sd1 = [v.detach().cpu().reshape(-1).double().numpy()
                                     for v in (mu0, sd0, mu1, sd1)]
                z = np.asarray(latent, dtype=np.float64).reshape(-1)
                if z.shape != mu0.shape or mu0.shape != mu1.shape or np.any(sd0 <= 0) or np.any(sd1 <= 0):
                    raise ValueError("invalid suffix Gaussian distribution shape/scale")
                log_ratio = np.log1p((sd0 - sd1) / sd1)
                x = 2 * log_ratio
                variance_term = np.maximum(0., .5 * (np.expm1(x) - x))
                kl = float(np.sum(variance_term + .5 * ((mu0 - mu1) / sd1) ** 2))
                noise = (z - mu0) / sd0
                # Reuse the recorded innovation; never draw extra policy noise.
                action_difference = np.tanh(mu1 + sd1 * noise) - np.tanh(z)
                kls.append(kl)
                mean_changes.append(float(np.max(np.abs(mu1 - mu0))))
                std_changes.append(float(np.max(np.abs(sd1 - sd0))))
                action_changes.append(float(np.max(np.abs(action_difference))))
                total_mean_elements += mu0.size
                epsilon0 = float(getattr(old.phi, "mean_epsilon", 1e-6))
                epsilon1 = float(getattr(new.phi, "mean_epsilon", 1e-6))
                saturated_old_mean += int(np.count_nonzero(np.abs(np.tanh(mu0)) >= 1 - epsilon0))
                saturated_new_mean += int(np.count_nonzero(np.abs(np.tanh(mu1)) >= 1 - epsilon1))
    values = [*kls, *mean_changes, *std_changes, *action_changes]
    if values and not np.isfinite(values).all():
        raise ValueError("nonfinite suffix behavior diagnostics")
    return dict(
        observation_count=len(kls), observation_source="recorded_pre_update_suffix_trajectories",
        kl_direction="old_to_new", kl_mean=float(np.mean(kls)) if kls else None,
        kl_max=max(kls) if kls else None,
        max_abs_mu_change=max(mean_changes) if mean_changes else None,
        max_abs_std_change=max(std_changes) if std_changes else None,
        max_abs_shared_noise_action_change=max(action_changes) if action_changes else None,
        old_mean_boundary_element_count=saturated_old_mean,
        new_mean_boundary_element_count=saturated_new_mean,
        mean_element_count=total_mean_elements,
    )


def diagnose_suffix_update(policy, trajectories, *, gamma, lr, contract):
    """Return ``(updated_copy, JSON-safe report, float64 NumPy vectors)``.

    ``g_phi`` is the ascent gradient actually used by the original SGD update.
    ``trajectory_h`` removes only gamma**tau for observation; it never supplies
    the update.  Diagnostic autograd releases every individual score graph.
    Input policy and trajectory objects remain unmodified.
    """
    if isinstance(gamma, bool) or not math.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError("gamma must be finite in (0,1]")
    if isinstance(lr, bool) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if "gamma" in contract and contract["gamma"] != gamma:
        raise ValueError("suffix diagnostic gamma differs from runtime contract")
    _validate_trajectories(trajectories)
    input_identity = behavior_identity(policy, contract)
    theta_identity = behavior_identity(policy.theta_minus, contract)
    old_phi_identity = behavior_identity(policy.phi, contract)
    for trajectory in trajectories:
        for key, identity in (("theta_behavior_sha256", theta_identity),
                              ("suffix_behavior_sha256", old_phi_identity)):
            if key in trajectory and trajectory[key] != identity.sha256:
                raise ValueError(f"stale suffix trajectory {key}")
    new = copy.deepcopy(policy)
    new.phi.requires_grad_(True)
    if hasattr(new, "assert_isolated"):
        new.assert_isolated()
    named = list(new.phi.named_parameters())
    parameters = [p for _, p in named]
    if not parameters:
        raise ValueError("suffix policy has no parameters")
    before_tensors = [p.detach().clone() for p in parameters]
    before = _flat(parameters)
    rows, h_vectors, discounted_vectors = [], [], []
    effective_steps = 0
    for index, trajectory in enumerate(trajectories):
        h = np.zeros(before.size, dtype=np.float64)
        tau = trajectory["tau"]
        returns = reward_to_go(trajectory["rewards"], gamma)
        score_steps = 0
        if tau is not None:
            for t in range(tau, len(trajectory["records"])):
                record = trajectory["records"][t]
                score = new.score_log_prob(record, suffix=True)
                _finite(score, "suffix score")
                gradients = torch.autograd.grad(score, parameters, allow_unused=True)
                score_gradient = _gradient_array(parameters, gradients)
                if not np.isfinite(score_gradient).all():
                    raise ValueError("nonfinite suffix score gradient")
                h += gamma ** (t - tau) * float(returns[t]) * score_gradient
                score_steps += int(record["latents"][3] is not None)
                del gradients, score
        weight = 0. if tau is None else gamma ** tau
        weighted = weight * h
        if not np.isfinite(h).all() or not np.isfinite(weighted).all():
            raise ValueError("nonfinite suffix gradient decomposition")
        h_vectors.append(h)
        discounted_vectors.append(weighted)
        effective_steps += score_steps
        rows.append(dict(trajectory_index=index, tau=tau, gamma_tau=None if tau is None else weight,
                         effective_suffix_score_steps=score_steps, h_norm=_norm(h),
                         discounted_h_norm=_norm(weighted), trajectory_length=len(returns)))

    # Exactly one unchanged training update.  Its remaining .grad values were
    # computed at the pre-update policy, and are the authoritative native grads.
    optimizer = torch.optim.SGD(parameters, lr=lr)
    loss = update_suffix(new, optimizer, trajectories, gamma)
    if not math.isfinite(loss):
        raise ValueError("nonfinite original suffix loss")
    native_gradients = [p.grad for p in parameters]
    g_phi = -_gradient_array(parameters, native_gradients)
    after = _flat(parameters)
    if not np.isfinite(g_phi).all() or not np.isfinite(after).all():
        raise ValueError("nonfinite original suffix update")
    expected_native = [previous if gradient is None else previous.clone().add_(gradient, alpha=-lr)
                       for previous, gradient in zip(before_tensors, native_gradients)]
    if not all(torch.equal(expected, p.detach()) for expected, p in zip(expected_native, parameters)):
        raise RuntimeError("original suffix SGD differs from its native-dtype step")
    delta = after - before
    ideal = lr * g_phi
    rounding = delta - ideal
    h_matrix = np.stack(h_vectors)
    discounted_matrix = np.stack(discounted_vectors)
    sum_vector = discounted_matrix.sum(axis=0)
    denominator = sum(_norm(vector) for vector in discounted_vectors)
    decomposition_g = sum_vector / len(trajectories)
    new_theta_identity = behavior_identity(new.theta_minus, contract)
    if not identical_behavior(theta_identity, new_theta_identity):
        raise RuntimeError("suffix update changed theta behavior")
    new_phi_identity = behavior_identity(new.phi, contract)
    if not identical_behavior(input_identity, behavior_identity(policy, contract)):
        raise RuntimeError("suffix diagnostic changed its input policy")
    report = dict(
        schema=SCHEMA, status="EXERCISED" if effective_steps else "NOT_EXERCISED",
        batch_size=len(trajectories), handoff_count=sum(t["tau"] is not None for t in trajectories),
        effective_suffix_score_steps=effective_steps, gamma=gamma, lr=lr,
        parameter_updates=1, suffix_loss=loss, trajectory_rows=rows,
        gradient=dict(g_phi_norm=_norm(g_phi), groups=_group_gradients(named),
                      cancellation_ratio=None if denominator == 0 else _norm(sum_vector) / denominator,
                      decomposition_gradient_norm=_norm(decomposition_g),
                      native_vs_float64_decomposition_error_norm=_norm(g_phi - decomposition_g)),
        sgd_update=dict(delta_phi_norm=_norm(delta), max_abs_delta_phi=float(np.max(np.abs(delta))),
                        changed_parameter_elements=int(np.count_nonzero(delta)),
                        parameter_elements=before.size, ideal_delta_phi_norm=_norm(ideal),
                        rounding_residual_norm=_norm(rounding),
                        max_abs_rounding_residual=float(np.max(np.abs(rounding))),
                        native_step_bitwise_match=True),
        behavior=dict(theta_identity_unchanged=True, theta_sha256=theta_identity.sha256,
                      phi0_sha256=old_phi_identity.sha256, phi1_sha256=new_phi_identity.sha256,
                      phi_identity_changed=not identical_behavior(old_phi_identity, new_phi_identity),
                      **_behavior_change(policy, new, trajectories)),
        original_policy_unchanged=True,
        diagnostic_discount_removal_used_for_update=False,
    )
    json.dumps(report, allow_nan=False)
    vectors = dict(g_phi=g_phi, trajectory_h=h_matrix, trajectory_discounted_h=discounted_matrix,
                   delta_phi=delta, ideal_delta_phi=ideal, rounding_residual=rounding,
                   decomposition_g_phi=decomposition_g)
    return new, report, vectors
