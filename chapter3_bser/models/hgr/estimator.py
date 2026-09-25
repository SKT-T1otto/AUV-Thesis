"""Score-matched estimators, including the strong direct boundary baseline."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def reward_to_go(rewards, gamma):
    result = [0.0] * len(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = float(rewards[t]) + gamma * running
        result[t] = running
    return result


class BoundaryPredictor(nn.Module):
    """Signed scalar regression; normalization fitted only to independent pilots."""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.register_buffer("center", torch.zeros(input_dim))
        self.register_buffer("scale", torch.ones(input_dim))
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, features):
        x = torch.as_tensor(features, dtype=self.center.dtype, device=self.center.device)
        return self.network((x - self.center) / self.scale).squeeze(-1)

    def fit(self, features, labels, *, updates, lr):
        self.requires_grad_(True)
        if not features:
            self.requires_grad_(False)
            return dict(updates=0, mse=None, samples=0)
        x = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        y = torch.as_tensor(labels, dtype=torch.float32).detach()
        self.center.copy_(x.mean(0)); self.scale.copy_(x.std(0, unbiased=False).clamp_min(1e-4))
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(updates):
            optimizer.zero_grad(set_to_none=True)
            loss = (self(x) - y).square().mean()
            loss.backward(); optimizer.step()
        with torch.no_grad():
            mse = float((self(x) - y).square().mean())
        self.requires_grad_(False)
        return dict(updates=updates, mse=mse, samples=len(labels), optimizer=optimizer.state_dict())


def prefix_losses(policy, trajectories, predictions, draws, *, gamma, method="hgr", ablation="none", bypass=None):
    """Each draw is (index, q_index, fresh branch label). N includes failures.

    The returned losses share one immutable theta and must be summed before the
    single optimizer step. Returns/labels/predictions are constants, never graphs.
    """
    n, k = len(trajectories), len(draws)
    if n == 0 or len(predictions) != n:
        raise ValueError("empty or inconsistent main batch")
    if bypass is not None:
        from .phase1 import NoUpdateProof
        if not isinstance(bypass, NoUpdateProof):
            raise ValueError("unverified K=0 bypass")
        if gamma != bypass.contract.get("gamma"):
            raise ValueError("bypass discount contract mismatch")
        bypass.validate(policy, trajectories, predictions, draws, method, ablation)
    if method != "stochastic_direct_mc" and k == 0 and bypass is None:
        raise ValueError("corrected estimator requires positive K")
    zero = next(policy.theta_minus.parameters()).sum() * 0
    old, prediction, correction = zero, zero, zero
    boundary_scores = []
    for e, trajectory in enumerate(trajectories):
        records, rewards, tau = trajectory["records"], trajectory["rewards"], trajectory["tau"]
        boundary = len(rewards) if tau is None else tau
        if method == "direct_boundary_corrected":
            returns = reward_to_go(rewards[:boundary], gamma)
        else:
            returns = reward_to_go(rewards, gamma)
        logs = [policy.score_log_prob(record) for record in records[:boundary]]
        old = old - sum((gamma ** t * log * float(returns[t]) for t, log in enumerate(logs)), zero) / n
        c = zero if tau is None else gamma ** tau * sum(logs, zero)
        boundary_scores.append(c)
        if method != "stochastic_direct_mc" and bypass is None:
            prediction = prediction - c * float(predictions[e]) / n
    if method != "stochastic_direct_mc":
        for index, probability, label in draws:
            if not 0 <= index < n or not 0 < probability <= 1:
                raise ValueError("invalid correction draw")
            if trajectories[index]["tau"] is None and float(label) != 0:
                raise ValueError("no-handoff record cannot have a branch label")
            correction = correction - boundary_scores[index] * (float(label) - float(predictions[index])) / (n * k * probability)
    if ablation == "remove_correction":
        correction = zero
    elif ablation == "remove_prediction":
        prediction = zero
    elif ablation not in ("none", "predictor_zero"):
        raise ValueError("unknown ablation")
    return dict(old=old, prediction=prediction, correction=correction)


def gradient_vector(loss, parameters, *, retain_graph=True):
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=retain_graph)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1) for p, g in zip(parameters, gradients)])


def update_prefix(policy, optimizer, trajectories, predictions, draws, *, gamma, method, ablation="none", bypass=None):
    from .policy import weights_hash
    current_hash = weights_hash(policy.theta_minus)
    if any(t.get("consumed", False) or t.get("theta_hash", current_hash) != current_hash for t in trajectories):
        raise ValueError("stale prefix batch: theta changed or batch already consumed")
    if not isinstance(optimizer, torch.optim.SGD) or any(group.get("momentum", 0) or group.get("weight_decay", 0) or group.get("maximize", False) for group in optimizer.param_groups):
        raise ValueError("score-matched prefix update requires plain SGD")
    parameters = list(policy.theta_minus.parameters())
    losses = prefix_losses(policy, trajectories, predictions, draws, gamma=gamma, method=method, ablation=ablation, bypass=bypass)
    vectors = {name: -gradient_vector(loss, parameters) for name, loss in losses.items()}
    delta = vectors["prediction"] + vectors["correction"]
    full = vectors["old"] + delta
    if not bool(torch.isfinite(full).all()):
        raise ValueError("nonfinite score estimator; parameters were not updated")
    before = torch.cat([p.detach().reshape(-1).clone() for p in parameters])
    optimizer.zero_grad(set_to_none=True)
    sum(losses.values()).backward()
    optimizer.step()
    for trajectory in trajectories:
        trajectory["consumed"] = True
    after = torch.cat([p.detach().reshape(-1) for p in parameters])
    lr = optimizer.param_groups[0]["lr"]
    if not torch.allclose(after - before, lr * full, atol=2e-7, rtol=2e-4):
        raise RuntimeError("prefix SGD increment does not match complete score estimator")
    return dict(g0_norm=float(vectors["old"].norm()), prediction_norm=float(vectors["prediction"].norm()),
                correction_norm=float(vectors["correction"].norm()), delta_g_norm=float(delta.norm()),
                g1_norm=float(full.norm()), parameter_change_norm=float((after - before).norm()),
                loss_old=float(losses["old"].detach()), loss_delta=float((losses["prediction"] + losses["correction"]).detach()))


def update_suffix(policy, optimizer, trajectories, gamma):
    # The denominator is the full initial-distribution batch, including failures.
    loss = next(policy.phi.parameters()).sum() * 0
    for trajectory in trajectories:
        returns = reward_to_go(trajectory["rewards"], gamma)
        for t, record in enumerate(trajectory["records"]):
            if record["suffix"]:
                loss = loss - gamma ** t * policy.score_log_prob(record, suffix=True) * returns[t] / len(trajectories)
    optimizer.zero_grad(set_to_none=True)
    loss.backward(); optimizer.step()
    return float(loss.detach())
