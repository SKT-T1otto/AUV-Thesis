"""28D local stochastic policies; all gates live inside the Gaussian mean.

Only the final parameter-independent tanh transforms a sampled latent. The
PRRAC mean mapping is approximate within epsilon at the action boundaries.
"""
from __future__ import annotations

import copy
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from chapter3_bser.models.prrac.phase_routed_actor import PhaseRoutedResidualActor


def weights_hash(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        item = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((item.dtype, tuple(item.shape))).encode())
        digest.update(item.numpy().tobytes())
    return digest.hexdigest()


class TanhGaussianPolicy(nn.Module):
    def __init__(self, actor_config=None, initial_std=0.12, mean_epsilon=1e-6):
        super().__init__()
        if not 0 < initial_std or not 0 < mean_epsilon < 0.01:
            raise ValueError("invalid Gaussian initialization or mean epsilon")
        self.mean_actor = PhaseRoutedResidualActor(**dict(actor_config or {}))
        self.log_std = nn.Parameter(torch.full((3,), math.log(initial_std)))
        self.mean_epsilon = float(mean_epsilon)

    def distribution_parameters(self, observation):
        observation = torch.as_tensor(observation, dtype=self.log_std.dtype, device=self.log_std.device)
        bounded = self.mean_actor(observation).gated_residual_action
        mu = torch.atanh(bounded.clamp(-1 + self.mean_epsilon, 1 - self.mean_epsilon))
        # A fixed bound on log standard deviation defines the policy family.
        std = self.log_std.clamp(-10.0, 2.0).exp().expand_as(mu)
        return mu, std

    @torch.no_grad()
    def sample_action(self, observation, *, generator=None, standard_noise=None):
        mu, std = self.distribution_parameters(observation)
        if standard_noise is None:
            latent = mu + std * torch.randn(mu.shape, generator=generator, dtype=mu.dtype, device=mu.device)
        else:
            noise = torch.as_tensor(standard_noise, dtype=mu.dtype, device=mu.device).detach()
            if noise.numel() != mu.numel() or not bool(torch.isfinite(noise).all()):
                raise ValueError("invalid supplied standard policy noise")
            latent = mu + std * noise.reshape(mu.shape)
        return latent.tanh(), latent.detach().clone()

    def log_prob(self, observation, pre_tanh_latent):
        mu, std = self.distribution_parameters(observation)
        # Fixed observations and actions: this is a likelihood-ratio score.
        latent = torch.as_tensor(pre_tanh_latent, dtype=mu.dtype, device=mu.device).detach()
        normal = -0.5 * ((latent - mu) / std).square() - std.log() - 0.5 * math.log(2 * math.pi)
        log_jacobian = 2 * (math.log(2.0) - latent - F.softplus(-2 * latent))
        return (normal - log_jacobian).sum(-1)

    @torch.no_grad()
    def deterministic_action(self, observation):
        return self.distribution_parameters(observation)[0].tanh()


class HandoffPolicy(nn.Module):
    """Four independent prefix actors, including standby, and one suffix actor."""
    def __init__(self, config=None):
        super().__init__()
        config = dict(config or {})
        self.config = copy.deepcopy(config)
        kwargs = dict(actor_config=config.get("actor"),
                      initial_std=float(config.get("initial_std", 0.12)),
                      mean_epsilon=float(config.get("mean_epsilon", 1e-6)))
        self.theta_minus = nn.ModuleList([TanhGaussianPolicy(**kwargs) for _ in range(4)])
        self.phi = TanhGaussianPolicy(**kwargs)
        self.assert_isolated()

    def assert_isolated(self):
        blocks = [*self.theta_minus, self.phi]
        seen = set()
        for block in blocks:
            storage = {v.untyped_storage().data_ptr() for v in (*block.parameters(), *block.buffers()) if v.numel()}
            if seen & storage:
                raise RuntimeError("handoff policy blocks share parameter or buffer storage")
            seen.update(storage)

    @torch.no_grad()
    def actions(self, observations, *, suffix, active, generator, deterministic=False, standard_noise=None):
        if standard_noise is not None and tuple(standard_noise.shape) != (4, 3):
            raise ValueError("branch policy noise must have shape (4,3)")
        actions, latents = [], []
        for i, observation in enumerate(observations):
            if not active[i]:
                actions.append(torch.zeros(3)); latents.append(None)
                continue
            actor = self.phi if suffix and i == 3 else self.theta_minus[i]
            if suffix and i != 3:
                raise RuntimeError("HGR v1 requires frozen searchers after reliable handoff")
            if deterministic:
                actions.append(actor.deterministic_action(observation).reshape(3)); latents.append(None)
            else:
                if standard_noise is None:
                    action, latent = actor.sample_action(observation, generator=generator)
                else:
                    action, latent = actor.sample_action(observation, generator=generator, standard_noise=standard_noise[i])
                actions.append(action.reshape(3)); latents.append(latent.reshape(3).cpu().numpy().copy())
        return torch.stack(actions), latents

    def score_log_prob(self, record, *, suffix=False):
        values = []
        for i, latent in enumerate(record["latents"]):
            if latent is None:
                continue
            if suffix:
                if i != 3 or not record["suffix"]:
                    continue
                actor = self.phi
            else:
                if record["suffix"]:
                    continue
                actor = self.theta_minus[i]
            observation = torch.as_tensor(record["observations"][i]).detach()
            values.append(actor.log_prob(observation, latent).sum())
        block = self.phi if suffix else self.theta_minus
        zero = next(block.parameters()).sum() * 0
        return sum(values, zero)

    def import_prrac_means(self, actors):
        if len(actors) != 4:
            raise ValueError("mean initialization requires exactly four PRRAC actors")
        for index, actor in enumerate(actors):
            expected, actual = self.theta_minus[index].mean_actor.state_dict(), actor.state_dict()
            if expected.keys() != actual.keys() or any(expected[k].shape != actual[k].shape for k in expected):
                raise ValueError("PRRAC-to-Gaussian mean mapping keys/shapes mismatch")
            if any(not bool(torch.isfinite(value).all()) for value in actual.values()):
                raise ValueError("nonfinite PRRAC mean initialization")
        for index, actor in enumerate(actors):
            state = actor.state_dict()
            self.theta_minus[index].mean_actor.load_state_dict(state, strict=True)
        self.phi.mean_actor.load_state_dict(actors[3].state_dict(), strict=True)
        self.assert_isolated()
        return dict(mapping="mu=atanh(clamp(PRRAC_action,-1+eps,1-eps))",
                    epsilon=self.phi.mean_epsilon, max_boundary_action_error=self.phi.mean_epsilon,
                    initial_std=self.config.get("initial_std", 0.12),
                    standby_and_suffix="independent copies; no optimizer, critic or replay imported")
