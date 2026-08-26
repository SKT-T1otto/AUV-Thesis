"""Phase-routed residual actor; waypoint prior remains environment-owned."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from .phase_router import PhaseRouter
from .residual_experts import ResidualExperts
from .residual_trust_gate import ResidualTrustGate


class PRRACActorOutput(NamedTuple):
    gated_residual_action: torch.Tensor
    router_logits: torch.Tensor
    router_probabilities: torch.Tensor
    expert_actions: torch.Tensor
    trust_gate: torch.Tensor
    alignment_cosine: torch.Tensor
    residual_mix: torch.Tensor


class PhaseRoutedResidualActor(nn.Module):
    observation_dim = 28
    action_dim = 3

    def __init__(
        self,
        hidden_dim: int = 128,
        expert_hidden_dim: int = 128,
        router_temperature: float = 1.0,
        gate_initial_mean: float = 0.75,
        alignment_scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.observation_dim),
            nn.Linear(self.observation_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.router = PhaseRouter(hidden_dim, 3, router_temperature)
        self.residual_experts = ResidualExperts(hidden_dim, expert_hidden_dim, 3)
        self.trust_gate_module = ResidualTrustGate(
            hidden_dim,
            3,
            hidden_dim=max(32, hidden_dim // 2),
            initial_mean=gate_initial_mean,
            alignment_scale_init=alignment_scale_init,
        )

    def forward(self, observation: torch.Tensor) -> PRRACActorOutput:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if observation.ndim != 2 or observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"PRRAC actor expects [B,28], got {tuple(observation.shape)}"
            )
        embedding = self.encoder(observation)
        logits, probabilities = self.router(embedding)
        expert_actions = self.residual_experts(embedding)
        residual_mix = (probabilities.unsqueeze(-1) * expert_actions).sum(dim=1)
        gate, alignment, _ = self.trust_gate_module(
            embedding, probabilities, observation, residual_mix
        )
        gated = torch.clamp(gate * residual_mix, -1.0, 1.0)
        return PRRACActorOutput(
            gated,
            logits,
            probabilities,
            expert_actions,
            gate,
            alignment,
            residual_mix,
        )


__all__ = ("PRRACActorOutput", "PhaseRoutedResidualActor")
