"""Public-observation residual trust gate with monotone alignment response."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


class ResidualTrustGate(nn.Module):
    PUBLIC_CONTEXT_DIM = 1 + 1 + 1 + 1 + 1 + 2

    def __init__(
        self,
        embedding_dim: int = 128,
        num_stages: int = 3,
        hidden_dim: int = 64,
        initial_mean: float = 0.75,
        alignment_scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0.0 < float(initial_mean) < 1.0:
            raise ValueError("gate_initial_mean must be inside (0,1)")
        input_dim = int(embedding_dim) + int(num_stages) + self.PUBLIC_CONTEXT_DIM
        self.base_mlp = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.base_mlp[-1].weight)
        nn.init.constant_(
            self.base_mlp[-1].bias,
            math.log(float(initial_mean) / (1.0 - float(initial_mean))),
        )
        self.raw_alignment_scale = nn.Parameter(
            torch.tensor(_inverse_softplus(alignment_scale_init), dtype=torch.float32)
        )

    @property
    def positive_alignment_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_alignment_scale)

    @staticmethod
    def alignment_cosine(
        residual_mix: torch.Tensor, navigation_direction: torch.Tensor
    ) -> torch.Tensor:
        if residual_mix.shape != navigation_direction.shape or residual_mix.shape[-1] != 3:
            raise ValueError("alignment inputs must have matching [B,3] shapes")
        residual_norm = torch.linalg.vector_norm(residual_mix, dim=-1, keepdim=True)
        navigation_norm = torch.linalg.vector_norm(
            navigation_direction, dim=-1, keepdim=True
        )
        denominator = (residual_norm * navigation_norm).clamp_min(1e-8)
        cosine = (residual_mix * navigation_direction).sum(dim=-1, keepdim=True) / denominator
        valid = (residual_norm > 1e-8) & (navigation_norm > 1e-8)
        return torch.where(valid, cosine.clamp(-1.0, 1.0), torch.zeros_like(cosine))

    def gate_from_base_logit(
        self, base_logit: torch.Tensor, alignment_cosine: torch.Tensor
    ) -> torch.Tensor:
        return torch.sigmoid(base_logit + self.positive_alignment_scale * alignment_cosine)

    def forward(
        self,
        embedding: torch.Tensor,
        router_probabilities: torch.Tensor,
        observation: torch.Tensor,
        residual_mix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observation.ndim != 2 or observation.shape[-1] != 28:
            raise ValueError("trust gate requires exactly the public 28D observation")
        public_context = torch.cat(
            (
                observation[:, 15:16],
                observation[:, 17:18],
                observation[:, 18:19],
                observation[:, 19:20],
                observation[:, 21:22],
                observation[:, 26:28],
            ),
            dim=-1,
        )
        base_logit = self.base_mlp(
            torch.cat((embedding, router_probabilities, public_context), dim=-1)
        )
        alignment = self.alignment_cosine(residual_mix, observation[:, 9:12])
        gate = self.gate_from_base_logit(base_logit, alignment)
        return gate, alignment, base_logit


__all__ = ("ResidualTrustGate",)
