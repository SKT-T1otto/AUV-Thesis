"""Independent search, intercept, and hold residual expert heads."""

from __future__ import annotations

import torch
from torch import nn


class ResidualExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(action_dim)),
            nn.Tanh(),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


class ResidualExperts(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        expert_hidden_dim: int = 128,
        action_dim: int = 3,
    ) -> None:
        super().__init__()
        if int(action_dim) != 3:
            raise ValueError("PRRAC residual action dimension must be 3")
        self.experts = nn.ModuleList(
            ResidualExpert(input_dim, expert_hidden_dim, action_dim) for _ in range(3)
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return torch.stack([expert(embedding) for expert in self.experts], dim=1)


__all__ = ("ResidualExpert", "ResidualExperts")
