"""Deterministic soft phase router."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PhaseRouter(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        num_stages: int = 3,
        temperature: float = 1.0,
        *,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim is not None:
            input_dim = int(hidden_dim)
        if int(num_stages) != 3:
            raise ValueError("PRRAC v1 requires exactly three stages")
        if float(temperature) <= 0.0:
            raise ValueError("router temperature must be positive")
        self.input_dim = int(input_dim)
        self.num_stages = int(num_stages)
        self.temperature = float(temperature)
        self.logit_head = nn.Linear(self.input_dim, self.num_stages)

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if embedding.ndim != 2 or embedding.shape[-1] != self.input_dim:
            raise ValueError(
                f"router expects [B,{self.input_dim}], got {tuple(embedding.shape)}"
            )
        logits = self.logit_head(embedding)
        probabilities = F.softmax(logits / self.temperature, dim=-1)
        return logits, probabilities


__all__ = ("PhaseRouter",)
