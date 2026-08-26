"""Independent twin critics with three phase heads and a strict 124D input."""

from __future__ import annotations

import torch
from torch import nn


def gather_stage_values(values: torch.Tensor, stages: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2 or values.shape[-1] != 3:
        raise ValueError("phase critic values must have shape [B,3]")
    indices = torch.as_tensor(stages, device=values.device, dtype=torch.long).reshape(-1, 1)
    if indices.shape[0] != values.shape[0]:
        raise ValueError("stage labels must contain one value per critic row")
    if torch.any((indices < 0) | (indices > 2)):
        raise ValueError("stage labels must be in [0,2]")
    return values.gather(1, indices)


class PhaseCritic(nn.Module):
    input_dim = 124
    num_stages = 3

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.backbone = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList(nn.Linear(hidden_dim, 1) for _ in range(3))

    def forward(self, joint_observation_action: torch.Tensor) -> torch.Tensor:
        if (
            joint_observation_action.ndim != 2
            or joint_observation_action.shape[-1] != self.input_dim
        ):
            raise ValueError(
                "PRRAC critic input must be [B,124]; stage labels are not input features"
            )
        hidden = self.backbone(joint_observation_action)
        return torch.cat([head(hidden) for head in self.heads], dim=-1)


class PhaseTwinCritic(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.critic1 = PhaseCritic(hidden_dim)
        self.critic2 = PhaseCritic(hidden_dim)

    def forward(self, joint_observation_action: torch.Tensor):
        return self.critic1(joint_observation_action), self.critic2(
            joint_observation_action
        )


__all__ = ("PhaseCritic", "PhaseTwinCritic", "gather_stage_values")
