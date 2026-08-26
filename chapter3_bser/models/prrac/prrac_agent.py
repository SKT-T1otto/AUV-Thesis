"""Independent PRRAC agent state; no mutation of core DDPGAgent fields."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch.optim import Adam

from core.algorithms.misc import hard_update
from core.algorithms.noise import OUNoise

from .phase_routed_actor import PhaseRoutedResidualActor
from .phase_twin_critic import PhaseCritic


class PRRACAgent:
    def __init__(
        self,
        *,
        actor_config: Mapping[str, Any] | None = None,
        critic_hidden_dim: int = 256,
        lr_actor: float = 1e-3,
        lr_critic: float = 5e-4,
        noise_sigma: float = 0.12,
    ) -> None:
        actor_kwargs = dict(actor_config or {})
        self.actor = PhaseRoutedResidualActor(**actor_kwargs)
        self.target_actor = PhaseRoutedResidualActor(**actor_kwargs)
        self.critic1 = PhaseCritic(critic_hidden_dim)
        self.critic2 = PhaseCritic(critic_hidden_dim)
        self.target_critic1 = PhaseCritic(critic_hidden_dim)
        self.target_critic2 = PhaseCritic(critic_hidden_dim)
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic1, self.critic1)
        hard_update(self.target_critic2, self.critic2)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=float(lr_actor))
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=float(lr_critic))
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=float(lr_critic))
        self.noise = OUNoise(3, sigma=float(noise_sigma))

    @property
    def policy(self):
        return self.actor

    @property
    def target_policy(self):
        return self.target_actor

    def _device(self) -> torch.device:
        return next(self.actor.parameters()).device

    def sync_noise_device(self) -> None:
        self.noise.to(device=self._device())

    def step(self, observation: torch.Tensor, explore: bool = False) -> torch.Tensor:
        self.sync_noise_device()
        with torch.no_grad():
            action = self.actor(observation).gated_residual_action
            if explore:
                noise = self.noise.sample().to(action)
                if action.ndim == 2:
                    noise = noise.unsqueeze(0).expand_as(action)
                action = torch.clamp(action + noise, -1.0, 1.0)
            return action

    def reset_noise(self) -> None:
        self.noise.reset()

    def scale_noise(self, value: float, *, multiply: bool = False) -> None:
        self.noise.sigma = (
            float(self.noise.sigma) * float(value) if multiply else float(value)
        )

    @staticmethod
    def _cpu(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: PRRACAgent._cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [PRRACAgent._cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(PRRACAgent._cpu(item) for item in value)
        return value

    def training_state_dict(self) -> dict[str, Any]:
        return {
            "actor": self._cpu(self.actor.state_dict()),
            "target_actor": self._cpu(self.target_actor.state_dict()),
            "actor_optimizer": self._cpu(self.actor_optimizer.state_dict()),
            "critic1": self._cpu(self.critic1.state_dict()),
            "critic2": self._cpu(self.critic2.state_dict()),
            "target_critic1": self._cpu(self.target_critic1.state_dict()),
            "target_critic2": self._cpu(self.target_critic2.state_dict()),
            "critic1_optimizer": self._cpu(self.critic1_optimizer.state_dict()),
            "critic2_optimizer": self._cpu(self.critic2_optimizer.state_dict()),
            "noise_sigma": float(self.noise.sigma),
            "noise_state": self.noise.state.detach().cpu().clone(),
        }

    def load_training_state_dict(self, state: Mapping[str, Any]) -> None:
        for name in (
            "actor",
            "target_actor",
            "critic1",
            "critic2",
            "target_critic1",
            "target_critic2",
        ):
            getattr(self, name).load_state_dict(state[name])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic1_optimizer.load_state_dict(state["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(state["critic2_optimizer"])
        self.noise.sigma = float(state.get("noise_sigma", self.noise.sigma))
        self.sync_noise_device()
        if "noise_state" in state:
            self.noise.state.copy_(state["noise_state"].to(self.noise.state))


__all__ = ("PRRACAgent",)
