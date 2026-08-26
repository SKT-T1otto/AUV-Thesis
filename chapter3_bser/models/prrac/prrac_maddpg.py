"""Independent four-agent PRRAC learner with phase-aware twin-Q updates."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from core.algorithms.misc import soft_update

from .phase_twin_critic import gather_stage_values
from .prrac_agent import PRRACAgent


class PRRACMADDPG:
    SCHEMA = "bser.phase1c.prrac.algorithm_state.v1"
    OBS_DIMS = (28, 28, 28, 28)
    ACTION_DIMS = (3, 3, 3, 3)
    CRITIC_DIM = 124

    def __init__(
        self,
        *,
        architecture: Mapping[str, Any] | None = None,
        loss: Mapping[str, Any] | None = None,
        gamma: float = 0.95,
        tau: float = 0.005,
        lr_actor: float = 1e-3,
        lr_critic: float = 5e-4,
        noise_sigmas=(0.18, 0.14, 0.10, 0.08),
    ) -> None:
        architecture = dict(architecture or {})
        loss = dict(loss or {})
        if int(architecture.get("num_stages", 3)) != 3:
            raise ValueError("PRRAC requires exactly three stages")
        actor_config = {
            "hidden_dim": int(architecture.get("encoder_hidden_dim", 128)),
            "expert_hidden_dim": int(architecture.get("expert_hidden_dim", 128)),
            "router_temperature": float(architecture.get("router_temperature", 1.0)),
            "gate_initial_mean": float(architecture.get("gate_initial_mean", 0.75)),
            "alignment_scale_init": float(
                architecture.get("alignment_scale_init", 1.0)
            ),
        }
        critic_hidden = int(architecture.get("critic_hidden_dim", 256))
        self.agents = [
            PRRACAgent(
                actor_config=actor_config,
                critic_hidden_dim=critic_hidden,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                noise_sigma=float(noise_sigmas[index]),
            )
            for index in range(4)
        ]
        self.architecture = copy.deepcopy(architecture)
        self.loss_config = copy.deepcopy(loss)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.router_ce_coef = float(loss.get("router_ce_coef", 0.05))
        self.gate_conflict_coef = float(loss.get("gate_conflict_coef", 0.01))
        self.gate_entropy_coef = float(loss.get("gate_entropy_coef", 0.001))
        self.residual_action_reg = float(loss.get("residual_action_reg", 0.01))
        self.device = torch.device("cpu")
        self.niter = 0
        self.last_update: dict[str, Any] = {}

    @property
    def policies(self):
        return [agent.actor for agent in self.agents]

    @property
    def target_policies(self):
        return [agent.target_actor for agent in self.agents]

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        resolved = torch.device("cpu" if device is None else device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device!r} requested but unavailable")
        return resolved

    @staticmethod
    def _move_optimizer(optimizer, device):
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    def _move_to(self, device: torch.device) -> None:
        for agent in self.agents:
            for module in (
                agent.actor,
                agent.target_actor,
                agent.critic1,
                agent.critic2,
                agent.target_critic1,
                agent.target_critic2,
            ):
                module.to(device)
            for optimizer in (
                agent.actor_optimizer,
                agent.critic1_optimizer,
                agent.critic2_optimizer,
            ):
                self._move_optimizer(optimizer, device)
            agent.sync_noise_device()
        self.device = device

    def prep_training(self, device: str | torch.device = "cpu") -> None:
        self._move_to(self._resolve_device(device))
        for agent in self.agents:
            for module in (
                agent.actor,
                agent.target_actor,
                agent.critic1,
                agent.critic2,
                agent.target_critic1,
                agent.target_critic2,
            ):
                module.train()

    def prep_rollouts(self, device: str | torch.device | None = None) -> None:
        if device is not None:
            self._move_to(self._resolve_device(device))
        for agent in self.agents:
            agent.actor.eval()
            agent.target_actor.eval()

    def reset_noise(self) -> None:
        for agent in self.agents:
            agent.reset_noise()

    def scale_noise(self, value: float, multiply: bool = False) -> None:
        for agent in self.agents:
            agent.scale_noise(value, multiply=multiply)

    def _obs(self, observation: Any) -> torch.Tensor:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor

    def step(self, observations, explore: bool = False):
        return [
            agent.step(self._obs(observation), explore=explore)
            for agent, observation in zip(self.agents, observations)
        ]

    step_residual = step

    def _prepare_batch(self, batch) -> dict[str, Any]:
        def tensors(values):
            return [torch.as_tensor(value, dtype=torch.float32, device=self.device) for value in values]

        obs = tensors(batch.obs)
        actions = tensors(batch.actions)
        rewards = tensors(batch.rewards)
        next_obs = tensors(batch.next_obs)
        dones = tensors(batch.dones)
        weights = torch.as_tensor(
            batch.importance_weights, dtype=torch.float32, device=self.device
        ).reshape(-1, 1)
        before = torch.as_tensor(batch.stage_before, dtype=torch.long, device=self.device).reshape(-1)
        after = torch.as_tensor(batch.stage_after, dtype=torch.long, device=self.device).reshape(-1)
        joint = torch.cat((*obs, *actions), dim=1)
        if joint.shape[-1] != self.CRITIC_DIM:
            raise ValueError(f"joint critic input must be 124D, got {joint.shape[-1]}")
        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "dones": dones,
            "weights": weights,
            "stage_before": before,
            "stage_after": after,
            "joint": joint,
        }

    def _target_q(self, agent: PRRACAgent, data: dict[str, Any], agent_i: int):
        with torch.no_grad():
            next_actions = [
                other.target_actor(obs).gated_residual_action
                for other, obs in zip(self.agents, data["next_obs"])
            ]
            target_input = torch.cat((*data["next_obs"], *next_actions), dim=1)
            q1 = gather_stage_values(
                agent.target_critic1(target_input), data["stage_after"]
            )
            q2 = gather_stage_values(
                agent.target_critic2(target_input), data["stage_after"]
            )
            target = data["rewards"][agent_i].reshape(-1, 1) + self.gamma * torch.minimum(
                q1, q2
            ) * (1.0 - data["dones"][agent_i].reshape(-1, 1))
            return torch.clamp(target, -10.0, 10.0)

    @staticmethod
    def _stage_statistics(
        loss_rows: torch.Tensor, td_error: torch.Tensor, stages: torch.Tensor
    ) -> tuple[dict[str, float | None], dict[str, float | None]]:
        names = ("search", "intercept", "hold")
        losses: dict[str, float | None] = {}
        errors: dict[str, float | None] = {}
        for stage, name in enumerate(names):
            mask = stages == stage
            losses[name] = (
                float(loss_rows[mask].mean().detach().item()) if bool(mask.any()) else None
            )
            errors[name] = (
                float(td_error[mask].abs().mean().detach().item()) if bool(mask.any()) else None
            )
        return losses, errors

    def _critic_update(self, batch, agent_i: int):
        data = self._prepare_batch(batch)
        agent = self.agents[int(agent_i)]
        target = self._target_q(agent, data, int(agent_i))
        q1 = gather_stage_values(agent.critic1(data["joint"]), data["stage_before"])
        q2 = gather_stage_values(agent.critic2(data["joint"]), data["stage_before"])
        loss_rows = F.smooth_l1_loss(q1, target, reduction="none") + F.smooth_l1_loss(
            q2, target, reduction="none"
        )
        critic_loss = (loss_rows * data["weights"]).mean()
        agent.critic1_optimizer.zero_grad(set_to_none=True)
        agent.critic2_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic1.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(agent.critic2.parameters(), 0.5)
        agent.critic1_optimizer.step()
        agent.critic2_optimizer.step()
        td_error = (target - torch.minimum(q1, q2)).detach().reshape(-1)
        stage_losses, stage_errors = self._stage_statistics(
            loss_rows.detach().reshape(-1), td_error, data["stage_before"]
        )
        return data, {
            "critic_loss": float(critic_loss.detach().item()),
            "td_error": td_error,
            "stage_critic_losses": stage_losses,
            "stage_td_errors": stage_errors,
        }

    def update_critic_only(self, batch, agent_i: int):
        _, result = self._critic_update(batch, agent_i)
        self.last_update = result
        return result

    def update(self, batch, agent_i: int, parallel: bool = False, logger=None):
        del parallel
        data, result = self._critic_update(batch, agent_i)
        agent_i = int(agent_i)
        agent = self.agents[agent_i]
        agent.actor_optimizer.zero_grad(set_to_none=True)
        actor_output = agent.actor(data["obs"][agent_i])
        policy_actions = []
        for index, (other, observation) in enumerate(zip(self.agents, data["obs"])):
            if index == agent_i:
                policy_actions.append(actor_output.gated_residual_action)
            else:
                with torch.no_grad():
                    policy_actions.append(other.actor(observation).gated_residual_action)
        policy_input = torch.cat((*data["obs"], *policy_actions), dim=1)
        selected_q = gather_stage_values(
            agent.critic1(policy_input), data["stage_before"]
        )
        actor_q_loss = -selected_q.mean()
        router_loss = F.cross_entropy(
            actor_output.router_logits, data["stage_before"]
        )
        residual_reg = actor_output.residual_mix.square().mean()
        conflict = (
            actor_output.trust_gate * F.relu(-actor_output.alignment_cosine)
        ).mean()
        eta = actor_output.trust_gate.clamp(1e-6, 1.0 - 1e-6)
        gate_entropy = (-(eta * eta.log() + (1.0 - eta) * (1.0 - eta).log())).mean()
        actor_loss = (
            actor_q_loss
            + self.residual_action_reg * residual_reg
            + self.router_ce_coef * router_loss
            + self.gate_conflict_coef * conflict
            - self.gate_entropy_coef * gate_entropy
        )
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.2)
        agent.actor_optimizer.step()
        accuracy = (
            actor_output.router_probabilities.argmax(dim=-1) == data["stage_before"]
        ).float().mean()
        result.update(
            {
                "actor_loss": float(actor_loss.detach().item()),
                "actor_q_loss": float(actor_q_loss.detach().item()),
                "router_loss": float(router_loss.detach().item()),
                "router_accuracy": float(accuracy.detach().item()),
                "residual_regularization": float(residual_reg.detach().item()),
                "gate_conflict_loss": float(conflict.detach().item()),
                "gate_entropy": float(gate_entropy.detach().item()),
            }
        )
        if logger is not None:
            logger.add_scalars(
                f"agent{agent_i}/loss",
                {"critic": result["critic_loss"], "actor": result["actor_loss"]},
                self.niter,
            )
        self.last_update = result
        return result

    def update_all_targets(self, compute_diff: bool = False):
        differences = []
        for agent in self.agents:
            for target, source in (
                (agent.target_actor, agent.actor),
                (agent.target_critic1, agent.critic1),
                (agent.target_critic2, agent.critic2),
            ):
                differences.append(
                    soft_update(target, source, self.tau, return_diff=compute_diff)
                )
        self.niter += 1
        return differences if compute_diff else None

    def training_state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "architecture": copy.deepcopy(self.architecture),
            "loss": copy.deepcopy(self.loss_config),
            "gamma": self.gamma,
            "tau": self.tau,
            "niter": int(self.niter),
            "agents": [agent.training_state_dict() for agent in self.agents],
        }

    def load_training_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != self.SCHEMA:
            raise ValueError("unsupported PRRAC algorithm state schema")
        if dict(state.get("architecture", {})) != self.architecture:
            raise ValueError("PRRAC architecture mismatch during resume")
        if dict(state.get("loss", {})) != self.loss_config:
            raise ValueError("PRRAC loss configuration mismatch during resume")
        states = list(state.get("agents", ()))
        if len(states) != 4:
            raise ValueError("PRRAC checkpoint must contain four agents")
        for agent, agent_state in zip(self.agents, states):
            agent.load_training_state_dict(agent_state)
        self.niter = int(state.get("niter", 0))

    def policy_snapshot(self):
        return tuple(
            {key: value.detach().cpu().clone() for key, value in agent.actor.state_dict().items()}
            for agent in self.agents
        )

    def load_policy_snapshot(self, snapshot) -> None:
        states = tuple(snapshot)
        if len(states) != 4:
            raise ValueError("PRRAC policy snapshot must contain four actors")
        for agent, state in zip(self.agents, states):
            agent.actor.load_state_dict(state)

    @classmethod
    def init_from_env(cls, env, **kwargs):
        obs_dims = tuple(env.observation_space[f"agent_{i}"].shape[0] for i in range(4))
        action_dims = tuple(env.action_space[f"agent_{i}"].shape[0] for i in range(4))
        if obs_dims != cls.OBS_DIMS or action_dims != cls.ACTION_DIMS:
            raise ValueError("PRRAC requires the frozen 28D/3D four-agent contract")
        return cls(**kwargs)


__all__ = ("PRRACMADDPG",)
