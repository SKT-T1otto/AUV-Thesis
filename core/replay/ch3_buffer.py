"""Replay storage for pure Chapter-3 training.

The buffer intentionally contains no communication graph, semantic message,
belief-fusion, quarantine, scenario-risk, or tail-training metadata.
"""

from __future__ import annotations

import torch
import math

def _resolve_device(device):
    if device is None:
        return torch.device("cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return resolved


class CH3ReplayBuffer:
    def __init__(
        self,
        max_steps,
        num_agents,
        obs_dims,
        ac_dims,
        *,
        success_priority=2.0,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=100_000,
        storage_device="cpu",
    ):
        self.max_steps = int(max_steps)
        self.num_agents = int(num_agents)
        self.obs_dims = tuple(int(x) for x in obs_dims)
        self.ac_dims = tuple(int(x) for x in ac_dims)
        self.success_priority = float(success_priority)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_frames = int(beta_frames)
        self.storage_device = torch.device(storage_device)
        self.frame = 1.0
        self.filled_i = 0
        self.next_idx = 0

        def alloc(shape, dtype=torch.float32):
            return torch.empty(shape, dtype=dtype, device=self.storage_device)

        self.obs_buffs = [alloc((self.max_steps, dim)) for dim in self.obs_dims]
        self.ac_buffs = [alloc((self.max_steps, dim)) for dim in self.ac_dims]
        self.rew_buffs = [alloc((self.max_steps,)) for _ in range(self.num_agents)]
        self.next_obs_buffs = [alloc((self.max_steps, dim)) for dim in self.obs_dims]
        self.done_buffs = [alloc((self.max_steps,)) for _ in range(self.num_agents)]
        self.success_buffs = alloc((self.max_steps,), dtype=torch.bool)
        self.success_buffs.zero_()
        self.priorities = alloc((self.max_steps,))
        self.priorities.fill_(1.0)

    def _tensor(self, value, *, dtype=torch.float32):
        if torch.is_tensor(value):
            return value.detach().to(self.storage_device, dtype=dtype).reshape(-1)
        return torch.as_tensor(value, dtype=dtype, device=self.storage_device).reshape(-1)

    def push(self, obs, actions, rewards, next_obs, dones, success_flags):
        idx = self.next_idx
        success = bool(any(success_flags))
        rewards_t = self._tensor(rewards)
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][idx].copy_(self._tensor(obs[agent_i]))
            self.ac_buffs[agent_i][idx].copy_(self._tensor(actions[agent_i]))
            self.rew_buffs[agent_i][idx] = rewards_t[agent_i]
            self.next_obs_buffs[agent_i][idx].copy_(self._tensor(next_obs[agent_i]))
            self.done_buffs[agent_i][idx] = float(dones[agent_i])
        self.success_buffs[idx] = success
        self.priorities[idx] = self.success_priority if success else 1.0
        self.next_idx = (idx + 1) % self.max_steps
        self.filled_i = min(self.filled_i + 1, self.max_steps)

    def _beta(self):
        return min(
            1.0,
            self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames,
        )

    def sample(self, n, to_gpu=False, norm_rews=True, device=None):
        if self.filled_i == 0:
            raise ValueError("CH3ReplayBuffer is empty")
        n = min(int(n), self.filled_i)
        requested_device = (
            "cuda" if to_gpu else (device if device is not None else self.storage_device)
        )
        target_device = _resolve_device(requested_device)
        probabilities = self.priorities[: self.filled_i].pow(self.alpha)
        probabilities = probabilities / probabilities.sum().clamp_min(1e-8)
        indices_storage = torch.multinomial(probabilities, n, replacement=False)
        self.frame += 1.0 / max(1, self.num_agents)
        weights = (self.filled_i * probabilities[indices_storage]).pow(-self._beta())
        weights = weights / weights.max().clamp_min(1e-8)

        def move(tensor):
            return tensor if tensor.device == target_device else tensor.to(target_device)

        if norm_rews:
            rewards = []
            for agent_i in range(self.num_agents):
                source = self.rew_buffs[agent_i][: self.filled_i]
                normalized = (
                    self.rew_buffs[agent_i][indices_storage] - source.mean()
                ) / source.std(unbiased=False).clamp_min(1e-6)
                rewards.append(move(normalized))
        else:
            rewards = [
                move(self.rew_buffs[i][indices_storage]) for i in range(self.num_agents)
            ]

        return (
            [move(self.obs_buffs[i][indices_storage]) for i in range(self.num_agents)],
            [move(self.ac_buffs[i][indices_storage]) for i in range(self.num_agents)],
            rewards,
            [move(self.next_obs_buffs[i][indices_storage]) for i in range(self.num_agents)],
            [move(self.done_buffs[i][indices_storage]) for i in range(self.num_agents)],
            move(weights),
            move(indices_storage),
            move(self.success_buffs[indices_storage]),
        )

    def update_priorities(self, indices, td_errors, success_flags=None, eps=1e-5):
        if indices is None or td_errors is None:
            return
        indices = self._tensor(indices, dtype=torch.long)
        errors = torch.nan_to_num(
            self._tensor(td_errors).abs(), nan=0.0, posinf=100.0, neginf=0.0
        )
        n = min(indices.numel(), errors.numel())
        if n == 0:
            return
        indices = indices[:n].clamp(0, self.filled_i - 1)
        priorities = errors[:n] + float(eps)
        if success_flags is not None:
            success = self._tensor(success_flags, dtype=torch.bool)[:n]
            priorities *= torch.where(
                success,
                torch.full_like(priorities, self.success_priority),
                torch.ones_like(priorities),
            )
        self.priorities[indices] = priorities.clamp(float(eps), 100.0)

    def state_dict(self):
        """Serialize only valid transitions so resume files do not mirror capacity."""
        used = int(self.filled_i)
        return {
            "max_steps": self.max_steps,
            "num_agents": self.num_agents,
            "obs_dims": self.obs_dims,
            "ac_dims": self.ac_dims,
            "success_priority": self.success_priority,
            "alpha": self.alpha,
            "beta_start": self.beta_start,
            "beta_frames": self.beta_frames,
            "frame": float(self.frame),
            "filled_i": used,
            "next_idx": int(self.next_idx),
            "obs_buffs": [item[:used].detach().cpu().clone() for item in self.obs_buffs],
            "ac_buffs": [item[:used].detach().cpu().clone() for item in self.ac_buffs],
            "rew_buffs": [item[:used].detach().cpu().clone() for item in self.rew_buffs],
            "next_obs_buffs": [item[:used].detach().cpu().clone() for item in self.next_obs_buffs],
            "done_buffs": [item[:used].detach().cpu().clone() for item in self.done_buffs],
            "success_buffs": self.success_buffs[:used].detach().cpu().clone(),
            "priorities": self.priorities[:used].detach().cpu().clone(),
        }

    def load_state_dict(self, state):
        """Restore the same-capacity state only, validating before any mutation."""
        if not isinstance(state, dict):
            raise ValueError("replay state must be a mapping")
        expected = {
            "max_steps": self.max_steps,
            "num_agents": self.num_agents,
            "obs_dims": self.obs_dims,
            "ac_dims": self.ac_dims,
            "success_priority": self.success_priority,
            "alpha": self.alpha,
            "beta_start": self.beta_start,
            "beta_frames": self.beta_frames,
        }
        for key, value in expected.items():
            loaded = state.get(key)
            if key in {"obs_dims", "ac_dims"}:
                if not isinstance(loaded, (tuple, list)):
                    raise ValueError(f"invalid replay {key}")
                loaded = tuple(loaded)
            if key in {"max_steps", "num_agents", "beta_frames"} and type(loaded) is not int:
                raise ValueError(f"replay {key} must be an integer")
            if loaded != value:
                raise ValueError(f"replay buffer {key} mismatch: expected {value}, got {loaded}")
        used, next_idx = state["filled_i"], state["next_idx"]
        if type(used) is not int or not 0 <= used <= self.max_steps:
            raise ValueError(
                f"replay buffer filled_i={used} exceeds configured capacity={self.max_steps}"
            )
        if (type(next_idx) is not int or not 0 <= next_idx < self.max_steps
                or (used < self.max_steps and next_idx != used)):
            raise ValueError("invalid replay ring next_idx/filled_i relationship")
        frame = state["frame"]
        if isinstance(frame, bool) or not isinstance(frame, (int, float)) or not math.isfinite(frame) or frame < 1:
            raise ValueError("invalid replay sampling frame")
        fields = (
            "obs_buffs", "ac_buffs", "rew_buffs", "next_obs_buffs", "done_buffs"
        )
        staged = {}
        for field in fields:
            targets = getattr(self, field)
            sources = state[field]
            if len(targets) != len(sources):
                raise ValueError(f"replay buffer {field} agent count mismatch")
            prepared = []
            for target, source in zip(targets, sources):
                if not torch.is_tensor(source) or tuple(source.shape) != tuple(target[:used].shape):
                    raise ValueError(f"invalid replay {field} tensor shape")
                if not bool(torch.isfinite(source).all()):
                    raise ValueError(f"nonfinite replay {field}")
                if field == "done_buffs" and not bool(((source == 0) | (source == 1)).all()):
                    raise ValueError("invalid replay done flags")
                converted = source.detach().to(target.device, dtype=target.dtype).clone()
                if not bool(torch.isfinite(converted).all()):
                    raise ValueError(f"replay {field} overflows storage dtype")
                prepared.append(converted)
            staged[field] = prepared
        for field in ("success_buffs", "priorities"):
            source, target = state[field], getattr(self, field)
            if not torch.is_tensor(source) or source.shape != (used,) or not bool(torch.isfinite(source).all()):
                raise ValueError(f"invalid replay {field}")
            if field == "success_buffs" and not bool(((source == 0) | (source == 1)).all()):
                raise ValueError("invalid replay success flags")
            if field == "priorities" and not bool((source > 0).all()):
                raise ValueError("replay priorities must be positive")
            staged[field] = source.detach().to(target.device, dtype=target.dtype).clone()
            if not bool(torch.isfinite(staged[field]).all()):
                raise ValueError(f"replay {field} overflows storage dtype")
        # All validation and device/dtype conversions have completed.
        for field in fields:
            for target, source in zip(getattr(self, field), staged[field]):
                target[:used].copy_(source)
        self.success_buffs[:used].copy_(staged["success_buffs"])
        self.priorities[:used].copy_(staged["priorities"])
        self.filled_i = used
        self.next_idx = next_idx
        self.frame = float(frame)

    def __len__(self):
        return self.filled_i
