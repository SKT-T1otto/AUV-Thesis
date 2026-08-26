"""Independent phase-aware prioritized replay for Phase 1C-v2."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import torch

from chapter3_bser.experiments.phase1c_common import (
    Phase1CTransitionMetadata,
    TransitionPhase,
)


def _resolve_device(device: Any) -> torch.device:
    if device is None:
        return torch.device("cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return resolved


class PhaseAwareReplayBuffer:
    """Four-stratum PER with replay-only phase metadata.

    The MADDPG-facing sample tuple remains exactly eight items:
    obs, actions, rewards, next_obs, dones, IS weights, indices, success-tail flags.
    """

    STRATA = ("pre_found", "post_found", "contact_hold", "success_tail")

    def __init__(
        self,
        max_steps: int,
        num_agents: int,
        obs_dims,
        ac_dims,
        *,
        config: Mapping[str, Any] | None = None,
        storage_device: str | torch.device = "cpu",
        generator_seed: int = 2729,
    ) -> None:
        cfg = dict(config or {})
        self.max_steps = int(max_steps)
        self.num_agents = int(num_agents)
        self.obs_dims = tuple(int(value) for value in obs_dims)
        self.ac_dims = tuple(int(value) for value in ac_dims)
        self.storage_device = torch.device(storage_device)
        self.alpha = float(cfg.get("alpha", 0.6))
        self.beta_start = float(cfg.get("beta_start", 0.4))
        self.beta_frames = int(cfg.get("beta_frames", 100_000))
        self.success_tail_steps = int(cfg.get("success_tail_steps", 32))
        self.rare_stratum_with_replacement = bool(
            cfg.get(
                "rare_stratum_with_replacement",
                cfg.get("rare_stratum_replacement", True),
            )
        )
        self.success_priority_multiplier = float(
            cfg.get(
                "success_priority_multiplier",
                cfg.get("success_priority", 1.0),
            )
        )
        fractions = {
            "pre_found": float(cfg.get("pre_found_fraction", 0.40)),
            "post_found": float(cfg.get("post_found_fraction", 0.30)),
            "contact_hold": float(cfg.get("contact_hold_fraction", 0.20)),
            "success_tail": float(cfg.get("success_tail_fraction", 0.10)),
        }
        if any(value < 0.0 for value in fractions.values()):
            raise ValueError("phase-aware replay fractions must be non-negative")
        total = sum(fractions.values())
        if total <= 0.0:
            raise ValueError("phase-aware replay fractions must have positive sum")
        self.fractions = {key: value / total for key, value in fractions.items()}

        def alloc(shape, *, dtype=torch.float32):
            return torch.empty(shape, dtype=dtype, device=self.storage_device)

        self.obs_buffs = [alloc((self.max_steps, dim)) for dim in self.obs_dims]
        self.ac_buffs = [alloc((self.max_steps, dim)) for dim in self.ac_dims]
        self.rew_buffs = [alloc((self.max_steps,)) for _ in range(self.num_agents)]
        self.next_obs_buffs = [alloc((self.max_steps, dim)) for dim in self.obs_dims]
        self.done_buffs = [alloc((self.max_steps,)) for _ in range(self.num_agents)]
        self.priorities = alloc((self.max_steps,))
        self.priorities.fill_(1.0)
        self.episode_ids = alloc((self.max_steps,), dtype=torch.int64)
        self.episode_indices = alloc((self.max_steps,), dtype=torch.int64)
        self.steps = alloc((self.max_steps,), dtype=torch.int64)
        self.phase_codes = alloc((self.max_steps,), dtype=torch.int64)
        self.terminal_success_flags = alloc((self.max_steps,), dtype=torch.bool)
        self.success_tail_flags = alloc((self.max_steps,), dtype=torch.bool)
        self.insertion_order = alloc((self.max_steps,), dtype=torch.int64)
        self.episode_ids.fill_(-1)
        self.episode_indices.fill_(-1)
        self.steps.zero_()
        self.phase_codes.fill_(int(TransitionPhase.PRE_FOUND))
        self.terminal_success_flags.zero_()
        self.success_tail_flags.zero_()
        self.insertion_order.fill_(-1)

        self.frame = 1.0
        self.filled_i = 0
        self.next_idx = 0
        self.total_push_count = 0
        self.success_tail_mark_count = 0
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(generator_seed))
        self.last_sample_diagnostics: dict[str, Any] = {}

    def _tensor(self, value: Any, *, dtype=torch.float32) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.detach().to(self.storage_device, dtype=dtype).reshape(-1)
        return torch.as_tensor(value, dtype=dtype, device=self.storage_device).reshape(-1)

    def push(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        success_flags,
        metadata: Phase1CTransitionMetadata,
    ) -> int:
        if not isinstance(metadata, Phase1CTransitionMetadata):
            raise TypeError("phase-aware replay requires Phase1CTransitionMetadata")
        idx = int(self.next_idx)
        rewards_t = self._tensor(rewards)
        if rewards_t.numel() != self.num_agents:
            raise ValueError("reward vector does not match replay agent count")
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][idx].copy_(self._tensor(obs[agent_i]))
            self.ac_buffs[agent_i][idx].copy_(self._tensor(actions[agent_i]))
            self.rew_buffs[agent_i][idx] = rewards_t[agent_i]
            self.next_obs_buffs[agent_i][idx].copy_(self._tensor(next_obs[agent_i]))
            self.done_buffs[agent_i][idx] = float(dones[agent_i])
        self.episode_ids[idx] = int(metadata.episode_id)
        self.episode_indices[idx] = int(metadata.episode_index)
        self.steps[idx] = int(metadata.step)
        self.phase_codes[idx] = int(metadata.phase)
        terminal = bool(metadata.mission_complete or any(bool(v) for v in success_flags))
        self.terminal_success_flags[idx] = terminal
        self.success_tail_flags[idx] = bool(metadata.phase == TransitionPhase.SUCCESS)
        self.insertion_order[idx] = int(self.total_push_count)
        current_max = (
            float(self.priorities[: self.filled_i].max().item())
            if self.filled_i > 0
            else 1.0
        )
        self.priorities[idx] = max(1e-5, current_max)
        self.next_idx = (idx + 1) % self.max_steps
        self.filled_i = min(self.filled_i + 1, self.max_steps)
        self.total_push_count += 1
        return idx

    def _valid_indices(self) -> torch.Tensor:
        if self.filled_i == 0:
            return torch.empty(0, dtype=torch.long, device=self.storage_device)
        if self.filled_i < self.max_steps:
            return torch.arange(self.filled_i, device=self.storage_device)
        return torch.arange(self.max_steps, device=self.storage_device)

    def finalize_episode(self, episode_id: int, *, success: bool) -> int:
        if not success or self.filled_i == 0:
            return 0
        valid = self._valid_indices()
        mask = self.episode_ids[valid] == int(episode_id)
        candidates = valid[mask]
        if candidates.numel() == 0:
            return 0
        order = self.insertion_order[candidates]
        sorted_candidates = candidates[torch.argsort(order)]
        selected = sorted_candidates[-min(self.success_tail_steps, sorted_candidates.numel()) :]
        newly_marked = ~self.success_tail_flags[selected]
        marked_count = int(newly_marked.sum().item())
        self.success_tail_flags[selected] = True
        if self.success_priority_multiplier != 1.0:
            self.priorities[selected] = torch.clamp(
                self.priorities[selected] * self.success_priority_multiplier,
                min=1e-5,
                max=100.0,
            )
        self.success_tail_mark_count += marked_count
        return int(selected.numel())

    def _stratum_indices(self) -> dict[str, torch.Tensor]:
        valid = self._valid_indices()
        if valid.numel() == 0:
            return {name: valid.clone() for name in self.STRATA}
        tail = self.success_tail_flags[valid]
        phases = self.phase_codes[valid]
        return {
            "success_tail": valid[tail | (phases == int(TransitionPhase.SUCCESS))],
            "pre_found": valid[(~tail) & (phases == int(TransitionPhase.PRE_FOUND))],
            "post_found": valid[(~tail) & (phases == int(TransitionPhase.POST_FOUND))],
            "contact_hold": valid[
                (~tail)
                & (
                    (phases == int(TransitionPhase.CONTACT))
                    | (phases == int(TransitionPhase.HOLD))
                )
            ],
        }

    @staticmethod
    def _largest_remainder_counts(
        total: int,
        weights: Mapping[str, float],
        order: tuple[str, ...],
    ) -> dict[str, int]:
        if total <= 0:
            return {name: 0 for name in order}
        weight_sum = sum(float(weights.get(name, 0.0)) for name in order)
        if weight_sum <= 0.0:
            counts = {name: 0 for name in order}
            counts[order[0]] = int(total)
            return counts
        raw = {
            name: total * float(weights.get(name, 0.0)) / weight_sum
            for name in order
        }
        counts = {name: int(raw[name] // 1) for name in order}
        remainder = int(total - sum(counts.values()))
        ranked = sorted(
            order,
            key=lambda name: (-(raw[name] - counts[name]), order.index(name)),
        )
        for name in ranked[:remainder]:
            counts[name] += 1
        return counts

    def _requested_counts(self, batch_size: int) -> dict[str, int]:
        return self._largest_remainder_counts(batch_size, self.fractions, self.STRATA)

    def _effective_counts(
        self,
        batch_size: int,
        strata: Mapping[str, torch.Tensor],
    ) -> tuple[dict[str, int], dict[str, Any]]:
        requested = self._requested_counts(batch_size)
        nonempty = [name for name in self.STRATA if strata[name].numel() > 0]
        if not nonempty:
            raise ValueError("phase-aware replay is empty")
        empty = [name for name in self.STRATA if name not in nonempty]
        effective = dict(requested)
        redistributed = sum(effective.pop(name, 0) for name in empty)
        for name in empty:
            effective[name] = 0
        if redistributed:
            allocation = self._largest_remainder_counts(
                redistributed,
                {name: self.fractions[name] for name in nonempty},
                tuple(nonempty),
            )
            for name, count in allocation.items():
                effective[name] += count

        # If replacement is disabled, shift shortages to strata with spare rows.
        shortage_log: dict[str, int] = {}
        if not self.rare_stratum_with_replacement:
            shortage = 0
            for name in self.STRATA:
                available = int(strata[name].numel())
                if effective[name] > available:
                    missing = effective[name] - available
                    shortage_log[name] = missing
                    effective[name] = available
                    shortage += missing
            while shortage > 0:
                candidates = [
                    name
                    for name in self.STRATA
                    if int(strata[name].numel()) > effective[name]
                ]
                if not candidates:
                    break
                allocation = self._largest_remainder_counts(
                    shortage,
                    {name: self.fractions[name] for name in candidates},
                    tuple(candidates),
                )
                moved = 0
                for name, requested_extra in allocation.items():
                    capacity = int(strata[name].numel()) - effective[name]
                    take = min(capacity, requested_extra)
                    effective[name] += take
                    moved += take
                if moved == 0:
                    break
                shortage -= moved
            if shortage:
                # The requested batch is <= replay size, so this can only occur
                # after pathological metadata loss. Fill deterministically from
                # any valid stratum rather than changing batch size.
                for name in nonempty:
                    if shortage <= 0:
                        break
                    capacity = int(strata[name].numel()) - effective[name]
                    take = min(shortage, max(0, capacity))
                    effective[name] += take
                    shortage -= take
        fallback = {
            "empty_strata": empty,
            "redistributed_count": int(redistributed),
            "shortage_without_replacement": shortage_log,
        }
        return effective, fallback

    def _beta(self) -> float:
        return min(
            1.0,
            self.beta_start
            + (1.0 - self.beta_start) * float(self.frame) / max(1, self.beta_frames),
        )

    def _sample_one_stratum(
        self,
        candidates: torch.Tensor,
        count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if count <= 0:
            empty = torch.empty(0, dtype=torch.long, device=self.storage_device)
            return empty, torch.empty(0, dtype=torch.float32, device=self.storage_device), 0
        priorities = torch.clamp(self.priorities[candidates], min=1e-5).pow(self.alpha)
        conditional = priorities / priorities.sum().clamp_min(1e-12)
        replacement = bool(count > candidates.numel())
        if replacement and not self.rare_stratum_with_replacement:
            raise RuntimeError("internal replay quota exceeds stratum without replacement")
        # torch.multinomial's generator must live on CPU. Storage is CPU in the
        # formal v2 config; keep a deterministic fallback for any other device.
        probs_cpu = conditional.detach().cpu()
        sampled_local_cpu = torch.multinomial(
            probs_cpu,
            count,
            replacement=replacement,
            generator=self.generator,
        )
        sampled_local = sampled_local_cpu.to(candidates.device)
        sampled = candidates[sampled_local]
        sampled_conditional = conditional[sampled_local]
        replacement_count = max(0, int(count - candidates.numel())) if replacement else 0
        return sampled, sampled_conditional, replacement_count

    def sample(
        self,
        n: int,
        to_gpu: bool = False,
        norm_rews: bool = True,
        device: str | torch.device | None = None,
    ):
        if self.filled_i == 0:
            raise ValueError("PhaseAwareReplayBuffer is empty")
        batch_size = min(int(n), int(self.filled_i))
        if batch_size <= 0:
            raise ValueError("sample size must be positive")
        target_device = _resolve_device(
            "cuda" if to_gpu else (device if device is not None else self.storage_device)
        )
        strata = self._stratum_indices()
        requested = self._requested_counts(batch_size)
        effective, fallback = self._effective_counts(batch_size, strata)

        sampled_parts: list[torch.Tensor] = []
        probability_parts: list[torch.Tensor] = []
        replacement_counts: dict[str, int] = {}
        actual_counts: dict[str, int] = {}
        for name in self.STRATA:
            count = int(effective[name])
            sampled, conditional, replacement_count = self._sample_one_stratum(
                strata[name], count
            )
            q = count / float(batch_size) if count else 0.0
            sampled_parts.append(sampled)
            probability_parts.append(conditional * q)
            replacement_counts[name] = int(replacement_count)
            actual_counts[name] = int(sampled.numel())

        indices_storage = torch.cat(sampled_parts, dim=0)
        probabilities = torch.cat(probability_parts, dim=0)
        if indices_storage.numel() != batch_size:
            raise RuntimeError(
                f"phase-aware replay produced {indices_storage.numel()} rows, "
                f"expected {batch_size}"
            )
        # Shuffle the concatenated strata so optimizers do not receive phase blocks.
        permutation = torch.randperm(
            batch_size, generator=self.generator, device="cpu"
        ).to(indices_storage.device)
        indices_storage = indices_storage[permutation]
        probabilities = probabilities[permutation]

        beta = self._beta()
        self.frame += 1.0
        weights = (self.filled_i * probabilities.clamp_min(1e-12)).pow(-beta)
        weights = weights / weights.max().clamp_min(1e-12)
        if not torch.isfinite(weights).all():
            raise RuntimeError("phase-aware replay produced non-finite IS weights")

        def move(tensor: torch.Tensor) -> torch.Tensor:
            return tensor if tensor.device == target_device else tensor.to(target_device)

        if norm_rews:
            rewards = []
            valid = self._valid_indices()
            for agent_i in range(self.num_agents):
                source = self.rew_buffs[agent_i][valid]
                values = self.rew_buffs[agent_i][indices_storage]
                normalized = (values - source.mean()) / source.std(
                    unbiased=False
                ).clamp_min(1e-6)
                rewards.append(move(normalized))
        else:
            rewards = [
                move(self.rew_buffs[i][indices_storage])
                for i in range(self.num_agents)
            ]

        self.last_sample_diagnostics = {
            "requested_counts": requested,
            "actual_counts": actual_counts,
            "fallback": fallback,
            "replacement_counts": replacement_counts,
            "effective_fractions": {
                name: actual_counts[name] / float(batch_size) for name in self.STRATA
            },
            "beta": float(beta),
            "batch_size": int(batch_size),
        }
        return (
            [move(self.obs_buffs[i][indices_storage]) for i in range(self.num_agents)],
            [move(self.ac_buffs[i][indices_storage]) for i in range(self.num_agents)],
            rewards,
            [
                move(self.next_obs_buffs[i][indices_storage])
                for i in range(self.num_agents)
            ],
            [move(self.done_buffs[i][indices_storage]) for i in range(self.num_agents)],
            move(weights),
            move(indices_storage),
            move(self.success_tail_flags[indices_storage]),
        )

    def update_priorities(
        self,
        indices,
        td_errors,
        success_flags=None,
        eps: float = 1e-5,
    ) -> None:
        if indices is None or td_errors is None or self.filled_i == 0:
            return
        index_tensor = self._tensor(indices, dtype=torch.long)
        error_tensor = torch.nan_to_num(
            self._tensor(td_errors).abs(), nan=0.0, posinf=100.0, neginf=0.0
        )
        n = min(index_tensor.numel(), error_tensor.numel())
        if n == 0:
            return
        index_tensor = index_tensor[:n].clamp(0, self.max_steps - 1)
        error_tensor = error_tensor[:n]
        grouped: dict[int, float] = defaultdict(float)
        for idx, error in zip(index_tensor.tolist(), error_tensor.tolist()):
            grouped[int(idx)] = max(grouped[int(idx)], float(error))
        for idx, error in grouped.items():
            priority = float(error) + float(eps)
            if bool(self.success_tail_flags[idx]) and self.success_priority_multiplier != 1.0:
                priority *= self.success_priority_multiplier
            self.priorities[idx] = min(100.0, max(float(eps), priority))

    def phase_counts(self) -> dict[str, int]:
        strata = self._stratum_indices()
        return {name: int(strata[name].numel()) for name in self.STRATA}

    def state_dict(self) -> dict[str, Any]:
        """Serialize the complete logical ring state needed for deterministic resume."""

        return {
            "schema": "bser.phase1c.phase_aware_replay.state.v1",
            "max_steps": self.max_steps,
            "num_agents": self.num_agents,
            "obs_dims": self.obs_dims,
            "ac_dims": self.ac_dims,
            "alpha": self.alpha,
            "beta_start": self.beta_start,
            "beta_frames": self.beta_frames,
            "success_tail_steps": self.success_tail_steps,
            "rare_stratum_with_replacement": self.rare_stratum_with_replacement,
            "success_priority_multiplier": self.success_priority_multiplier,
            "fractions": dict(self.fractions),
            "frame": float(self.frame),
            "filled_i": int(self.filled_i),
            "next_idx": int(self.next_idx),
            "total_push_count": int(self.total_push_count),
            "success_tail_mark_count": int(self.success_tail_mark_count),
            "obs_buffs": [item.detach().cpu().clone() for item in self.obs_buffs],
            "ac_buffs": [item.detach().cpu().clone() for item in self.ac_buffs],
            "rew_buffs": [item.detach().cpu().clone() for item in self.rew_buffs],
            "next_obs_buffs": [
                item.detach().cpu().clone() for item in self.next_obs_buffs
            ],
            "done_buffs": [item.detach().cpu().clone() for item in self.done_buffs],
            "priorities": self.priorities.detach().cpu().clone(),
            "episode_ids": self.episode_ids.detach().cpu().clone(),
            "episode_indices": self.episode_indices.detach().cpu().clone(),
            "steps": self.steps.detach().cpu().clone(),
            "phase_codes": self.phase_codes.detach().cpu().clone(),
            "terminal_success_flags": self.terminal_success_flags.detach().cpu().clone(),
            "success_tail_flags": self.success_tail_flags.detach().cpu().clone(),
            "insertion_order": self.insertion_order.detach().cpu().clone(),
            "generator_state": self.generator.get_state().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") not in {
            None,
            "bser.phase1c.phase_aware_replay.state.v1",
        }:
            raise ValueError("unsupported phase-aware replay state schema")
        for key, expected in (
            ("num_agents", self.num_agents),
            ("obs_dims", self.obs_dims),
            ("ac_dims", self.ac_dims),
        ):
            loaded = state.get(key)
            if key in {"obs_dims", "ac_dims"}:
                loaded = tuple(loaded)
            if loaded != expected:
                raise ValueError(
                    f"phase-aware replay {key} mismatch: expected {expected}, got {loaded}"
                )
        if int(state.get("max_steps", self.max_steps)) != self.max_steps:
            raise ValueError("phase-aware replay capacity mismatch")

        for field in (
            "obs_buffs",
            "ac_buffs",
            "rew_buffs",
            "next_obs_buffs",
            "done_buffs",
        ):
            targets = getattr(self, field)
            sources = state[field]
            if len(targets) != len(sources):
                raise ValueError(f"phase-aware replay {field} agent count mismatch")
            for target, source in zip(targets, sources):
                if tuple(target.shape) != tuple(source.shape):
                    raise ValueError(f"phase-aware replay {field} shape mismatch")
                target.copy_(source.to(target.device, dtype=target.dtype))
        for field in (
            "priorities",
            "episode_ids",
            "episode_indices",
            "steps",
            "phase_codes",
            "terminal_success_flags",
            "success_tail_flags",
            "insertion_order",
        ):
            target = getattr(self, field)
            source = state[field]
            if tuple(target.shape) != tuple(source.shape):
                raise ValueError(f"phase-aware replay {field} shape mismatch")
            target.copy_(source.to(target.device, dtype=target.dtype))
        self.frame = float(state["frame"])
        self.filled_i = int(state["filled_i"])
        self.next_idx = int(state["next_idx"])
        self.total_push_count = int(state.get("total_push_count", self.filled_i))
        self.success_tail_mark_count = int(state.get("success_tail_mark_count", 0))
        if not 0 <= self.filled_i <= self.max_steps:
            raise ValueError("invalid phase-aware replay filled size")
        if not 0 <= self.next_idx < self.max_steps:
            raise ValueError("invalid phase-aware replay next index")
        if "generator_state" in state:
            self.generator.set_state(state["generator_state"].detach().cpu())

    def __len__(self) -> int:
        return int(self.filled_i)


# Descriptive alias retained for the design document / thesis terminology.
PhaseAwarePhase1CReplayBuffer = PhaseAwareReplayBuffer

__all__ = ("PhaseAwareReplayBuffer", "PhaseAwarePhase1CReplayBuffer")
