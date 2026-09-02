"""Composition adapter adding PRRAC stage pairs to frozen v2 replay."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.phase_aware_replay import (
    PhaseAwareReplayBuffer,
)
from chapter3_bser.experiments.phase1c_prrac import REPLAY_SCHEMA
from chapter3_bser.experiments.phase1c_prrac.transition_protocol import (
    PRRACTransitionMetadata,
)
from chapter3_bser.models.search_value_head import (
    SEARCHER_COUNT,
    SEARCH_FEATURE_DIM,
    resolve_search_value_config,
)


@dataclass
class PRRACBatch:
    obs: list[torch.Tensor]
    actions: list[torch.Tensor]
    rewards: list[torch.Tensor]
    next_obs: list[torch.Tensor]
    dones: list[torch.Tensor]
    importance_weights: torch.Tensor
    indices: torch.Tensor
    success_tail_flags: torch.Tensor
    stage_before: torch.Tensor
    stage_after: torch.Tensor
    search_features: torch.Tensor | None = None
    future_found: torch.Tensor | None = None
    search_value_valid: torch.Tensor | None = None

    def as_base_tuple(self):
        return (
            self.obs,
            self.actions,
            self.rewards,
            self.next_obs,
            self.dones,
            self.importance_weights,
            self.indices,
            self.success_tail_flags,
        )


class PRRACReplayAdapter:
    def __init__(
        self,
        max_steps: int,
        num_agents: int = 4,
        obs_dims=(28, 28, 28, 28),
        ac_dims=(3, 3, 3, 3),
        *,
        config: Mapping[str, Any] | None = None,
        storage_device: str | torch.device = "cpu",
        generator_seed: int = 2729,
        base_replay: PhaseAwareReplayBuffer | None = None,
        search_value_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.search_value_config = resolve_search_value_config(search_value_config)
        self.search_value_enabled = bool(self.search_value_config["enabled"])
        self.base = base_replay or PhaseAwareReplayBuffer(
            max_steps=max_steps,
            num_agents=num_agents,
            obs_dims=obs_dims,
            ac_dims=ac_dims,
            config=config,
            storage_device=storage_device,
            generator_seed=generator_seed,
        )
        self.stage_before = torch.full(
            (self.base.max_steps,), -1, dtype=torch.int64, device=self.base.storage_device
        )
        self.stage_after = torch.full_like(self.stage_before, -1)
        self.search_features = None
        self.future_found = None
        self.search_value_valid = None
        if self.search_value_enabled:
            self.search_features = torch.zeros(
                (self.base.max_steps, SEARCHER_COUNT, SEARCH_FEATURE_DIM),
                dtype=torch.float32,
                device=self.base.storage_device,
            )
            self.future_found = torch.zeros(
                (self.base.max_steps, SEARCHER_COUNT, 1),
                dtype=torch.float32,
                device=self.base.storage_device,
            )
            self.search_value_valid = torch.zeros(
                self.base.max_steps,
                dtype=torch.bool,
                device=self.base.storage_device,
            )

    @property
    def last_sample_diagnostics(self):
        return self.base.last_sample_diagnostics

    @property
    def STRATA(self):
        return self.base.STRATA

    def push(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        success_flags,
        metadata: PRRACTransitionMetadata,
        *,
        search_features=None,
        future_found=None,
    ) -> int:
        if not isinstance(metadata, PRRACTransitionMetadata):
            raise TypeError("PRRAC replay requires PRRACTransitionMetadata")
        index = self.base.push(
            obs,
            actions,
            rewards,
            next_obs,
            dones,
            success_flags,
            metadata.base,
        )
        self.stage_before[index] = int(metadata.stage_before)
        self.stage_after[index] = int(metadata.stage_after)
        if self.search_value_enabled:
            self.search_value_valid[index] = False
            if search_features is not None and future_found is not None:
                features = torch.as_tensor(search_features, dtype=torch.float32)
                labels = torch.as_tensor(future_found, dtype=torch.float32)
                expected_features = (SEARCHER_COUNT, SEARCH_FEATURE_DIM)
                expected_labels = (SEARCHER_COUNT, 1)
                if tuple(features.shape) != expected_features:
                    raise ValueError(
                        f"search features must have shape {expected_features}"
                    )
                if tuple(labels.shape) != expected_labels:
                    raise ValueError(
                        f"future-found labels must have shape {expected_labels}"
                    )
                if not bool(torch.isfinite(features).all()):
                    raise ValueError("search features must be finite")
                if not bool(torch.all((labels == 0.0) | (labels == 1.0))):
                    raise ValueError("future-found labels must be binary")
                self.search_features[index].copy_(
                    features.to(self.search_features.device)
                )
                self.future_found[index].copy_(labels.to(self.future_found.device))
                self.search_value_valid[index] = True
        return index

    def sample(self, n: int, **kwargs) -> PRRACBatch:
        base = self.base.sample(n, **kwargs)
        indices_cpu = base[6].detach().cpu().long()
        target_device = base[6].device
        before = self.stage_before.detach().cpu()[indices_cpu].to(target_device)
        after = self.stage_after.detach().cpu()[indices_cpu].to(target_device)
        if bool(torch.any(before < 0)) or bool(torch.any(after < 0)):
            raise RuntimeError("sampled PRRAC replay row has missing stage metadata")
        search_features = None
        future_found = None
        search_value_valid = None
        if self.search_value_enabled:
            search_features = self.search_features.detach().cpu()[indices_cpu].to(
                target_device
            )
            future_found = self.future_found.detach().cpu()[indices_cpu].to(
                target_device
            )
            search_value_valid = self.search_value_valid.detach().cpu()[
                indices_cpu
            ].to(target_device)
        return PRRACBatch(
            obs=base[0],
            actions=base[1],
            rewards=base[2],
            next_obs=base[3],
            dones=base[4],
            importance_weights=base[5],
            indices=base[6],
            success_tail_flags=base[7],
            stage_before=before,
            stage_after=after,
            search_features=search_features,
            future_found=future_found,
            search_value_valid=search_value_valid,
        )

    def update_priorities(self, *args, **kwargs) -> None:
        self.base.update_priorities(*args, **kwargs)

    def finalize_episode(self, *args, **kwargs) -> int:
        return self.base.finalize_episode(*args, **kwargs)

    def phase_counts(self):
        return self.base.phase_counts()

    def stage_counts(self) -> dict[str, int]:
        valid = self.base._valid_indices()
        values = self.stage_before[valid]
        return {
            "search": int((values == 0).sum().item()),
            "intercept": int((values == 1).sum().item()),
            "hold": int((values == 2).sum().item()),
        }

    def state_dict(self) -> dict[str, Any]:
        state = {
            "schema": REPLAY_SCHEMA,
            "base_replay": self.base.state_dict(),
            "stage_before": self.stage_before.detach().cpu().clone(),
            "stage_after": self.stage_after.detach().cpu().clone(),
        }
        if self.search_value_enabled:
            state.update(
                search_value=copy.deepcopy(self.search_value_config),
                search_features=self.search_features.detach().cpu().clone(),
                future_found=self.future_found.detach().cpu().clone(),
                search_value_valid=self.search_value_valid.detach().cpu().clone(),
            )
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != REPLAY_SCHEMA:
            raise ValueError("unsupported PRRAC replay schema")
        self.base.load_state_dict(state["base_replay"])
        for name in ("stage_before", "stage_after"):
            source = torch.as_tensor(state[name], dtype=torch.int64)
            target = getattr(self, name)
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(f"PRRAC replay {name} shape mismatch")
            target.copy_(source.to(target.device))
        if self.search_value_enabled:
            stored_config = state.get("search_value")
            if stored_config is not None and resolve_search_value_config(
                stored_config
            ) != self.search_value_config:
                raise ValueError("PRRAC replay search-value configuration mismatch")
            if "search_features" in state:
                for name in (
                    "search_features",
                    "future_found",
                    "search_value_valid",
                ):
                    source = torch.as_tensor(state[name])
                    target = getattr(self, name)
                    if tuple(source.shape) != tuple(target.shape):
                        raise ValueError(f"PRRAC replay {name} shape mismatch")
                    target.copy_(source.to(target.device, dtype=target.dtype))
            else:
                self.search_features.zero_()
                self.future_found.zero_()
                self.search_value_valid.zero_()
        valid = self.base._valid_indices()
        if valid.numel() and (
            bool(torch.any(self.stage_before[valid] < 0))
            or bool(torch.any(self.stage_after[valid] < 0))
        ):
            raise ValueError("PRRAC replay state contains missing stage metadata")

    def __len__(self) -> int:
        return len(self.base)


__all__ = ("PRRACBatch", "PRRACReplayAdapter")
