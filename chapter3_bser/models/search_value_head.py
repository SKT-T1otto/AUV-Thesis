"""Auxiliary future-discovery value model for Phase 1C search decisions.

The extractor and head are deliberately separate from the frozen 28D actor
observation path.  The head consumes a copy of that observation plus public,
training-only search statistics; none of these values are appended to actor or
critic inputs.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


ACTOR_OBSERVATION_DIM = 28
SEARCHER_COUNT = 3
SEARCH_AUXILIARY_DIM = 6
SEARCH_FEATURE_DIM = ACTOR_OBSERVATION_DIM + SEARCH_AUXILIARY_DIM

DEFAULT_SEARCH_VALUE_CONFIG = {
    "enabled": False,
    "hidden_dim": 128,
    "horizon": 50,
    "loss_weight": 0.05,
    "threshold": 0.5,
}


def resolve_search_value_config(
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the validated, fully resolved auxiliary-head configuration."""

    resolved = dict(DEFAULT_SEARCH_VALUE_CONFIG)
    resolved.update(dict(config or {}))
    resolved["enabled"] = bool(resolved["enabled"])
    resolved["hidden_dim"] = int(resolved["hidden_dim"])
    resolved["horizon"] = int(resolved["horizon"])
    resolved["loss_weight"] = float(resolved["loss_weight"])
    resolved["threshold"] = float(resolved["threshold"])
    if resolved["hidden_dim"] <= 0:
        raise ValueError("search_value.hidden_dim must be positive")
    if resolved["horizon"] <= 0:
        raise ValueError("search_value.horizon must be positive")
    if not 0.0 <= resolved["loss_weight"] <= 0.1:
        raise ValueError("search_value.loss_weight must be in [0, 0.1]")
    if not 0.0 <= resolved["threshold"] <= 1.0:
        raise ValueError("search_value.threshold must be in [0, 1]")
    return resolved


class SearchValueHead(nn.Module):
    """Small MLP estimating discovery probability within the configured horizon."""

    def __init__(
        self,
        feature_dim: int = SEARCH_FEATURE_DIM,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if int(feature_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("SearchValueHead dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, search_feature: torch.Tensor) -> torch.Tensor:
        if search_feature.shape[-1] != self.feature_dim:
            raise ValueError(
                f"search feature must be {self.feature_dim}D, "
                f"got {search_feature.shape[-1]}"
            )
        return torch.sigmoid(self.network(search_feature))


class SearchStateFeatureExtractor:
    """Build bounded side-channel features from public search state.

    Auxiliary columns after the copied 28D actor observation are, in order:
    normalized step, known-area ratio, recent collision ratio, normalized
    distance since new map information, normalized belief entropy, and belief
    peak probability.
    """

    def __init__(self, max_steps: int, *, collision_window: int = 20) -> None:
        if int(max_steps) <= 0 or int(collision_window) <= 0:
            raise ValueError("feature extractor windows must be positive")
        self.max_steps = int(max_steps)
        self.collision_window = int(collision_window)
        self._collisions = [
            deque(maxlen=self.collision_window) for _ in range(SEARCHER_COUNT)
        ]
        self._distance_since_information = np.zeros(SEARCHER_COUNT, dtype=np.float64)
        self._last_positions: np.ndarray | None = None
        self._last_known_ratio = 0.0
        self._last_map_revision = -1

    @staticmethod
    def _known_ratio(planning_state: Any) -> float:
        known = np.asarray(planning_state.occupancy.known_mask, dtype=np.float64)
        return 0.0 if known.size == 0 else float(np.mean(known))

    @staticmethod
    def _positions(planning_state: Any) -> np.ndarray:
        searcher_ids = tuple(int(value) for value in planning_state.searcher_ids)
        if len(searcher_ids) != SEARCHER_COUNT:
            raise ValueError("SearchValueHead requires exactly three searchers")
        by_id = {int(agent.agent_id): agent for agent in planning_state.agents}
        return np.asarray(
            [by_id[agent_id].position for agent_id in searcher_ids],
            dtype=np.float64,
        ).reshape(SEARCHER_COUNT, 3)

    @staticmethod
    def _map_diagonal(planning_state: Any) -> float:
        shape = np.asarray(planning_state.grid.shape, dtype=np.float64)
        spacing = np.asarray(planning_state.grid.spacing, dtype=np.float64)
        return max(float(np.linalg.norm(shape * spacing)), 1e-6)

    @staticmethod
    def _belief_statistics(planning_state: Any) -> tuple[float, float]:
        probabilities = np.asarray(
            planning_state.target_belief.probabilities, dtype=np.float64
        ).reshape(-1)
        cell_count = max(2, int(probabilities.size))
        entropy = float(planning_state.target_belief.entropy)
        normalized_entropy = entropy / math.log(cell_count)
        peak = float(planning_state.target_belief.peak_probability)
        return (
            float(np.clip(normalized_entropy, 0.0, 1.0)),
            float(np.clip(peak, 0.0, 1.0)),
        )

    def reset(self, planning_state: Any) -> None:
        for history in self._collisions:
            history.clear()
        self._distance_since_information.fill(0.0)
        self._last_positions = self._positions(planning_state)
        self._last_known_ratio = self._known_ratio(planning_state)
        self._last_map_revision = int(planning_state.map_revision)

    def extract(
        self,
        actor_observations: Sequence[Any],
        planning_state: Any,
    ) -> np.ndarray:
        observations = np.asarray(actor_observations, dtype=np.float32)
        if observations.shape != (SEARCHER_COUNT, ACTOR_OBSERVATION_DIM):
            raise ValueError(
                "search actor observations must have shape "
                f"({SEARCHER_COUNT}, {ACTOR_OBSERVATION_DIM})"
            )
        if self._last_positions is None:
            self.reset(planning_state)
        normalized_step = float(
            np.clip(float(planning_state.step) / self.max_steps, 0.0, 1.0)
        )
        known_ratio = float(np.clip(self._known_ratio(planning_state), 0.0, 1.0))
        belief_entropy, belief_peak = self._belief_statistics(planning_state)
        diagonal = self._map_diagonal(planning_state)
        rows = []
        for agent_i in range(SEARCHER_COUNT):
            collision_ratio = sum(self._collisions[agent_i]) / self.collision_window
            distance_ratio = float(
                np.clip(self._distance_since_information[agent_i] / diagonal, 0.0, 1.0)
            )
            auxiliary = np.asarray(
                (
                    normalized_step,
                    known_ratio,
                    collision_ratio,
                    distance_ratio,
                    belief_entropy,
                    belief_peak,
                ),
                dtype=np.float32,
            )
            rows.append(np.concatenate((observations[agent_i], auxiliary)))
        result = np.stack(rows).astype(np.float32, copy=False)
        if result.shape != (SEARCHER_COUNT, SEARCH_FEATURE_DIM):
            raise RuntimeError("invalid SearchValueHead feature shape")
        if not np.all(np.isfinite(result)):
            raise ValueError("search features must be finite")
        return result

    def observe_transition(
        self,
        planning_state_after: Any,
        collision_flags: Sequence[Any],
    ) -> None:
        current_positions = self._positions(planning_state_after)
        if self._last_positions is None:
            self._last_positions = current_positions
        self._distance_since_information += np.linalg.norm(
            current_positions - self._last_positions, axis=1
        )
        collisions = np.asarray(collision_flags, dtype=np.bool_).reshape(-1)
        if collisions.size < SEARCHER_COUNT:
            raise ValueError("collision flags must include all searchers")
        for agent_i in range(SEARCHER_COUNT):
            self._collisions[agent_i].append(int(collisions[agent_i]))
        known_ratio = self._known_ratio(planning_state_after)
        map_revision = int(planning_state_after.map_revision)
        if (
            known_ratio > self._last_known_ratio + 1e-12
            or map_revision > self._last_map_revision
        ):
            self._distance_since_information.fill(0.0)
        self._last_positions = current_positions
        self._last_known_ratio = known_ratio
        self._last_map_revision = map_revision


def future_found_labels(found_after_step: Sequence[Any], horizon: int) -> np.ndarray:
    """Label states whose first future discovery occurs within ``horizon`` steps."""

    if int(horizon) <= 0:
        raise ValueError("future-found horizon must be positive")
    flags = np.asarray(found_after_step, dtype=np.bool_).reshape(-1)
    labels = np.zeros(flags.shape, dtype=np.float32)
    found_indices = np.flatnonzero(flags)
    if found_indices.size:
        first_found = int(found_indices[0])
        start = max(0, first_found - int(horizon) + 1)
        labels[start : first_found + 1] = 1.0
    return labels


__all__ = (
    "ACTOR_OBSERVATION_DIM",
    "DEFAULT_SEARCH_VALUE_CONFIG",
    "SEARCHER_COUNT",
    "SEARCH_FEATURE_DIM",
    "SearchStateFeatureExtractor",
    "SearchValueHead",
    "future_found_labels",
    "resolve_search_value_config",
)
