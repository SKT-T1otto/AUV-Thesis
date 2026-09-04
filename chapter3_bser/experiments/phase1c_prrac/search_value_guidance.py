"""S2B inference-only, bounded auxiliary ranking of existing BSER candidates.

The objective and executor standby selection remain the original BSER result.
Only positive-marginal-gain search candidates are reranked. Counterfactual
features preview a candidate's initial local navigation target, not an imagined
future position, observation of the hidden world, or a newly trained value model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
import torch

from chapter3_bser.controllers.path_tracker import PathTracker
from chapter3_bser.models.search_value_head import (
    SEARCH_FEATURE_DIM, SearchStateFeatureExtractor, SearchValueHead,
)
from chapter3_bser.objective import evaluate_objective, marginal_gain
from chapter3_bser.online.allocator import BSEROnlineAllocator
from chapter3_bser.types import SolverResult


DEFAULT_SEARCH_VALUE_GUIDANCE = {
    "enabled": False, "weight": 0.1, "clip_min": 0.0, "clip_max": 1.0,
}


def resolve_search_value_guidance(config: Mapping[str, Any] | None = None) -> dict:
    result = {**DEFAULT_SEARCH_VALUE_GUIDANCE, **dict(config or {})}
    result["enabled"] = bool(result["enabled"])
    for key in ("weight", "clip_min", "clip_max"):
        result[key] = float(result[key])
        if not math.isfinite(result[key]):
            raise ValueError(f"search_value_guidance.{key} must be finite")
    if not 0.0 <= result["weight"] <= 0.1:
        raise ValueError("search_value_guidance.weight must be in [0, 0.1]")
    if not 0.0 <= result["clip_min"] <= result["clip_max"] <= 1.0:
        raise ValueError("search_value_guidance clips must satisfy 0 <= min <= max <= 1")
    return result


@dataclass(frozen=True)
class CandidateScore:
    candidate: Any
    bser_score: float
    search_value: float
    final_score: float


class SearchValueGuidedCandidateScore:
    """Frozen head inference and deterministic ranking, with episode diagnostics."""

    def __init__(self, config=None, *, head=None, max_steps=400):
        self.config = resolve_search_value_guidance(config)
        self.active = self.config["enabled"] and self.config["weight"] > 0.0
        if self.active and head is None:
            raise ValueError("active search value guidance requires a checkpoint search_value_head")
        self.head = head
        self.extractor = SearchStateFeatureExtractor(max_steps)
        self._state_step = None
        self._features = {}
        self.candidate_count = self.selected_count = self.ranking_changed_count = 0
        self.allocation_changed_count = self.accepted_search_change_count = 0
        self.value_sum = self.selected_value_sum = self.selected_rank_sum = 0.0
        self._changed_proposal = None

    @classmethod
    def from_snapshot(cls, config, *, snapshot, head_config, max_steps=400):
        resolved = resolve_search_value_guidance(config)
        if not resolved["enabled"] or resolved["weight"] == 0.0:
            return cls(resolved, max_steps=max_steps)
        if not snapshot or not head_config or not head_config.get("enabled"):
            raise ValueError("active search value guidance requires a checkpoint search_value_head")
        # Keep actor/exploration RNG streams unchanged by inference-only construction.
        with torch.random.fork_rng(devices=[]):
            head = SearchValueHead(hidden_dim=int(head_config["hidden_dim"]))
        head.load_state_dict({key: torch.as_tensor(value).clone() for key, value in snapshot.items()}, strict=True)
        head.eval().requires_grad_(False)
        return cls(resolved, head=head, max_steps=max_steps)

    def observe_state(self, observations, state, *, collision_flags=None):
        if not self.active:
            return
        if self._state_step is None:
            self.extractor.reset(state)
        elif self._state_step != int(state.step) and collision_flags is not None:
            self.extractor.observe_transition(state, collision_flags)
        else:
            self.extractor.synchronize_state(state)
        obs = np.stack([
            torch.as_tensor(observations[int(agent_id)]).detach().cpu().numpy()
            for agent_id in state.searcher_ids
        ])
        features = self.extractor.extract(obs, state)
        self._features = dict(zip(state.searcher_ids, features))
        self._state_step = int(state.step)

    def candidate_feature(self, candidate, state):
        if self._state_step != int(state.step):
            raise ValueError("candidate ranking requires current public search features")
        feature = self._features[candidate.agent_id].copy()
        agent = next(item for item in state.agents if item.agent_id == candidate.agent_id)
        # An isolated tracker previews the existing route; the live bridge is untouched.
        target = PathTracker().tracking_target(
            candidate.agent_id, agent.position, candidate.path_points, candidate.waypoint,
        )
        delta = np.asarray(target, dtype=np.float64) - np.asarray(agent.position)
        distance = float(np.linalg.norm(delta))
        direction = delta / max(distance, 1e-6)
        feature[6:9] = delta
        feature[9:12] = direction
        feature[15] = np.clip(distance / 10.0, 0.0, 1.0)
        feature[17] = np.tanh(np.dot(agent.velocity, direction) / (agent.horizontal_speed_limit + 1e-6))
        return feature

    def estimate_candidate_value(self, candidate_state) -> float:
        """Predict from a 34D candidate-conditioned side-channel feature, not actor input."""
        if self.head is None:
            raise ValueError("search_value_head is unavailable")
        feature = torch.as_tensor(candidate_state, dtype=torch.float32)
        if feature.shape != (SEARCH_FEATURE_DIM,) or not torch.isfinite(feature).all():
            raise ValueError("candidate_state must be a finite 34D feature")
        with torch.inference_mode():
            probability = float(self.head(feature.unsqueeze(0)).item())
        if not math.isfinite(probability):
            raise ValueError("non-finite candidate search value")
        return float(np.clip(probability, self.config["clip_min"], self.config["clip_max"]))

    def rank(self, candidates, bser_scores, values=None):
        """Scores are the original conditional marginal gains (no normalization)."""
        rows = []
        for candidate in candidates:
            base = float(bser_scores[candidate.key])
            value = float(values[candidate.key]) if self.active else 0.0
            if not math.isfinite(value):
                raise ValueError("candidate search value must be finite")
            value = float(np.clip(value, self.config["clip_min"], self.config["clip_max"])) if self.active else 0.0
            rows.append(CandidateScore(candidate, base, value, base + self.config["weight"] * value))
        baseline = sorted(rows, key=lambda row: (-row.bser_score, row.candidate.key))
        ranked = sorted(rows, key=lambda row: (-row.final_score, row.candidate.key))
        if self.active and [row.candidate.key for row in ranked] != [row.candidate.key for row in baseline]:
            self.ranking_changed_count += 1
        return ranked

    def record_installed(self, allocation, *, accepted=True):
        signature = {(item.agent_id, item.candidate_id) for item in allocation.search_assignments}
        if accepted and self._changed_proposal and self._changed_proposal.issubset(signature):
            self.accepted_search_change_count += 1
        self._changed_proposal = None

    def metrics(self):
        return {
            "enabled": self.config["enabled"], "active": self.active,
            "config": dict(self.config), "candidate_count": self.candidate_count,
            "mean_search_value": self.value_sum / self.candidate_count if self.candidate_count else 0.0,
            "selected_candidate_count": self.selected_count,
            "mean_selected_candidate_value": self.selected_value_sum / self.selected_count if self.selected_count else 0.0,
            "selected_value_rank": self.selected_rank_sum / self.selected_count if self.selected_count else 0.0,
            "ranking_changed_count": self.ranking_changed_count,
            "allocation_changed_count": self.allocation_changed_count,
            "accepted_search_change_count": self.accepted_search_change_count,
        }


class SearchValueGuidedBSERAllocator(BSEROnlineAllocator):
    """Use original generation/objective/executor; override only search ranking."""

    def __init__(self, scorer, phase1a1_config=None):
        super().__init__(phase1a1_config)
        self.scorer = scorer

    def _solve_candidates(self, candidates, standby_candidates, context):
        baseline = super()._solve_candidates(candidates, standby_candidates, context)
        scorer = self.scorer
        scorer._changed_proposal = None
        if not scorer.active or context.state.target_found or baseline.standby is None:
            return baseline
        # Freeze the original executor choice. No auxiliary term enters the
        # objective reported to the controller's existing acceptance/hysteresis.
        values = {item.key: scorer.estimate_candidate_value(scorer.candidate_feature(item, context.state)) for item in candidates}
        scorer.candidate_count += len(values)
        scorer.value_sum += sum(values.values())
        selected, used_agents = [], set()
        while True:
            feasible = [item for item in candidates if item.agent_id not in used_agents]
            gains = {item.key: marginal_gain(selected, item, baseline.standby, context) for item in feasible}
            feasible = [item for item in feasible if gains[item.key] > 1e-15]
            if not feasible:
                break
            chosen = scorer.rank(feasible, gains, values)[0].candidate
            selected.append(chosen)
            used_agents.add(chosen.agent_id)
        selected.sort(key=lambda item: item.key)
        # Partial allocation merges frozen unaffected assignments after solving.
        # Count only changes to generated search candidates, not frozen rows.
        baseline_keys = {(item.agent_id, item.candidate_id) for item in baseline.selected
                         if item.source != "current_assignment"}
        selected_keys = {(item.agent_id, item.candidate_id) for item in selected
                         if item.source != "current_assignment"}
        if selected_keys != baseline_keys:
            scorer.allocation_changed_count += 1
            scorer._changed_proposal = selected_keys - baseline_keys
        for item in selected:
            scorer.selected_count += 1
            scorer.selected_value_sum += values[item.key]
            # One-based value rank within this searcher's candidate pool; ties share rank.
            scorer.selected_rank_sum += 1 + sum(
                other.agent_id == item.agent_id and values[other.key] > values[item.key]
                for other in candidates
            )
        return SolverResult(baseline.solver, tuple(selected), baseline.standby,
                            evaluate_objective(selected, baseline.standby, context))


def aggregate_search_value_guidance(rows):
    """Weighted aggregation of episode statistics (safe across CSV/resume)."""
    metrics = [row["search_value_guidance"] for row in rows if "search_value_guidance" in row]
    result = {"enabled": any(item["enabled"] for item in metrics), "episode_count": len(metrics)}
    for name in ("candidate_count", "selected_candidate_count", "ranking_changed_count", "allocation_changed_count", "accepted_search_change_count"):
        result[name] = sum(item[name] for item in metrics)
    for name, count in (("mean_search_value", "candidate_count"), ("mean_selected_candidate_value", "selected_candidate_count"), ("selected_value_rank", "selected_candidate_count")):
        result[name] = sum(item[name] * item[count] for item in metrics) / result[count] if result[count] else 0.0
    result["definitions"] = {
        "candidate_count": "candidates scored once per allocation proposal",
        "ranking_changed_count": "greedy rounds whose positive-gain order differs from BSER at the same selected prefix",
        "selected_value_rank": "mean one-based descending value rank within the selected searcher's candidate pool; ties share rank",
        "allocation_changed_count": "proposals with different search selections from original joint BSER",
        "accepted_search_change_count": "changed proposals actually returned by the existing controller",
    }
    return result
