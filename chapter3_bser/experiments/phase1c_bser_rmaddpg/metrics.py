"""Metric accumulation for the action-free-update Phase 1C preflight."""

from __future__ import annotations

from collections import Counter
import math
import statistics
from typing import Any, Iterable


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(finite)) if finite else 0.0


class EpisodeMetrics:
    """Accumulate the required BSER, RL, and environment diagnostics."""

    def __init__(self) -> None:
        self.allocation_versions: list[str] = []
        self.events: Counter[str] = Counter()
        self.replan_attempt_count = 0
        self.accepted_replan_count = 0
        self.rejected_replan_count = 0
        self.action_norms: list[float] = []
        self.prior_norms: list[float] = []
        self.residual_norms: list[float] = []
        self.residual_ratios: list[float] = []
        self.collision_count = 0
        self.total_reward = 0.0

    def record_step(
        self,
        *,
        allocation_version: str,
        events: Iterable[str],
        replan_attempted: bool,
        accepted_replan: bool,
        action_norm: float,
        prior_norm: float,
        residual_norm: float,
        residual_ratio: float,
        collision_count: int,
        reward: float,
    ) -> None:
        self.allocation_versions.append(str(allocation_version))
        self.events.update(str(event) for event in events)
        self.replan_attempt_count += int(bool(replan_attempted))
        self.accepted_replan_count += int(bool(accepted_replan))
        self.rejected_replan_count += int(
            bool(replan_attempted) and not bool(accepted_replan)
        )
        self.action_norms.append(float(action_norm))
        self.prior_norms.append(float(prior_norm))
        self.residual_norms.append(float(residual_norm))
        self.residual_ratios.append(float(residual_ratio))
        self.collision_count += int(collision_count)
        self.total_reward += float(reward)

    def finalize(
        self,
        *,
        method: str,
        episode_index: int,
        scenario_seed: int,
        steps: int,
        found: bool,
        success: bool,
        found_step: int | None,
        success_step: int | None,
        initial_allocation_version: str,
        final_allocation_version: str,
    ) -> dict[str, Any]:
        return {
            "method": str(method),
            "episode_index": int(episode_index),
            "scenario_seed": int(scenario_seed),
            "steps": int(steps),
            "found": bool(found),
            "success": bool(success),
            "found_step": "" if found_step is None else int(found_step),
            "success_step": "" if success_step is None else int(success_step),
            "initial_allocation_version": str(initial_allocation_version),
            "final_allocation_version": str(final_allocation_version),
            "allocation_version_count": len(set(self.allocation_versions)),
            "event_count": int(sum(self.events.values())),
            "events": "|".join(
                f"{name}:{count}" for name, count in sorted(self.events.items())
            ),
            "replan_attempt_count": int(self.replan_attempt_count),
            "accepted_replan_count": int(self.accepted_replan_count),
            "rejected_replan_count": int(self.rejected_replan_count),
            "mean_action_norm": _mean(self.action_norms),
            "mean_prior_norm": _mean(self.prior_norms),
            "mean_residual_norm": _mean(self.residual_norms),
            "mean_residual_ratio": _mean(self.residual_ratios),
            "collision_count": int(self.collision_count),
            "total_reward": float(self.total_reward),
        }


def summarize_preflight(
    episode_rows: list[dict[str, Any]],
    *,
    expected_episodes: int,
    failures: list[dict[str, Any]],
    method: str,
) -> dict[str, Any]:
    completed = len(episode_rows)
    dimensions_ok = all(
        int(row.get("observation_dim", 28)) == 28
        and int(row.get("action_dim", 3)) == 3
        and int(row.get("critic_input_dim", 124)) == 124
        for row in episode_rows
    )
    finite_metric_keys = (
        "mean_action_norm",
        "mean_prior_norm",
        "mean_residual_norm",
        "mean_residual_ratio",
    )
    finite_metrics = all(
        math.isfinite(float(row[key]))
        for row in episode_rows
        for key in finite_metric_keys
    )
    allocation_versions_present = all(
        bool(row["initial_allocation_version"])
        and bool(row["final_allocation_version"])
        for row in episode_rows
    )
    passed = bool(
        completed == int(expected_episodes)
        and not failures
        and dimensions_ok
        and finite_metrics
        and allocation_versions_presen
    )
    return {
        "schema": "bser.phase1c.preflight.summary.v1",
        "method": str(method),
        "passed": passed,
        "training_update": False,
        "parameter_update_count": 0,
        "expected_episode_count": int(expected_episodes),
        "completed_episode_count": completed,
        "failure_count": len(failures),
        "failures": failures,
        "observation_dim": 28,
        "action_dim": 3,
        "critic_input_dim": 124,
        "dimensions_compatible": dimensions_ok,
        "allocation_versions_present": allocation_versions_present,
        "finite_rl_metrics": finite_metrics,
        "success_rate": (
            sum(bool(row["success"]) for row in episode_rows) / completed
            if completed
            else 0.0
        ),
        "found_rate": (
            sum(bool(row["found"]) for row in episode_rows) / completed
            if completed
            else 0.0
        ),
        "collision_count": sum(
            int(row["collision_count"]) for row in episode_rows
        ),
        "event_count": sum(int(row["event_count"]) for row in episode_rows),
        "replan_attempt_count": sum(
            int(row["replan_attempt_count"]) for row in episode_rows
        ),
        "accepted_replan_count": sum(
            int(row["accepted_replan_count"]) for row in episode_rows
        ),
        "mean_action_norm": _mean(
            row["mean_action_norm"] for row in episode_rows
        ),
        "mean_prior_norm": _mean(
            row["mean_prior_norm"] for row in episode_rows
        ),
        "mean_residual_norm": _mean(
            row["mean_residual_norm"] for row in episode_rows
        ),
        "mean_residual_ratio": _mean(
            row["mean_residual_ratio"] for row in episode_rows
        ),
    }
