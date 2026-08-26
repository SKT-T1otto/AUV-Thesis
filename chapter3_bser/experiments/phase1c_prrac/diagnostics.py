"""PRRAC routing, expert, gate, and stage-critic diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _quantile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.quantile(np.asarray(values), q))


@dataclass
class PRRACDiagnostics:
    gates: list[float] = field(default_factory=list)
    alignments: list[float] = field(default_factory=list)
    router_probabilities: list[list[float]] = field(default_factory=list)
    expert_norms: list[list[float]] = field(default_factory=list)
    router_confusion_matrix: list[list[int]] = field(
        default_factory=lambda: [[0, 0, 0] for _ in range(3)]
    )
    stage_critic_losses: dict[str, list[float]] = field(
        default_factory=lambda: {name: [] for name in ("search", "intercept", "hold")}
    )
    stage_td_errors: dict[str, list[float]] = field(
        default_factory=lambda: {name: [] for name in ("search", "intercept", "hold")}
    )

    def observe_actor(self, output, stages: Any) -> None:
        gate = output.trust_gate.detach().cpu().reshape(-1)
        alignment = output.alignment_cosine.detach().cpu().reshape(-1)
        probabilities = output.router_probabilities.detach().cpu()
        norms = torch.linalg.vector_norm(output.expert_actions.detach().cpu(), dim=-1)
        labels = torch.as_tensor(stages, dtype=torch.long).reshape(-1)
        predictions = probabilities.argmax(dim=-1)
        self.gates.extend(float(value) for value in gate.tolist())
        self.alignments.extend(float(value) for value in alignment.tolist())
        self.router_probabilities.extend(
            [[float(item) for item in row] for row in probabilities.tolist()]
        )
        self.expert_norms.extend(
            [[float(item) for item in row] for row in norms.tolist()]
        )
        for actual, predicted in zip(labels.tolist(), predictions.tolist()):
            self.router_confusion_matrix[int(actual)][int(predicted)] += 1

    def observe_update(self, update: Mapping[str, Any]) -> None:
        for field, target in (
            ("stage_critic_losses", self.stage_critic_losses),
            ("stage_td_errors", self.stage_td_errors),
        ):
            for name, value in dict(update.get(field, {})).items():
                if value is not None:
                    target[name].append(float(value))

    def summary(self) -> dict[str, Any]:
        total = sum(sum(row) for row in self.router_confusion_matrix)
        correct = sum(self.router_confusion_matrix[index][index] for index in range(3))
        probabilities = np.asarray(self.router_probabilities, dtype=np.float64)
        norms = np.asarray(self.expert_norms, dtype=np.float64)
        return {
            "gate_mean": _mean(self.gates),
            "gate_p10": _quantile(self.gates, 0.10),
            "gate_p50": _quantile(self.gates, 0.50),
            "gate_p90": _quantile(self.gates, 0.90),
            "gate_below_0_05_rate": (
                None if not self.gates else sum(value < 0.05 for value in self.gates) / len(self.gates)
            ),
            "gate_above_0_95_rate": (
                None if not self.gates else sum(value > 0.95 for value in self.gates) / len(self.gates)
            ),
            "gate_saturation_low_rate": (
                None if not self.gates else sum(value < 0.05 for value in self.gates) / len(self.gates)
            ),
            "gate_saturation_high_rate": (
                None if not self.gates else sum(value > 0.95 for value in self.gates) / len(self.gates)
            ),
            "alignment_mean": _mean(self.alignments),
            "alignment_negative_rate": (
                None
                if not self.alignments
                else sum(value < 0.0 for value in self.alignments) / len(self.alignments)
            ),
            "expert_action_norm_search": None if norms.size == 0 else float(norms[:, 0].mean()),
            "expert_action_norm_intercept": None if norms.size == 0 else float(norms[:, 1].mean()),
            "expert_action_norm_hold": None if norms.size == 0 else float(norms[:, 2].mean()),
            "router_probability_search": None if probabilities.size == 0 else float(probabilities[:, 0].mean()),
            "router_probability_intercept": None if probabilities.size == 0 else float(probabilities[:, 1].mean()),
            "router_probability_hold": None if probabilities.size == 0 else float(probabilities[:, 2].mean()),
            "router_argmax_accuracy": None if total == 0 else correct / total,
            "router_accuracy": None if total == 0 else correct / total,
            "router_confusion_matrix": [list(row) for row in self.router_confusion_matrix],
            "router_stage_counts": [sum(row) for row in self.router_confusion_matrix],
            "stage_critic_losses": {
                name: _mean(values) for name, values in self.stage_critic_losses.items()
            },
            "stage_td_errors": {
                name: _mean(values) for name, values in self.stage_td_errors.items()
            },
        }


__all__ = ("PRRACDiagnostics",)
