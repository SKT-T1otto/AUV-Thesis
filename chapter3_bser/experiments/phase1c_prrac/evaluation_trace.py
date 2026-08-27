"""Privileged failure tracing isolated from PRRAC policy inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch


def _plain(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def vector3(value: Any) -> list[float] | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size != 3 or not np.all(np.isfinite(array)):
        return None
    return [float(item) for item in array]


def failure_trace_row(**values: Any) -> dict[str, Any]:
    """Create a diagnostic-only row; callers must never reuse it as policy input."""

    row = {str(key): _plain(value) for key, value in values.items()}
    row["diagnostic_only"] = True
    row["privileged_oracle"] = True
    return row


@dataclass
class FailureTraceRecorder:
    enabled: bool = True
    only_found_failures: bool = True
    max_traces: int = 20
    accepted_trace_count: int = 0
    _episode_rows: list[dict[str, Any]] = field(default_factory=list)

    def begin_episode(self) -> None:
        self._episode_rows = []

    def record(self, row: Mapping[str, Any]) -> None:
        if self.enabled and self.accepted_trace_count < int(self.max_traces):
            self._episode_rows.append(_plain(dict(row)))

    def finish_episode(
        self,
        *,
        found: bool,
        success: bool,
        checkpoint: str,
        checkpoint_episode: int,
        evaluation_mode: str,
        scenario_id: str,
        scenario_seed: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        accepted = bool(
            self.enabled
            and self.accepted_trace_count < int(self.max_traces)
            and not success
            and (found or not self.only_found_failures)
        )
        if not accepted:
            self._episode_rows = []
            return [], None
        rows = [dict(row) for row in self._episode_rows]
        self._episode_rows = []
        self.accepted_trace_count += 1
        index = {
            "checkpoint": str(checkpoint),
            "checkpoint_episode": int(checkpoint_episode),
            "evaluation_mode": str(evaluation_mode),
            "scenario_id": str(scenario_id),
            "scenario_seed": int(scenario_seed),
            "trace_row_count": len(rows),
            "diagnostic_only": True,
            "privileged_oracle": True,
        }
        return rows, index


__all__ = ("FailureTraceRecorder", "failure_trace_row", "vector3")
