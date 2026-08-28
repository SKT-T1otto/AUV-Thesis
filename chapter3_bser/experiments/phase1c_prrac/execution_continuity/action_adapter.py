"""Shared residual-action adapter for B3 explicit SAFE_HOLD."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .config import parse_execution_variant
from .types import (
    ExecutionNavigationPlanV3,
    ExecutionVariant,
    NavigationMode,
    ResidualSuppressionDiagnostics,
)


def _norm(value: Any) -> float:
    if torch.is_tensor(value):
        return float(torch.linalg.vector_norm(value.detach()).cpu().item())
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))


class ExecutionContinuityActionAdapter:
    """Apply only the registered B3 SAFE_HOLD executor suppression."""

    def apply(
        self,
        raw_actions: Any,
        *,
        plan: ExecutionNavigationPlanV3 | None,
        variant: str | ExecutionVariant,
        mission_phase: str,
        executor_id: int = 3,
    ) -> tuple[Any, ResidualSuppressionDiagnostics]:
        selected = parse_execution_variant(variant)
        applied = raw_actions.clone() if torch.is_tensor(raw_actions) else np.asarray(raw_actions).copy()
        raw_norm = _norm(applied[int(executor_id)])
        suppress = bool(
            selected is ExecutionVariant.B3_PROXY_SAFE_SUPPRESSION
            and str(mission_phase).upper() == "EXECUTION"
            and plan is not None
            and plan.navigation_mode is NavigationMode.SAFE_HOLD
            and plan.safe_hold
            and not plan.reachable
        )
        if suppress:
            if torch.is_tensor(applied):
                applied[int(executor_id)].zero_()
            else:
                applied[int(executor_id)] = 0
        applied_norm = _norm(applied[int(executor_id)])
        return applied, ResidualSuppressionDiagnostics(
            suppressed=suppress,
            executor_id=int(executor_id),
            raw_norm=raw_norm,
            applied_norm=applied_norm,
            reason="B3_EXECUTION_SAFE_HOLD" if suppress else "NOT_SUPPRESSED",
        )


__all__ = ("ExecutionContinuityActionAdapter",)
