"""Copy-only deterministic evaluation action ablations."""

from __future__ import annotations

import torch

from chapter3_bser.models.prrac.stage_mapping import PRRACStage


def apply_residual_mode(
    actions: torch.Tensor, mode: str, stage_before: int | PRRACStage
) -> torch.Tensor:
    result = actions.clone()
    if mode in {"full_prrac", "oracle_current_target_diagnostic"}:
        return result
    if mode == "executor_residual_off":
        result[3].zero_()
        return result
    if mode == "all_residual_off":
        result.zero_()
        return result
    if mode == "searcher_residual_off":
        if int(stage_before) == int(PRRACStage.SEARCH):
            result[:3].zero_()
        return result
    raise ValueError(f"unsupported PRRAC evaluation mode: {mode!r}")


__all__ = ("apply_residual_mode",)
