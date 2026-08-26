"""Replay-only PRRAC phase-before/phase-after transition protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from chapter3_bser.experiments.phase1c_common import Phase1CTransitionMetadata
from chapter3_bser.models.prrac.stage_mapping import (
    PRRACStage,
    transition_phase_to_prrac_stage,
)


@dataclass(frozen=True)
class PRRACTransitionMetadata:
    base: Phase1CTransitionMetadata
    stage_before: PRRACStage
    stage_after: PRRACStage

    def __post_init__(self) -> None:
        if not isinstance(self.base, Phase1CTransitionMetadata):
            raise TypeError("PRRAC metadata base must be Phase1CTransitionMetadata")
        object.__setattr__(self, "stage_before", PRRACStage(int(self.stage_before)))
        object.__setattr__(self, "stage_after", PRRACStage(int(self.stage_after)))
        expected_after = transition_phase_to_prrac_stage(self.base.phase)
        if self.stage_after != expected_after:
            raise ValueError(
                f"stage_after {self.stage_after.name} does not match base phase "
                f"{self.base.phase.name}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": self.base.to_dict(),
            "stage_before": int(self.stage_before),
            "stage_before_name": self.stage_before.name,
            "stage_after": int(self.stage_after),
            "stage_after_name": self.stage_after.name,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PRRACTransitionMetadata":
        return cls(
            base=Phase1CTransitionMetadata.from_dict(value["base"]),
            stage_before=PRRACStage(int(value["stage_before"])),
            stage_after=PRRACStage(int(value["stage_after"])),
        )


__all__ = ("PRRACTransitionMetadata",)
