"""Fixed-seed random feasible allocation baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from chapter3_bser.exact_solver import partition_groups
from chapter3_bser.objective import ObjectiveContext, evaluate_objective
from chapter3_bser.types import SearchCandidate, StandbyCandidate


@dataclass(frozen=True)
class RandomAllocatorSummary:
    mean: float
    std: float
    median: float
    best: float
    repetitions: int


def evaluate_random_allocator(
    candidates: Sequence[SearchCandidate], standby_candidates: Sequence[StandbyCandidate], context: ObjectiveContext, *, repetitions: int, seed: int
) -> RandomAllocatorSummary:
    rng = np.random.default_rng(int(seed))
    groups = partition_groups(candidates)
    values = []
    for _ in range(int(repetitions)):
        selected = []
        for group in groups:
            choice = int(rng.integers(0, len(group) + 1))
            if choice:
                selected.append(group[choice - 1])
        standby = standby_candidates[int(rng.integers(0, len(standby_candidates)))]
        values.append(evaluate_objective(selected, standby, context))
    array = np.asarray(values, dtype=np.float64)
    return RandomAllocatorSummary(float(np.mean(array)), float(np.std(array)), float(np.median(array)), float(np.max(array)), len(values))
