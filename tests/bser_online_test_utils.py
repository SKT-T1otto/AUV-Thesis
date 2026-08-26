"""Deterministic state variations for Phase 1B unit tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from tests.bser_test_utils import locked, synthetic_state
from chapter3_bser.online.mission_context import OnlineMissionContext


def state_at(step: int, *, target_found: bool = False):
    return replace(synthetic_state(), step=int(step), target_found=bool(target_found))


def shifted_belief(step: int = 1):
    state = state_at(step)
    values = np.asarray(state.target_belief.probabilities).copy()
    values[0], values[8] = 0.02, values[0] + values[8] - 0.02
    values = locked(values, np.float64)
    positive = values[values > 0]
    belief = replace(
        state.target_belief,
        probabilities=values,
        entropy=float(-np.sum(positive * np.log(positive))),
        peak_probability=float(values.max()),
        peak_index=int(values.argmax()),
        revision=state.target_belief.revision + 1,
    )
    return replace(state, target_belief=belief)


def discovered_obstacle(step: int = 1):
    return discovered_obstacle_at(4, step=step)


def discovered_obstacle_at(index: int, step: int = 1):
    state = state_at(step)
    occupied = np.asarray(state.occupancy.occupied_mask).copy()
    occupied[int(index)] = True
    probability = np.asarray(state.occupancy.occupancy_probability).copy()
    probability[int(index)] = 0.95
    occupancy = replace(
        state.occupancy,
        occupied_mask=locked(occupied, np.bool_),
        occupancy_probability=locked(probability, np.float64),
        revision=state.occupancy.revision + 1,
    )
    return replace(state, occupancy=occupancy, map_revision=state.map_revision + 1)


def mission_context(state, *, executor_knows_target: bool = False, target_found: bool | None = None):
    return OnlineMissionContext(
        step=int(state.step),
        target_found=bool(state.target_found if target_found is None else target_found),
        finder_id=0 if (state.target_found if target_found is None else target_found) else -1,
        executor_knows_target=bool(executor_knows_target),
        handoff_step=int(state.step) if executor_knows_target else None,
        mission_complete=bool(state.mission_complete),
        executor_navigation_target=state.agents[state.executor_id].current_navigation_target,
        target_known_by_agent=(True, False, False, bool(executor_knows_target)),
        searcher_finished_flags=(False, False, False),
    )
