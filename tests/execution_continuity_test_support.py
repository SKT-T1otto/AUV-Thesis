from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.mapping.travel_cost_service import PathQueryResult, SingleSourceTravelResult

from chapter3_bser.experiments.phase1c_prrac.execution_continuity import (
    ExecutionNavigationPlanV3,
    ExecutionVariant,
    NavigationMode,
    OVERLAY_SCHEMA,
)


CENTERS = np.asarray(
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
    dtype=np.float64,
)


def state(*, executor_position=(0.0, 0.0, 0.0)):
    agents = tuple(
        SimpleNamespace(
            agent_id=index,
            role="executor" if index == 3 else "searcher",
            position=executor_position if index == 3 else (0.0, 0.0, 0.0),
        )
        for index in range(4)
    )
    return SimpleNamespace(
        step=20,
        executor_id=3,
        agents=agents,
        grid=SimpleNamespace(cell_centers=CENTERS),
        planning_graph=SimpleNamespace(valid_mask=np.ones(4, dtype=np.bool_)),
    )


def query_result(goal, *, reachable=True, cost=None):
    point = np.asarray(goal, dtype=np.float64).reshape(3)
    if not reachable:
        return PathQueryResult(
            False,
            float("inf"),
            float("inf"),
            float("inf"),
            0.0,
            0.0,
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            "unreachable",
        )
    value = float(np.linalg.norm(point)) if cost is None else float(cost)
    return PathQueryResult(
        True,
        value,
        value,
        value,
        0.0,
        0.0,
        np.asarray(((0.0, 0.0, 0.0), tuple(point)), dtype=np.float64),
        np.asarray((int(round(point[0])),), dtype=np.int64),
        None,
    )


class FakeTravelCostService:
    def __init__(self, reachable_goals=()):
        self.reachable_goals = {
            tuple(float(item) for item in goal) for goal in reachable_goals
        }
        self.queries = []

    def query(self, start, goal, agent):
        target = tuple(float(item) for item in goal)
        self.queries.append((tuple(float(item) for item in start), target))
        return query_result(target, reachable=target in self.reachable_goals)

    def single_source(self, start, agent):
        reachable = np.asarray(
            [tuple(row) in self.reachable_goals for row in CENTERS], dtype=np.bool_
        )
        costs = np.where(reachable, np.arange(4, dtype=np.float64), np.inf)
        return SingleSourceTravelResult(
            costs,
            costs.copy(),
            np.full(4, -1, dtype=np.int64),
            reachable,
            "fake-tree",
        )


def previous_plan(endpoint=(1.0, 0.0, 0.0)):
    endpoint = tuple(float(item) for item in endpoint)
    return ExecutionNavigationPlanV3(
        schema=OVERLAY_SCHEMA,
        variant=ExecutionVariant.B1_ATOMIC_LAST_VALID,
        semantic_target=(9.0, 0.0, 0.0),
        navigation_endpoint=endpoint,
        navigation_mode=NavigationMode.EXACT_PUBLIC_TARGET,
        reachable=True,
        path=((0.0, 0.0, 0.0), endpoint),
        path_cell_indices=(1,),
        planning_cost=1.0,
        estimated_arrival_time=1.0,
        source="previous",
        proxy_distance_to_semantic_target=None,
        preserved_from_previous=False,
        safe_hold=False,
        failure_reason=None,
        exact_public_target_reachable=True,
        exact_public_target_unreachable=False,
        proxy_attempted=False,
    )
