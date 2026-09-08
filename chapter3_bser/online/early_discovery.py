"""Optional time-discount ranking over unchanged conditional BSER gains."""
from __future__ import annotations

import math

import numpy as np

from chapter3_bser.objective import evaluate_objective, marginal_gain
from chapter3_bser.online.allocator import BSEROnlineAllocator
from chapter3_bser.types import SolverResult


def resolve_early_discovery(config=None):
    result = {"enabled": False, "lambda": 0.0, "nominal_speed": 1.0, **dict(config or {})}
    if type(result["enabled"]) is not bool:
        raise ValueError("early_discovery.enabled must be boolean")
    for key in ("lambda", "nominal_speed"):
        result[key] = float(result[key])
        if not math.isfinite(result[key]) or result[key] < 0 or (key == "nominal_speed" and result[key] == 0):
            raise ValueError(f"invalid early_discovery.{key}")
    unknown = set(result)-{"enabled", "lambda", "nominal_speed"}
    if unknown:
        raise ValueError(f"unknown early_discovery options: {sorted(unknown)}")
    return result


def estimated_arrival_time(candidate, searcher_position, nominal_speed):
    """Seconds: existing A* length in metres / configured nominal metres/sec."""
    speed = float(nominal_speed)
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("nominal_speed must be positive and finite")
    length = getattr(candidate, "path_length", None)
    if length is None:
        delta = np.asarray(candidate.waypoint, float)-np.asarray(searcher_position, float)
        if delta.shape != (3,):
            raise ValueError("candidate and searcher positions must be 3D")
        length = np.linalg.norm(delta)
    length = float(length)
    if not math.isfinite(length) or length < 0:
        raise ValueError("candidate navigation distance must be finite and nonnegative")
    return length/speed


def rank_candidates(candidates, scores, positions, config):
    """Return candidate/diagnostic pairs. Ties use the frozen solver's key."""
    settings = resolve_early_discovery(config)
    active = settings["enabled"] and settings["lambda"] > 0
    rows = []
    for candidate in candidates:
        original = scores[candidate.key]
        if not math.isfinite(original):
            raise ValueError("BSER gain must be finite")
        arrival = estimated_arrival_time(candidate, positions[candidate.agent_id], settings["nominal_speed"]) if active else None
        discount = math.exp(-settings["lambda"]*arrival) if active else 1.0
        # Identity branch preserves even signed zero; no disabled score arithmetic.
        final = original*discount if active else original
        rows.append((candidate, dict(candidate_id=candidate.candidate_id, agent_id=candidate.agent_id,
                                    original_bser_score=original, estimated_arrival_time=arrival,
                                    time_discount=discount, final_score=final)))
    return sorted(rows, key=lambda row: (-row[1]["final_score"], row[0].key))


class EarlyDiscoveryAllocator(BSEROnlineAllocator):
    """Keep baseline standby and original objective/acceptance; rerank search."""
    def __init__(self, config=None):
        super().__init__()
        self.early_config = resolve_early_discovery(config)
        self.candidate_diagnostics = []
        self.ranking_diagnostics = []

    def _solve_candidates(self, candidates, standby_candidates, context):
        baseline = super()._solve_candidates(candidates, standby_candidates, context)
        config = self.early_config
        if not config["enabled"] or config["lambda"] == 0 or context.state.target_found or baseline.standby is None:
            return baseline
        positions = {agent.agent_id: agent.position for agent in context.state.agents}
        selected, used_agents = [], set()
        round_index = 0
        while True:
            feasible = [c for c in candidates if c.agent_id not in used_agents]
            gains = {c.key: marginal_gain(selected, c, baseline.standby, context) for c in feasible}
            feasible = [c for c in feasible if gains[c.key] > 1e-15]
            if not feasible:
                break
            ranked = rank_candidates(feasible, gains, positions, config)
            old_top = min(feasible, key=lambda c: (-gains[c.key], c.key))
            new_top = ranked[0][0]
            common = dict(step=int(context.state.step), ranking_round=round_index,
                          decision_index=len(self.ranking_diagnostics))
            self.candidate_diagnostics.extend(dict(record, **common) for _, record in ranked)
            self.ranking_diagnostics.append(dict(common, candidate_count=len(ranked),
                                                mean_time_discount=sum(r[1]["time_discount"] for r in ranked)/len(ranked),
                                                top_candidate_changed=new_top.key != old_top.key))
            selected.append(new_top)
            used_agents.add(new_top.agent_id)
            round_index += 1
        selected.sort(key=lambda c: c.key)
        return SolverResult(baseline.solver, tuple(selected), baseline.standby,
                            evaluate_objective(selected, baseline.standby, context))
