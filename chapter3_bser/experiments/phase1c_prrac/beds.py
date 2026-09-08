"""Evaluation-only BEDS adapter. No truth target, policy or training changes."""
from dataclasses import replace
import csv
import json
import os

import numpy as np

from chapter3_bser.online.early_discovery import resolve_early_discovery
from chapter3_bser.online.executor_standby import (
    resolve_executor_standby, compute_standby_target, apply_standby_action,
)


DIAGNOSTIC_FILES = {
    "early_discovery_diagnostics.csv": "scenario_id episode step early_discovery_enabled candidate_count mean_time_discount top_candidate_changed action_applied",
    "early_discovery_candidates.csv": "scenario_id episode step decision_index ranking_round agent_id candidate_id original_bser_score estimated_arrival_time time_discount final_score",
    "executor_standby_diagnostics.csv": "scenario_id episode step executor_position standby_target standby_distance three_searcher_mean_distance executor_action_norm weights_source",
}
IDENTITY_FIELDS = "checkpoint checkpoint_episode evaluation_mode execution_variant search_recovery_variant".split()


def write_diagnostics(output, name, rows):
    """Atomically write stable headers; dedupe replayed combinations on resume."""
    fields = DIAGNOSTIC_FILES[name].split()+IDENTITY_FIELDS
    seen, unique = set(), []
    for row in rows:
        key = tuple(str(row.get(k, "")) for k in IDENTITY_FIELDS+["episode", "step", "decision_index", "ranking_round", "agent_id", "candidate_id"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    path = output/name
    temporary = path.with_name("."+name+".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in unique:
            writer.writerow({key: "NA" if row.get(key) is None else json.dumps(row[key], separators=(",", ":"))
                             if isinstance(row[key], (dict, list, tuple)) else row[key] for key in fields})
    os.replace(temporary, path)


def resolve_beds(config):
    return dict(early_discovery=resolve_early_discovery(config.get("early_discovery")),
                executor_standby=resolve_executor_standby(config.get("executor_standby")))


def beds_enabled(config):
    settings = resolve_beds(config)
    return any(value["enabled"] for value in settings.values())


def validate_beds(config):
    settings = resolve_beds(config)
    if not beds_enabled(config):
        return settings
    if config.get("runtime_integration_mode") != "native" or config.get("checkpoint_runtime_revision") != "dynamic_public_intercept_v3_atomic_continuity":
        raise ValueError("BEDS requires native B1 checkpoint/runtime")
    if config.get("modes") != ["full_prrac"] or config.get("execution_variants") != ["B1_ATOMIC_LAST_VALID"]:
        raise ValueError("BEDS requires full_prrac + B1")
    recovery = config.get("search_recovery_variants", config.get("search_collision_recovery", {}).get("variants"))
    if recovery != ["S2A1_C2_LOCAL_CONNECTOR"]:
        raise ValueError("BEDS requires C2 recovery")
    if config.get("search_value_guidance", {}).get("enabled", False):
        raise ValueError("BEDS requires SearchValue OFF; combined ranking is not defined")
    return settings


class BEDSEpisodeAdapter:
    def __init__(self, config, scenario_id, episode, allocator=None):
        self.settings = resolve_beds(config)
        self.scenario_id, self.episode = str(scenario_id), int(episode)
        self.allocator = allocator
        self.standby_target = None
        self.rows = {name: [] for name in DIAGNOSTIC_FILES}
        self.last_ranking = 0

    def _record_ranking(self, ranking, step, *, action_applied):
        count = sum(r["candidate_count"] for r in ranking)
        self.rows["early_discovery_diagnostics.csv"].append(dict(
            scenario_id=self.scenario_id, episode=self.episode, step=int(step),
            early_discovery_enabled=self.settings["early_discovery"]["enabled"], candidate_count=count,
            mean_time_discount=sum(r["candidate_count"]*r["mean_time_discount"] for r in ranking)/count if count else None,
            top_candidate_changed=any(r["top_candidate_changed"] for r in ranking), action_applied=action_applied))

    def prepare_guidance(self, guidance, state):
        self.standby_target = None
        if not self.settings["executor_standby"]["enabled"] or state.target_found or guidance.mission_phase != "SEARCH":
            return guidance
        agents = {agent.agent_id: agent for agent in state.agents}
        executor = agents[state.executor_id].position
        # No reliable current per-searcher value is exported by OnlineAllocation.
        # Use explicit equal weights, never relabel gains/belief as those weights.
        target = tuple(compute_standby_target(executor, [agents[i].position for i in state.searcher_ids]))
        self.standby_target = target
        assignments = tuple(replace(item, assignment_id="BEDS_STANDBY", final_waypoint=target,
                                    planned_path=(), tracking_waypoint=target, hold_state=False, reachable=True)
                            if item.agent_id == state.executor_id else item for item in guidance.agent_assignments)
        executor_assignment = replace(guidance.executor_assignment, source="BEDS_STANDBY", target_region=target,
                                      planned_path=(), tracking_waypoint=target, hold_state=False, reachable=True)
        return replace(guidance, agent_assignments=assignments, executor_assignment=executor_assignment)

    def before_action(self, actions, state):
        if state.target_found:
            self.standby_target = None
            return actions
        identity = dict(scenario_id=self.scenario_id, episode=self.episode, step=int(state.step))
        ranking = [] if self.allocator is None else self.allocator.ranking_diagnostics[self.last_ranking:]
        self.last_ranking += len(ranking)
        self._record_ranking(ranking, state.step, action_applied=True)
        if self.standby_target is None:
            return actions
        agents = {agent.agent_id: agent for agent in state.agents}
        position = np.asarray(agents[state.executor_id].position)
        result = apply_standby_action(actions, position, self.standby_target, enabled=True,
                                     gain=self.settings["executor_standby"]["gain"])
        action = result[3].detach().cpu().numpy() if hasattr(result[3], "detach") else result[3]
        self.rows["executor_standby_diagnostics.csv"].append(dict(identity,
            executor_position=position.tolist(), standby_target=list(self.standby_target),
            standby_distance=float(np.linalg.norm(position-self.standby_target)),
            three_searcher_mean_distance=float(np.mean([np.linalg.norm(position-agents[i].position) for i in state.searcher_ids])),
            executor_action_norm=float(np.linalg.norm(action)), weights_source="uniform_no_audited_searcher_values"))
        return result

    def payload(self):
        if self.allocator is not None:
            pending = self.allocator.ranking_diagnostics[self.last_ranking:]
            for step in sorted({r["step"] for r in pending}):
                self._record_ranking([r for r in pending if r["step"] == step], step, action_applied=False)
            self.last_ranking += len(pending)
            self.rows["early_discovery_candidates.csv"] = [dict(r, scenario_id=self.scenario_id, episode=self.episode)
                                                            for r in self.allocator.candidate_diagnostics]
        return self.rows
