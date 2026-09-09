"""Evaluation-only BEDS adapter. No truth target, policy or training changes."""
from dataclasses import replace
import csv
import json
import os

from chapter3_bser.online.early_discovery import resolve_early_discovery
from chapter3_bser.online.executor_standby import (
    resolve_executor_standby,
)
from chapter3_bser.online.safe_executor_standby import SafeStandbyNavigation, StandbyNavigationParameters
from chapter3_bser.experiments.phase1c_prrac.standby_diagnostics import StandbyDiagnostics, STANDBY_FIELDS, FOUND_FIELDS


DIAGNOSTIC_FILES = {
    "early_discovery_diagnostics.csv": "scenario_id episode step early_discovery_enabled candidate_count mean_time_discount top_candidate_changed action_applied",
    "early_discovery_candidates.csv": "scenario_id episode step decision_index ranking_round agent_id candidate_id original_bser_score estimated_arrival_time time_discount final_score",
    "executor_standby_diagnostics.csv": " ".join(STANDBY_FIELDS),
    "executor_standby_found_state.csv": " ".join(FOUND_FIELDS),
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
    standby = resolve_executor_standby(config.get("executor_standby"))
    # Resolve new routing parameters only when opted in. Old disabled settings
    # retain their shape and the existing early-only mathematical path.
    if standby["enabled"]:
        from core.config.ch3_config import build_ch3_config
        from chapter3_bser.online.config import execution_runtime_config
        environment = build_ch3_config(config.get("base_candidate", "ch3_v3_full_reference"),
                                       config.get("profile", "M20_MOVING_UNKNOWN_MULTI"))
        standby.setdefault("update_interval", int(environment["pse_standby_update_interval"]))
        standby.setdefault("target_shift_threshold", float(execution_runtime_config(config)["public_target_update_distance"]))
        standby = resolve_executor_standby(standby)
    return dict(early_discovery=resolve_early_discovery(config.get("early_discovery")), executor_standby=standby)


def beds_requested(config):
    return beds_enabled(config) or bool(config.get("standby_diagnostics_enabled", False))


def beds_enabled(config):
    settings = resolve_beds(config)
    return any(value["enabled"] for value in settings.values())


def validate_beds(config):
    if type(config.get("standby_diagnostics_enabled", False)) is not bool:
        raise ValueError("standby_diagnostics_enabled must be boolean")
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
        self.navigation = None
        self.standby_diagnostics = StandbyDiagnostics(scenario_id, episode,
            config.get("standby_diagnostics_enabled", self.settings["executor_standby"]["enabled"]))

    def bind_navigation(self, env, path_tracking_threshold):
        if not self.settings["executor_standby"]["enabled"]:
            return
        from core.mapping.path_planner import OnlineUnknownMapTaskPlanner
        from core.mapping.planning_state import extract_planning_state
        runtime = env.unwrapped
        planner = runtime.map_module
        if not isinstance(planner, OnlineUnknownMapTaskPlanner):
            raise ValueError("safe BEDS standby requires the live online unknown-map planner")
        if not runtime.use_residual_prior or float(runtime._prior_strength[3]) <= 0:
            raise ValueError("safe standby requires the existing Executor waypoint prior")
        settings = self.settings["executor_standby"]
        parameters = StandbyNavigationParameters(settings["update_interval"], settings["target_shift_threshold"],
            float(path_tracking_threshold), float(runtime.safe_dist), float(runtime.prior_slow_radius_xy),
            float(runtime.prior_slow_radius_z), float(runtime.executor_hold_radius))
        # Both callbacks are read-only existing planner APIs. They never use
        # env.is_inside_obstacle or the true target. Do not refresh the BSER
        # provider: its cadence and early-only input must remain unchanged.
        self.navigation = SafeStandbyNavigation(parameters, state_factory=lambda: extract_planning_state(env),
            segment_clear=lambda start, end: planner._segment_is_free_np(start, end, planner.planner_obstacle_clearance))

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
            if self.navigation is not None:
                self.navigation.clear()
            return guidance
        agents = {agent.agent_id: agent for agent in state.agents}
        executor = agents[state.executor_id].position
        if self.navigation is None:
            raise RuntimeError("enabled safe standby requires navigation binding")
        self.navigation.prepare(state)
        target = self.navigation.safe_target or tuple(executor)
        self.standby_target = self.navigation.safe_target
        path, tracking, hold = self.navigation.path, self.navigation.tracking_target, self.navigation.hold
        assignments = tuple(replace(item, assignment_id="BEDS_SAFE_STANDBY", final_waypoint=target,
                                    planned_path=path, tracking_waypoint=tracking, hold_position=tuple(executor),
                                    hold_state=hold, reachable=bool(path))
                            if item.agent_id == state.executor_id else item for item in guidance.agent_assignments)
        executor_assignment = replace(guidance.executor_assignment, source="BEDS_SAFE_STANDBY", target_region=target,
                                      planned_path=path, tracking_waypoint=tracking, hold_position=tuple(executor),
                                      hold_state=hold, reachable=bool(path))
        return replace(guidance, agent_assignments=assignments, executor_assignment=executor_assignment)

    def before_action(self, actions, state, c2_active=None):
        if state.target_found:
            self.standby_target = None
            if self.navigation is not None:
                self.navigation.clear()
            self.standby_diagnostics.before_action(state, actions, self.navigation, c2_active)
            return actions
        ranking = [] if self.allocator is None else self.allocator.ranking_diagnostics[self.last_ranking:]
        self.last_ranking += len(ranking)
        self._record_ranking(ranking, state.step, action_applied=True)
        result = actions if self.navigation is None else self.navigation.apply_residual_safety(actions, state)
        self.standby_diagnostics.before_action(state, result, self.navigation, c2_active)
        return result

    def observe_transition(self, before, after, actions, collision_flags, distance_at_found=None):
        self.standby_diagnostics.observe_transition(before, after, actions, collision_flags, distance_at_found)
        if after.target_found:
            self.standby_target = None
            if self.navigation is not None:
                self.navigation.clear()

    def payload(self):
        self.rows["executor_standby_diagnostics.csv"] = self.standby_diagnostics.rows
        self.rows["executor_standby_found_state.csv"] = self.standby_diagnostics.found_rows
        if self.allocator is not None:
            pending = self.allocator.ranking_diagnostics[self.last_ranking:]
            for step in sorted({r["step"] for r in pending}):
                self._record_ranking([r for r in pending if r["step"] == step], step, action_applied=False)
            self.last_ranking += len(pending)
            self.rows["early_discovery_candidates.csv"] = [dict(r, scenario_id=self.scenario_id, episode=self.episode)
                                                            for r in self.allocator.candidate_diagnostics]
        return self.rows
