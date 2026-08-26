"""Run the preregistered 80-condition-episode BSER Phase 1B.1 pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any

import numpy as np
import torch

from chapter3_bser.controllers.action_adapter import assignment_to_fixed_actions
from chapter3_bser.controllers.path_tracker import PathTracker
from chapter3_bser.controllers.state_provider import OnlinePlanningStateProvider
from chapter3_bser.online.allocator import BSEROnlineAllocator
from chapter3_bser.online.config import (
    load_phase1b1_config,
    load_phase1b2_config,
    load_phase1b_config,
)
from chapter3_bser.online.controller import OnlineBSERController
from chapter3_bser.online.mission_context import OnlineMissionContext
from chapter3_bser.online.waypoint_manager import WaypointManager
from core.config.ch3_config import build_ch3_config
from core.env import MissionCoreEnv, environment_kwargs_from_config
from core.mapping.planning_state import planning_state_sha256
from core.scenarios.ch3_generator_impl import build_scenario_manifests


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT / "experiments" / "chapter3" / "phase1b1_pilot"
METHODS = (
    "No-BSER-static",
    "Periodic-BSER",
    "Event-BSER-phase1b_v1",
    "Event-BSER-phase1b1_corrected",
)


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    names = fields or sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values) -> float:
    finite = [float(value) for value in values if value not in (None, "") and math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else 0.0


def _dither(actions: np.ndarray, step: int, episode_index: int) -> np.ndarray:
    agent = np.arange(actions.shape[0], dtype=np.float64)[:, None]
    axis = np.arange(3, dtype=np.float64)[None, :]
    perturbation = 0.02 * np.sin(0.17 * step + 0.31 * episode_index + 0.43 * agent + 0.71 * axis)
    return np.clip(actions + perturbation.astype(np.float32), -1.0, 1.0)


class MemoizedAllocator(BSEROnlineAllocator):
    def __init__(self):
        super().__init__()
        self._cache = {}

    def allocate(self, state, *, trigger_reason="online"):
        key = planning_state_sha256(state)
        cached = self._cache.get(key)
        if cached is None:
            cached = super().allocate(state, trigger_reason=trigger_reason)
            self._cache[key] = cached
        return replace(cached, trigger_reason=str(trigger_reason))


def _public_context(env: MissionCoreEnv, state) -> OnlineMissionContext:
    return OnlineMissionContext.from_public_views(
        env.get_task_state(), env.get_search_execution_state(), state
    )


def _serial_step(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    for key in (
        "detected_events",
        "affected_agent_ids",
        "old_waypoints",
        "proposed_waypoints",
        "switch_distance_by_agent",
    ):
        if key in output:
            output[key] = json.dumps(output[key], separators=(",", ":"))
    return output


def _config_for_method(method: str):
    if method == "Event-BSER-phase1b2_corrected":
        return load_phase1b2_config()
    if method in {"Event-BSER-phase1b1", "Event-BSER-phase1b1_corrected"}:
        return load_phase1b1_config()
    if method.startswith("Event-BSER"):
        return load_phase1b_config()
    return None


def _state_refresh_interval(method: str, config) -> int:
    if method == "No-BSER-static":
        return 0
    if method == "Periodic-BSER":
        return int(load_phase1b2_config()["experiment"]["periodic_replan_steps"])
    if config is None:
        raise ValueError(f"event method has no configuration: {method}")
    return int(config["online"]["state_refresh_interval"])


def _periodic_replan_due(method: str, step: int, interval: int) -> bool:
    return bool(
        method == "Periodic-BSER"
        and int(interval) > 0
        and int(step) % int(interval) == 0
    )


def _episode(
    method: str,
    scenario: dict,
    episode_index: int,
    max_steps: int,
    execution_consistent: bool = False,
    allocator: MemoizedAllocator | None = None,
    diagnostic_recorder: Any | None = None,
):
    scenario_seed = int(scenario["scenario_seed"])
    _seed_all(scenario_seed)
    env_config = build_ch3_config("ch3_v3_full_reference", "M20_MOVING_UNKNOWN_MULTI")
    env = MissionCoreEnv(**environment_kwargs_from_config(env_config, device="cpu", max_steps=max_steps, return_numpy=False))
    allocator = allocator or MemoizedAllocator()
    started = time.perf_counter()
    step_rows: list[dict[str, Any]] = []
    try:
        env.reset(scenario=copy.deepcopy(scenario))
        corrected_v2 = method == "Event-BSER-phase1b2_corrected"
        corrected_v1 = method in {
            "Event-BSER-phase1b1",
            "Event-BSER-phase1b1_corrected",
        }
        corrected = corrected_v1 or corrected_v2
        event_method = method.startswith("Event-BSER")
        config = _config_for_method(method)
        refresh = _state_refresh_interval(method, config)
        periodic_replan_steps = int(
            load_phase1b2_config()["experiment"]["periodic_replan_steps"]
        )
        provider = OnlinePlanningStateProvider(env, refresh_interval=refresh)
        state = provider.initialize()
        controller = None
        periodic_waypoints = WaypointManager()
        path_tracker = (
            PathTracker(
                threshold=float(
                    load_phase1b2_config()["execution"]["path_tracking_threshold"]
                )
            )
            if execution_consistent
            else None
        )
        if event_method:
            if config is None:
                raise RuntimeError(f"missing event configuration for {method}")
            controller = OnlineBSERController(config, allocator)
            context = _public_context(env, state) if corrected else None
            allocation = controller.initialize(state, context).allocation
        else:
            allocation = allocator.allocate(state, trigger_reason="INITIALIZE")
        if diagnostic_recorder is not None:
            if not corrected_v2 or path_tracker is None:
                raise ValueError("Phase 1B.3A diagnostics require corrected v2 path tracking")
            diagnostic_recorder.initialize(state, allocation)
        target_found_step = None
        completion_step = None
        executor_arrival_step = None
        event_counts: Counter[str] = Counter()
        reject_counts: Counter[str] = Counter()
        route_sources: Counter[str] = Counter()
        accepted_replans = 0
        rejected_replans = 0
        optimizer_invocations = 0
        replan_steps: list[int] = []
        waypoint_switch_count = 0
        total_switch_distance = 0.0
        maximum_switch_distance = 0.0
        same_agent_switch_within_10_steps = 0
        last_switch_step: dict[int, int] = {}
        obstacle_events = 0
        impacted_obstacle_events = 0
        collision_count = 0
        path_tracking_errors: list[float] = []
        partial_reallocation_attempts = 0
        partial_reallocation_successes = 0
        final_step = 0
        for action_step in range(max_steps):
            actions = _dither(
                assignment_to_fixed_actions(
                    state,
                    allocation,
                    path_tracker=path_tracker,
                ),
                action_step,
                episode_index,
            )
            _, _, done = env.step(torch.as_tensor(actions, dtype=torch.float32))
            task = env.get_task_state()
            public_agents = env.get_agent_state()
            collision_agent_ids = tuple(
                index for index, value in enumerate(public_agents.collision_flags) if bool(value)
            )
            collision_count += len(collision_agent_ids)
            final_step = int(task.step)
            if task.target_found and target_found_step is None:
                target_found_step = int(task.step)
            periodic_replan_due = _periodic_replan_due(
                method,
                task.step,
                periodic_replan_steps,
            )
            state = provider.snapshot(force=periodic_replan_due)
            if path_tracker is not None:
                path_tracking_errors.append(
                    path_tracker.mean_tracking_error(
                        {agent.agent_id: agent.position for agent in state.agents}
                    )
                )
            before = allocation
            events = ()
            accepted = False
            reason = "NO_REPLAN_EVENT"
            optimizer = False
            scope = "none"
            affected = ()
            route_impacted = False
            updates = ()
            executor_source = allocation.executor_assignment.source
            old_objective = float(allocation.objective_value)
            proposed_objective = old_objective
            if controller is not None:
                step_context = _public_context(env, state) if corrected else None
                result = controller.step(state, step_context)
                allocation = result.allocation
                events = result.events
                accepted = bool(result.replanned)
                reason = result.decision_reason
                updates = result.waypoint_updates
                if result.diagnostics is not None:
                    diagnostic = asdict(result.diagnostics)
                    optimizer = bool(diagnostic["optimizer_invoked"])
                    scope = diagnostic["allocation_scope"]
                    affected = tuple(diagnostic["affected_agent_ids"])
                    route_impacted = bool(diagnostic["obstacle_route_impacted"])
                    executor_source = diagnostic["executor_target_source"]
                    old_objective = float(diagnostic["old_objective"])
                    proposed_objective = float(diagnostic["proposed_objective"])
                    step_row = diagnostic
                else:
                    optimizer = bool(events and reason not in {"cooldown_active", "no_event"})
                    scope = "legacy_full" if optimizer else "none"
                    affected = tuple(state.searcher_ids) if optimizer else ()
                    executor_source = allocation.executor_assignment.source
                    step_row = {
                        "step": int(task.step), "mechanism_version": "phase1b_v1",
                        "detected_events": tuple(event.value for event in events),
                        "optimizer_invoked": optimizer, "allocation_scope": scope,
                        "old_objective": old_objective, "proposed_objective": float(allocation.objective_value),
                        "objective_gain": float(allocation.objective_value-old_objective), "accepted": accepted,
                        "accept_reason": reason if accepted else "", "reject_reason": "" if accepted else reason,
                        "affected_agent_ids": affected, "obstacle_route_impacted": False,
                        "old_waypoints": tuple((item.agent_id,item.waypoint) for item in before.search_assignments),
                        "proposed_waypoints": tuple((item.agent_id,item.waypoint) for item in allocation.search_assignments),
                        "switch_distance_by_agent": (), "executor_target_source": executor_source,
                    }
                if diagnostic_recorder is not None:
                    diagnostic_recorder.record_step(
                        state=state,
                        context=step_context,
                        before=before,
                        result=result,
                        path_tracker=path_tracker,
                        collision_agent_ids=collision_agent_ids,
                    )
            elif periodic_replan_due:
                optimizer = True
                scope = "periodic_full"
                proposal = allocator.allocate(state, trigger_reason="PERIODIC_20")
                updates = periodic_waypoints.updates(before, proposal, reason="PERIODIC_20", step=task.step)
                allocation = proposal
                accepted = True
                reason = "PERIODIC_20"
                proposed_objective = float(proposal.objective_value)
                affected = tuple(state.searcher_ids) + (state.executor_id,)
                executor_source = allocation.executor_assignment.source
                step_row = {
                    "step":int(task.step),"mechanism_version":"periodic_control","detected_events":(),
                    "optimizer_invoked":True,"allocation_scope":scope,"old_objective":old_objective,
                    "proposed_objective":proposed_objective,"objective_gain":proposed_objective-old_objective,
                    "accepted":True,"accept_reason":reason,"reject_reason":"","affected_agent_ids":affected,
                    "obstacle_route_impacted":False,"old_waypoints":tuple((item.agent_id,item.waypoint) for item in before.search_assignments),
                    "proposed_waypoints":tuple((item.agent_id,item.waypoint) for item in allocation.search_assignments),
                    "switch_distance_by_agent":(),"executor_target_source":executor_source,
                }
            else:
                step_row = {
                    "step":int(task.step),"mechanism_version":"static","detected_events":(),
                    "optimizer_invoked":False,"allocation_scope":"none","old_objective":old_objective,
                    "proposed_objective":old_objective,"objective_gain":0.0,"accepted":False,
                    "accept_reason":"","reject_reason":"NO_REPLAN_EVENT","affected_agent_ids":(),
                    "obstacle_route_impacted":False,"old_waypoints":tuple((item.agent_id,item.waypoint) for item in allocation.search_assignments),
                    "proposed_waypoints":tuple((item.agent_id,item.waypoint) for item in allocation.search_assignments),
                    "switch_distance_by_agent":(),"executor_target_source":executor_source,
                }
            for event in events:
                event_counts[event.value] += 1
                if event.value == "OBSTACLE_DISCOVERED":
                    obstacle_events += 1
            if route_impacted:
                impacted_obstacle_events += 1
            if optimizer and scope in {"affected_agents", "stale_searchers", "executor_only"}:
                partial_reallocation_attempts += 1
                partial_reallocation_successes += int(accepted)
            optimizer_invocations += int(optimizer)
            if accepted:
                accepted_replans += 1
                replan_steps.append(int(task.step))
            elif events or periodic_replan_due or optimizer:
                rejected_replans += 1
                reject_counts[str(reason)] += 1
            switch_distances = []
            for update in updates:
                distance = 0.0 if update.old_waypoint is None else float(np.linalg.norm(np.asarray(update.new_waypoint)-np.asarray(update.old_waypoint)))
                switch_distances.append((update.agent_id, distance))
                waypoint_switch_count += 1
                total_switch_distance += distance
                maximum_switch_distance = max(maximum_switch_distance, distance)
                if update.agent_id in last_switch_step and int(task.step)-last_switch_step[update.agent_id] <= 10:
                    same_agent_switch_within_10_steps += 1
                last_switch_step[update.agent_id] = int(task.step)
            step_row["switch_distance_by_agent"] = tuple(switch_distances)
            step_rows.append(_serial_step({
                "method":method,"scenario_seed":scenario_seed,"episode_index":episode_index,
                **step_row,
            }))
            if target_found_step is not None and (
                events or allocation.executor_assignment.source != before.executor_assignment.source
            ):
                if reason == "WAITING_FOR_PUBLIC_HANDOFF":
                    route_sources[reason] += 1
                else:
                    route_sources[str(executor_source)] += 1
            if target_found_step is not None and executor_arrival_step is None:
                executor = state.agents[state.executor_id]
                distance = float(np.linalg.norm(np.asarray(executor.position)-np.asarray(allocation.executor_assignment.target_region)))
                if distance <= 1.0:
                    executor_arrival_step = int(task.step)
            if task.mission_complete:
                completion_step = int(task.completion_step or task.step)
                break
            if all(bool(value) for value in done):
                break
        intervals = np.diff([0] + replan_steps).tolist() if replan_steps else [max_steps]
        completion_truncated = completion_step if completion_step is not None else max_steps
        found_truncated = target_found_step if target_found_step is not None else max_steps
        if target_found_step is None:
            delay_truncated = max_steps
        elif executor_arrival_step is None:
            delay_truncated = max_steps-target_found_step
        else:
            delay_truncated = executor_arrival_step-target_found_step
        metric = {
            "method":method,"scenario_seed":scenario_seed,"episode_index":episode_index,
            "scenario_id":scenario["scenario_id"],"max_steps":max_steps,"steps_completed":final_step,
            "success":completion_step is not None,"completion_step":"" if completion_step is None else completion_step,
            "found_step":"" if target_found_step is None else target_found_step,
            "executor_arrival_step":"" if executor_arrival_step is None else executor_arrival_step,
            "completion_time_truncated":completion_truncated,"target_found_time_truncated":found_truncated,
            "post_found_executor_delay_truncated":delay_truncated,"optimizer_invocation_count":optimizer_invocations,
            "accepted_replan_count":accepted_replans,"rejected_replan_count":rejected_replans,
            "replan_count":len(replan_steps),"mean_replan_interval":_mean(intervals),
            "waypoint_switch_count":waypoint_switch_count,"total_switch_distance":total_switch_distance,
            "mean_switch_distance":total_switch_distance/waypoint_switch_count if waypoint_switch_count else 0.0,
            "maximum_switch_distance":maximum_switch_distance,
            "same_agent_switch_within_10_steps":same_agent_switch_within_10_steps,
            "obstacle_event_count":obstacle_events,"obstacle_route_impacted_count":impacted_obstacle_events,
            "executor_invalid_count":int(event_counts.get("EXECUTOR_INVALID", 0)),
            "waypoint_stale_count":int(event_counts.get("WAYPOINT_STALE", 0)),
            "collision_count":int(collision_count),
            "path_tracking_error":_mean(path_tracking_errors),
            "partial_reallocation_attempt_count":partial_reallocation_attempts,
            "partial_reallocation_success_count":partial_reallocation_successes,
            "partial_reallocation_success_rate":(
                partial_reallocation_successes / partial_reallocation_attempts
                if partial_reallocation_attempts
                else 0.0
            ),
            "event_counts":json.dumps(dict(sorted(event_counts.items())),sort_keys=True),
            "reject_reason_counts":json.dumps(dict(sorted(reject_counts.items())),sort_keys=True),
            "target_route_source_counts":json.dumps(dict(sorted(route_sources.items())),sort_keys=True),
            "runtime_seconds":time.perf_counter()-started,"status":"OK","formal_training_run":False,
        }
        return metric, step_rows
    finally:
        env.close()


_WORKER_ALLOCATOR = None


def _worker(job):
    global _WORKER_ALLOCATOR
    torch.set_num_threads(1)
    if _WORKER_ALLOCATOR is None:
        _WORKER_ALLOCATOR = MemoizedAllocator()
    return _episode(*job, allocator=_WORKER_ALLOCATOR)


def _seed_worker(job):
    seed, scenario, indices, max_steps = job
    torch.set_num_threads(1)
    allocator = MemoizedAllocator()
    metrics = []
    steps = []
    failures = []
    for episode_index in indices:
        for method in METHODS:
            try:
                metric, diagnostics = _episode(
                    method, scenario, episode_index, max_steps, allocator=allocator
                )
                metrics.append(metric)
                steps.extend(diagnostics)
            except Exception as exc:
                failures.append({
                    "method":method,"scenario_seed":seed,"episode_index":episode_index,
                    "error_type":type(exc).__name__,"message":str(exc),
                })
    return {"scenario_seed":seed,"metrics":metrics,"steps":steps,"failures":failures}


def _summary(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        group = [row for row in metrics if row["method"] == method]
        rows.append({
            "method":method,"episode_count":len(group),
            "success_rate":sum(bool(row["success"]) for row in group)/len(group),
            "mean_completion_success_only":_mean(row["completion_step"] for row in group),
            "mean_truncated_completion":_mean(row["completion_time_truncated"] for row in group),
            "mean_truncated_target_found":_mean(row["target_found_time_truncated"] for row in group),
            "mean_truncated_post_found_executor_delay":_mean(row["post_found_executor_delay_truncated"] for row in group),
            "mean_replans":_mean(row["replan_count"] for row in group),
            "mean_replan_interval":_mean(row["mean_replan_interval"] for row in group),
            "optimizer_invocations":sum(int(row["optimizer_invocation_count"]) for row in group),
            "accepted_replans":sum(int(row["accepted_replan_count"]) for row in group),
            "rejected_replans":sum(int(row["rejected_replan_count"]) for row in group),
            "waypoint_switch_count":sum(int(row["waypoint_switch_count"]) for row in group),
            "total_switch_distance":sum(float(row["total_switch_distance"]) for row in group),
        })
    return rows


def run(output_dir: Path = DEFAULT_OUTPUT, *, smoke: bool = False, workers: int = 4) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = load_phase1b1_config()
    seeds = tuple(int(value) for value in config["experiment"]["scenario_seeds"])
    indices = (0,) if smoke else tuple(int(value) for value in config["experiment"]["episode_indices"])
    manifest = build_scenario_manifests(
        count=5, generator_seed=seeds[0], split="validation", profiles=("M20_MOVING_UNKNOWN_MULTI",)
    )["M20_MOVING_UNKNOWN_MULTI"]
    scenarios = {int(row["scenario_seed"]): row for row in manifest["scenarios"]}
    if tuple(sorted(scenarios)) != seeds:
        raise RuntimeError("pilot scenario generator did not produce seeds 2729-2733")
    max_steps = 5 if smoke else int(config["experiment"]["max_steps"])
    run_seeds = seeds[:1] if smoke else seeds
    planned = len(METHODS)*len(run_seeds)*len(indices)
    metrics: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    checkpoint_dir = output/"_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for seed in run_seeds:
        checkpoint_path = checkpoint_dir/f"seed_{seed}.json"
        checkpoint = None
        if checkpoint_path.is_file():
            candidate = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if (
                candidate.get("schema") == "bser.phase1b1.pilot.seed_checkpoint.v1"
                and int(candidate.get("max_steps", -1)) == max_steps
                and tuple(candidate.get("episode_indices", ())) == indices
                and tuple(candidate.get("methods", ())) == METHODS
            ):
                checkpoint = candidate
        if checkpoint is None:
            pending.append((seed, scenarios[seed], indices, max_steps))
        else:
            metrics.extend(checkpoint["metrics"])
            steps.extend(checkpoint["steps"])
            failures.extend(checkpoint["failures"])
    if pending:
        with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 4, len(pending)))) as executor:
            future_map = {executor.submit(_seed_worker, job): job[0] for job in pending}
            for future in as_completed(future_map):
                seed = future_map[future]
                result = future.result()
                checkpoint = {
                    "schema":"bser.phase1b1.pilot.seed_checkpoint.v1",
                    "scenario_seed":seed,"max_steps":max_steps,
                    "episode_indices":list(indices),"methods":list(METHODS),
                    "metrics":result["metrics"],"steps":result["steps"],"failures":result["failures"],
                }
                _write_json(checkpoint_dir/f"seed_{seed}.json",checkpoint)
                metrics.extend(result["metrics"])
                steps.extend(result["steps"])
                failures.extend(result["failures"])
    order = {method:index for index,method in enumerate(METHODS)}
    metrics.sort(key=lambda row:(order[row["method"]],int(row["scenario_seed"]),int(row["episode_index"])))
    steps.sort(key=lambda row:(order[row["method"]],int(row["scenario_seed"]),int(row["episode_index"]),int(row["step"])))
    failures.sort(key=lambda row:(order[row["method"]],int(row["scenario_seed"]),int(row["episode_index"])))
    method_summary = _summary(metrics) if not failures else []
    by_method = {row["method"]:row for row in method_summary}
    performance_passed = False
    gates = {}
    if not failures and len(metrics) == planned:
        static = by_method["No-BSER-static"]
        old = by_method["Event-BSER-phase1b_v1"]
        corrected = by_method["Event-BSER-phase1b1_corrected"]
        gates = {
            "success_noninferiority": corrected["success_rate"] >= static["success_rate"]-0.05,
            "truncated_completion_noninferiority": corrected["mean_truncated_completion"] <= 1.10*static["mean_truncated_completion"],
            "not_jointly_worse_than_old": not (
                corrected["success_rate"] < old["success_rate"]
                and corrected["mean_truncated_completion"] > old["mean_truncated_completion"]
            ),
            "mean_replans_at_most_20": corrected["mean_replans"] <= 20,
            "mean_replan_interval_at_least_10": corrected["mean_replan_interval"] >= 10,
            "switch_distance_below_old": corrected["total_switch_distance"] < old["total_switch_distance"],
        }
        performance_passed = all(gates.values())
    engineering_passed = not failures and len(metrics) == planned
    status = "PASS_BSER_PHASE1B1_PILOT" if engineering_passed and performance_passed else "PASS_BSER_PHASE1B1_DIAGNOSTIC_ONLY" if engineering_passed else "FAIL_BSER_PHASE1B1"
    event_rows=[]; accept_rows=[]; route_rows=[]; source_rows=[]; waypoint_rows=[]
    for method in METHODS:
        group_steps=[row for row in steps if row["method"]==method]
        events=Counter(event for row in group_steps for event in json.loads(row["detected_events"]))
        for event,count in sorted(events.items()): event_rows.append({"method":method,"event":event,"count":count})
        decisions=Counter(("ACCEPT" if row["accepted"] else "REJECT",row["accept_reason"] or row["reject_reason"]) for row in group_steps if json.loads(row["detected_events"]) or row["optimizer_invoked"])
        for (decision,reason),count in sorted(decisions.items()): accept_rows.append({"method":method,"decision":decision,"reason":reason,"count":count})
        obstacle=[row for row in group_steps if "OBSTACLE_DISCOVERED" in json.loads(row["detected_events"])]
        impacted=sum(bool(row["obstacle_route_impacted"]) for row in obstacle)
        route_rows.append({"method":method,"obstacle_event_count":len(obstacle),"route_impacted_count":impacted,"route_impact_ratio":impacted/len(obstacle) if obstacle else 0.0})
        sources=Counter(row["executor_target_source"] for row in group_steps if row["executor_target_source"])
        for source,count in sorted(sources.items()): source_rows.append({"method":method,"source":source,"count":count})
        summary=by_method.get(method,{})
        waypoint_rows.append({"method":method,"waypoint_switch_count":summary.get("waypoint_switch_count",0),"total_switch_distance":summary.get("total_switch_distance",0.0)})
    protocol={
        "schema":"bser.phase1b1.pilot.experiment.v1","profile":"M20_MOVING_UNKNOWN_MULTI",
        "scenario_seeds":list(run_seeds),"episode_indices":list(indices),"methods":list(METHODS),
        "planned_condition_episode_count":planned,"max_steps":max_steps,"formal_training":False,
        "oracle_access":False,"smoke":smoke,
    }
    snapshot={"phase1b_v1":load_phase1b_config(),"phase1b1_corrected":config}
    snapshot_hash=hashlib.sha256(json.dumps(snapshot,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    summary={
        "schema":"bser.phase1b1.pilot.summary.v1","status":status,
        "planned_condition_episode_count":planned,"completed_condition_episode_count":len(metrics),
        "failure_count":len(failures),"engineering_passed":engineering_passed,
        "performance_gates":gates,"performance_passed":performance_passed,
        "method_summary":method_summary,"config_snapshot_sha256":snapshot_hash,
        "formal_training_run":False,"oracle_access":False,
    }
    _write_json(output/"experiment_manifest.json",protocol)
    _write_json(output/"config_snapshot.json",snapshot)
    _write_csv(output/"episode_metrics.csv",metrics)
    _write_csv(output/"step_diagnostics.csv",steps)
    _write_csv(output/"episode_diagnostics.csv",metrics)
    _write_csv(output/"method_summary.csv",method_summary)
    _write_csv(output/"event_summary.csv",event_rows)
    _write_csv(output/"accept_reject_summary.csv",accept_rows)
    _write_csv(output/"route_impact_summary.csv",route_rows)
    _write_csv(output/"target_route_source_summary.csv",source_rows)
    _write_csv(output/"waypoint_stability_summary.csv",waypoint_rows)
    _write_csv(output/"failure_cases.csv",failures)
    _write_json(output/"pilot_summary.json",summary)
    lines=["# BSER Phase 1B.1 pilot","",f"Status: **{status}**.","",f"Completed {len(metrics)}/{planned} condition-episodes; failures: {len(failures)}.","","| Method | Success | Truncated completion | Replans | Switch distance |","|---|---:|---:|---:|---:|"]
    for row in method_summary:
        lines.append(f"| {row['method']} | {row['success_rate']:.3f} | {row['mean_truncated_completion']:.3f} | {row['mean_replans']:.3f} | {row['total_switch_distance']:.3f} |")
    (output/"pilot_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(summary,sort_keys=True))
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    result=run(args.output_dir,smoke=args.smoke,workers=args.workers)
    raise SystemExit(0 if result["engineering_passed"] else 1)


if __name__ == "__main__":
    main()
