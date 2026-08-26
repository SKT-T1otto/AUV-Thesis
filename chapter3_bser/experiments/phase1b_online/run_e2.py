"""Run BSER-E2 online reallocation and fixed event ablations without training."""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import replace
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
from chapter3_bser.controllers.state_provider import OnlinePlanningStateProvider
from chapter3_bser.online.allocator import BSEROnlineAllocator
from chapter3_bser.online.config import load_phase1b_config
from chapter3_bser.online.controller import OnlineBSERController
from chapter3_bser.online.waypoint_manager import WaypointManager
from core.config.ch3_config import build_ch3_config
from core.env import MissionCoreEnv, environment_kwargs_from_config
from core.mapping.planning_state import planning_state_sha256


ROOT = Path(__file__).resolve().parents[3]
SCENARIO_PATH = ROOT / "configs" / "scenarios" / "e0_equivalence" / "M20_MOVING_UNKNOWN_MULTI.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "chapter3" / "phase1b_online"


def _seed_all(seed: int) -> None:
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def _write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in values for key in row}) if values else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(values)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _mean(values):
    finite = [float(value) for value in values if value not in (None, "") and math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else ""


class MemoizedAllocator(BSEROnlineAllocator):
    def __init__(self):
        super().__init__(); self._cache = {}

    def allocate(self, state, *, trigger_reason="online"):
        key = planning_state_sha256(state)
        cached = self._cache.get(key)
        if cached is None:
            cached = super().allocate(state, trigger_reason=trigger_reason)
            self._cache[key] = cached
        return replace(cached, trigger_reason=str(trigger_reason))


_WORKER_ALLOCATOR = None


def _episode_worker(arguments):
    global _WORKER_ALLOCATOR
    if _WORKER_ALLOCATOR is None:
        torch.set_num_threads(1)
        _WORKER_ALLOCATOR = MemoizedAllocator()
    condition, episode_index, scenario, config, max_steps = arguments
    return _episode(condition, episode_index, scenario, config, _WORKER_ALLOCATOR, max_steps)


def _condition_config(base: dict, condition: str) -> dict:
    config = copy.deepcopy(base)
    if condition == "Event-BSER-no-belief": config["events"]["enable_belief_trigger"] = False
    if condition == "Event-BSER-no-obstacle": config["events"]["enable_obstacle_trigger"] = False
    if condition == "Event-BSER-no-target": config["events"]["enable_target_trigger"] = False
    if condition == "Event-BSER-no-hysteresis": config["hysteresis"]["enabled"] = False
    return config


def _dither(actions: np.ndarray, step: int, episode_index: int) -> np.ndarray:
    agent = np.arange(actions.shape[0], dtype=np.float64)[:, None]
    axis = np.arange(3, dtype=np.float64)[None, :]
    perturbation = 0.02 * np.sin(0.17 * step + 0.31 * episode_index + 0.43 * agent + 0.71 * axis)
    return np.clip(actions + perturbation.astype(np.float32), -1.0, 1.0)


def _episode(condition: str, episode_index: int, scenario: dict, config: dict, allocator: MemoizedAllocator, max_steps: int):
    scenario_seed = int(scenario["scenario_seed"])
    _seed_all(scenario_seed)
    env_config = build_ch3_config("ch3_v3_full_reference", "M20_MOVING_UNKNOWN_MULTI")
    env = MissionCoreEnv(**environment_kwargs_from_config(env_config, device="cpu", max_steps=max_steps, return_numpy=False))
    events, replans = [], []
    started = time.perf_counter()
    try:
        env.reset(scenario=copy.deepcopy(scenario))
        refresh = 0 if condition == "No-BSER-static" else int(config["online"]["state_refresh_interval"])
        provider = OnlinePlanningStateProvider(env, refresh_interval=refresh)
        state = provider.initialize()
        controller = None
        if condition.startswith("Event-BSER"):
            controller = OnlineBSERController(_condition_config(config, condition), allocator)
            initialized = controller.initialize(state)
            allocation = initialized.allocation
            waypoint_switches = len(initialized.waypoint_updates)
        else:
            allocation = allocator.allocate(state, trigger_reason="INITIALIZE")
            waypoint_switches = len(allocation.search_assignments)
        initial_hash = allocation.allocation_sha256
        replan_steps = []
        target_found_step = None; completion_step = None; executor_arrival_step = None
        executor_knew_target = False
        final_step = 0
        for step in range(max_steps):
            actions = assignment_to_fixed_actions(state, allocation)
            actions = _dither(actions, step, episode_index)
            _, _, done = env.step(torch.as_tensor(actions, dtype=torch.float32))
            task = env.get_task_state(); final_step = int(task.step)
            if task.target_found and target_found_step is None: target_found_step = int(task.step)
            executor_knew_target = executor_knew_target or bool(task.executor_knows_target)
            force = condition == "Periodic-BSER" and task.step % int(config["experiment"]["periodic_replan_steps"]) == 0
            state = provider.snapshot(force=force)
            if controller is not None:
                result = controller.step(state)
                for event in result.events:
                    events.append({"condition":condition,"episode_index":episode_index,"scenario_seed":scenario_seed,"step":task.step,"event":event.value,"replanned":result.replanned,"decision_reason":result.decision_reason,"belief_shift_score":result.event_detection.belief_shift_score,"new_obstacle_cells":result.event_detection.new_obstacle_cells,"risk_change":result.event_detection.risk_change})
                if result.replanned:
                    allocation = result.allocation; waypoint_switches += len(result.waypoint_updates); replan_steps.append(int(task.step))
                    replans.append({"condition":condition,"episode_index":episode_index,"scenario_seed":scenario_seed,"step":task.step,"reason":result.decision_reason,"allocation_sha256":allocation.allocation_sha256,"objective":allocation.objective_value})
            elif condition == "Periodic-BSER" and force:
                previous = allocation; allocation = allocator.allocate(state, trigger_reason="PERIODIC_20")
                updates = WaypointManager().updates(previous, allocation, reason="PERIODIC_20", step=task.step)
                waypoint_switches += len(updates); replan_steps.append(int(task.step))
                replans.append({"condition":condition,"episode_index":episode_index,"scenario_seed":scenario_seed,"step":task.step,"reason":"PERIODIC_20","allocation_sha256":allocation.allocation_sha256,"objective":allocation.objective_value})
            if target_found_step is not None and executor_arrival_step is None:
                executor = state.agents[state.executor_id]
                distance = float(np.linalg.norm(np.asarray(executor.position) - np.asarray(allocation.executor_assignment.target_region)))
                if distance <= 1.0: executor_arrival_step = int(task.step)
            if task.mission_complete:
                completion_step = int(task.completion_step or task.step); break
            if all(bool(value) for value in done): break
        intervals = np.diff([0] + replan_steps).tolist() if replan_steps else []
        metric = {
            "condition": condition, "episode_index": episode_index, "scenario_id": scenario["scenario_id"], "scenario_seed": scenario_seed,
            "max_steps": max_steps, "steps_completed": final_step, "success": completion_step is not None,
            "completion_time": "" if completion_step is None else completion_step,
            "target_found_time": "" if target_found_step is None else target_found_step,
            "executor_arrival_time": "" if target_found_step is None or executor_arrival_step is None else executor_arrival_step-target_found_step,
            "failed_handoff_count": int(target_found_step is not None and not executor_knew_target),
            "replan_count": len(replan_steps), "average_replanning_interval": _mean(intervals), "waypoint_switches": waypoint_switches,
            "initial_allocation_sha256": initial_hash, "final_allocation_sha256": allocation.allocation_sha256,
            "runtime_seconds": time.perf_counter()-started, "formal_training_run": False, "status": "OK",
        }
        return metric, events, replans
    finally:
        env.close()


def _aggregate(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for condition in sorted({row["condition"] for row in metrics}):
        group = [row for row in metrics if row["condition"] == condition]
        output.append({
            "condition": condition, "episode_count": len(group), "success_count": sum(bool(row["success"]) for row in group),
            "success_rate": sum(bool(row["success"]) for row in group)/max(len(group),1),
            "mean_completion_time_success_only": _mean(row["completion_time"] for row in group),
            "mean_target_found_time_found_only": _mean(row["target_found_time"] for row in group),
            "mean_executor_arrival_time_observed_only": _mean(row["executor_arrival_time"] for row in group),
            "total_replans": sum(int(row["replan_count"]) for row in group), "mean_replans": _mean(row["replan_count"] for row in group),
            "mean_replanning_interval": _mean(row["average_replanning_interval"] for row in group),
            "total_waypoint_switches": sum(int(row["waypoint_switches"]) for row in group),
            "failed_handoff_count": sum(int(row["failed_handoff_count"]) for row in group),
        })
    return output


def run(output_dir: Path = DEFAULT_OUTPUT, *, smoke: bool = False, workers: int = 4) -> dict[str, Any]:
    config = load_phase1b_config(); output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))
    allowed = set(int(value) for value in config["experiment"]["scenario_seeds"])
    scenarios = [item for item in manifest["scenarios"] if int(item["scenario_seed"]) in allowed]
    if len(scenarios) != 5: raise RuntimeError("Phase 1B requires exactly seeds 1729-1733")
    primary = list(config["experiment"]["methods"]); ablations = list(config["experiment"]["ablations"])
    conditions = primary + ablations
    max_steps = 40 if smoke else int(config["experiment"]["max_steps"])
    counts = {condition: 1 for condition in conditions} if smoke else {
        condition: int(config["experiment"]["condition_episode_counts"][condition]) for condition in conditions
    }
    if not smoke and sum(counts.values()) != int(config["experiment"]["total_condition_episodes"]):
        raise RuntimeError("Phase 1B formal protocol must contain exactly 100 condition-episodes")
    jobs = [
        (condition, episode_index, scenarios[episode_index % len(scenarios)], config, max_steps)
        for condition in conditions for episode_index in range(counts[condition])
    ]
    metrics=[]; events=[]; replans=[]; failures=[]
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 4))) as executor:
        future_map = {executor.submit(_episode_worker, job): job for job in jobs}
        for future in as_completed(future_map):
            condition, episode_index, scenario, _, _ = future_map[future]
            try:
                metric, event_rows, replan_rows = future.result()
                metrics.append(metric); events.extend(event_rows); replans.extend(replan_rows)
            except Exception as exc:
                failures.append({"condition":condition,"episode_index":episode_index,"scenario_seed":scenario["scenario_seed"],"error_type":type(exc).__name__,"message":str(exc)})
    order = {condition:index for index,condition in enumerate(conditions)}
    metrics.sort(key=lambda row:(order[row["condition"]],int(row["episode_index"])))
    events.sort(key=lambda row:(order[row["condition"]],int(row["episode_index"]),int(row["step"]),row["event"]))
    replans.sort(key=lambda row:(order[row["condition"]],int(row["episode_index"]),int(row["step"])))
    failures.sort(key=lambda row:(order[row["condition"]],int(row["episode_index"])))
    aggregate = _aggregate(metrics)
    _write_csv(output/"episode_metrics.csv",metrics); _write_csv(output/"event_log.csv",events); _write_csv(output/"replan_log.csv",replans)
    _write_csv(output/"aggregate_by_method.csv",[row for row in aggregate if row["condition"] in primary])
    _write_csv(output/"ablation_summary.csv",[row for row in aggregate if row["condition"] in ablations])
    _write_csv(output/"failure_cases.csv",failures)
    planned = sum(counts.values())
    protocol = {"schema":"bser.e2.experiment.v1","profile":"M20_MOVING_UNKNOWN_MULTI","scenario_seeds":sorted(allowed),"conditions":conditions,"condition_episode_counts":counts,"planned_condition_episode_count":planned,"max_steps":max_steps,"workers":max(1,min(int(workers),4)),"formal_training":False,"oracle_target_used":False,"thresholds_tuned_from_results":False,"smoke":smoke}
    _write_json(output/"experiment_manifest.json",protocol)
    core = {"schema":"bser.e2.summary.v1","primary_method_count":len(primary),"ablation_count":len(ablations),"condition_episode_counts":counts,"planned_condition_episode_count":planned,"completed_condition_episode_count":len(metrics),"failure_count":len(failures),"all_episodes_recorded":len(metrics)+len(failures)==planned,"formal_training_run":False,"oracle_target_used":False,"passed":not failures and len(metrics)==planned}
    stable = hashlib.sha256(json.dumps(core,sort_keys=True,separators=(",",":")).encode()).hexdigest(); summary={**core,"deterministic_summary_sha256":stable}
    _write_json(output/"e2_summary.json",summary)
    (output/"e2_summary.md").write_text(f"# BSER-E2 online summary\n\nStatus: **{'PASS' if summary['passed'] else 'FAIL'}**\n\nCompleted condition-episodes: {len(metrics)}/{planned}. Failures: {len(failures)}. No formal training or oracle target was used.\n",encoding="utf-8")
    print(json.dumps(summary,sort_keys=True)); return summary


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT); parser.add_argument("--smoke",action="store_true"); parser.add_argument("--workers",type=int,default=4); args=parser.parse_args()
    result=run(args.output_dir,smoke=args.smoke,workers=args.workers); raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__": main()
