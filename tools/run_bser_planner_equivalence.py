"""Run EQ1 differential checks between PlanningGraphView and the frozen planner."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

import numpy as np
import torch

from chapter3_bser.candidate_generator import generate_candidates
from chapter3_bser.config import load_bser_phase1a1_config
from chapter3_bser.experiments.instance_builder import ENVIRONMENT_MAX_STEPS, PROFILES, SCENARIO_ROOT, TRACES, action_trace
from core.config.ch3_config import build_ch3_config
from core.env import MissionCoreEnv, environment_kwargs_from_config
from core.mapping.planning_state import extract_planning_state, planning_state_sha256
from core.mapping.travel_cost_service import TravelCostService


ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT=ROOT/"experiments/chapter3/bser_planner_equivalence"


def _seed_all(seed):
    random.seed(seed); np.random.seed(seed%(2**32-1)); torch.manual_seed(seed)


def _write_csv(path,rows):
    fields=sorted({key for row in rows for key in row}) if rows else ["status"]
    with Path(path).open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _fixed_targets(state,state_hash,count=16):
    valid=np.flatnonzero(state.planning_graph.valid_mask)
    ordered=sorted((hashlib.sha256(f"{state_hash}|fixed_grid|{int(index)}".encode()).hexdigest(),int(index)) for index in valid)
    return [(f"fixed_grid_{index}",tuple(float(value) for value in state.grid.cell_centers[index]),"fixed_hash_grid") for _,index in ordered[:count]]


def _flatten_cells(cells,shape):
    return tuple(int(np.ravel_multi_index(tuple(int(value) for value in cell),shape)) for cell in cells)


def _compare_query(planner,state,service,source_agent,goal_id,goal,goal_kind,state_hash,profile,snapshot_id):
    new=service.query(source_agent.position,goal,source_agent)
    old=planner.grid_astar_path(source_agent.position,goal,role=source_agent.role)
    repeated_new=service.query(source_agent.position,goal,source_agent)
    repeated_old=planner.grid_astar_path(source_agent.position,goal,role=source_agent.role)
    old_planning=float(old.get("planning_cost",old.get("cost",math.inf)))
    old_physical=float(old.get("travel_time",old.get("cost",math.inf)))
    old_cells=_flatten_cells(old.get("cells",()),state.grid.shape)
    new_cells=tuple(int(value) for value in new.path_cell_indices)
    reachable_mismatch=int(bool(new.reachable)!=bool(old.get("reachable")))
    failure_mismatch=int((new.failure_reason or "")!=(old.get("failure_reason") or "")) if not new.reachable and not old.get("reachable") else 0
    cost_error=abs(new.planning_cost-old_planning) if new.reachable and old.get("reachable") else 0.0
    time_error=abs(new.physical_travel_time-old_physical) if new.reachable and old.get("reachable") else 0.0
    path_mismatch=int(new_cells!=old_cells)
    start_connector_mismatch=int(bool(new_cells) and old.get("resolved_start_cell") is not None and new_cells[0]!=int(np.ravel_multi_index(tuple(old["resolved_start_cell"]),state.grid.shape)))
    goal_connector_mismatch=int(bool(new_cells) and old.get("resolved_goal_cell") is not None and new_cells[-1]!=int(np.ravel_multi_index(tuple(old["resolved_goal_cell"]),state.grid.shape)))
    component_mismatch=0
    if new_cells and old.get("resolved_component_id") is not None:
        component_mismatch=int(int(state.planning_graph.component_labels[new_cells[0]])!=int(old["resolved_component_id"]))
    endpoint_error=0.0
    collision_mismatch=0
    if new.reachable:
        endpoint_error=max(float(np.linalg.norm(new.path_points[0]-np.asarray(source_agent.position))),float(np.linalg.norm(new.path_points[-1]-np.asarray(goal))))
        collision_mismatch=int(any(not planner.segment_is_free(new.path_points[index].copy(),new.path_points[index+1].copy()) for index in range(len(new.path_points)-1)))
    repeat_mismatch=int(
        new.reachable!=repeated_new.reachable or new.planning_cost!=repeated_new.planning_cost or
        new.physical_travel_time!=repeated_new.physical_travel_time or new_cells!=tuple(repeated_new.path_cell_indices) or
        old.get("reachable")!=repeated_old.get("reachable") or old.get("cost")!=repeated_old.get("cost") or
        old.get("points")!=repeated_old.get("points")
    )
    mismatch=any((reachable_mismatch,failure_mismatch,path_mismatch,start_connector_mismatch,goal_connector_mismatch,component_mismatch,collision_mismatch,repeat_mismatch)) or cost_error>1e-6 or time_error>1e-6 or endpoint_error>1e-9
    return {
        "unique_state_sha256":state_hash,"representative_protocol_snapshot_id":snapshot_id,"profile":profile,"knowledge_mode":state.knowledge_mode,
        "source_agent_id":source_agent.agent_id,"source_role":source_agent.role,"goal_id":goal_id,"goal_kind":goal_kind,
        "new_reachable":new.reachable,"original_reachable":bool(old.get("reachable")),"new_failure_reason":new.failure_reason or "","original_failure_reason":old.get("failure_reason") or "","reachable_mismatch":reachable_mismatch,"failure_reason_mismatch":failure_mismatch,
        "planning_cost_absolute_error":cost_error,"physical_time_absolute_error":time_error,"endpoint_absolute_error":endpoint_error,
        "path_cell_sequence_mismatch":path_mismatch,"start_connector_mismatch":start_connector_mismatch,"goal_connector_mismatch":goal_connector_mismatch,
        "component_mismatch":component_mismatch,"collision_validity_mismatch":collision_mismatch,"tie_repeat_mismatch":repeat_mismatch,"mismatch":bool(mismatch),
    }


def run(output_dir=DEFAULT_OUTPUT,*,smoke=False,profiles_override=None,scenario_id_override=None):
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True); config=load_bser_phase1a1_config()
    rows=[]; state_records=[]; seen=set(); protocol_requests=0; valid_requests=0
    profiles=tuple(profiles_override) if profiles_override else (PROFILES[:1] if smoke else PROFILES)
    for profile_index,profile in enumerate(profiles):
        manifest=json.loads((SCENARIO_ROOT/f"{profile}.json").read_text())
        entries=[entry for entry in manifest["scenarios"] if entry["scenario_id"]==scenario_id_override] if scenario_id_override else (manifest["scenarios"][:1] if smoke else manifest["scenarios"])
        env=MissionCoreEnv(**environment_kwargs_from_config(build_ch3_config("ch3_v3_full_reference",profile),device="cpu",max_steps=ENVIRONMENT_MAX_STEPS,return_numpy=False))
        try:
            for entry in entries:
                seed=int(entry["scenario_seed"]); scenario_id=str(entry["scenario_id"])
                for trace_name in (TRACES[:1] if smoke else TRACES):
                    _seed_all(seed); env.reset(scenario=copy.deepcopy(entry)); actions=action_trace(trace_name,seed,profile_index,50); terminated=False
                    for step in range(51):
                        if step in (0,10,25,50):
                            protocol_requests+=1; task=env.get_task_state()
                            if not terminated and not task.target_found and not task.mission_complete:
                                valid_requests+=1; state=extract_planning_state(env); state_hash=planning_state_sha256(state)
                                if state_hash not in seen:
                                    seen.add(state_hash); generation=generate_candidates(state,config); service=TravelCostService(state); planner=env.unwrapped.map_module
                                    goals=[]
                                    for agent in state.agents:
                                        if agent.current_navigation_target is not None: goals.append((f"navigation_target_{agent.agent_id}",agent.current_navigation_target,"navigation_target"))
                                    goals.extend((candidate.candidate_id,candidate.waypoint,"search_candidate") for candidate in generation.search_candidates)
                                    goals.extend((candidate.candidate_id,candidate.waypoint,"standby_candidate") for candidate in generation.standby_candidates)
                                    goals.extend(_fixed_targets(state,state_hash))
                                    snapshot_id=f"{profile}|{scenario_id}|{trace_name}|step_{step:03d}"
                                    for source in state.agents:
                                        for goal_id,goal,goal_kind in goals:
                                            rows.append(_compare_query(planner,state,service,source,goal_id,goal,goal_kind,state_hash,profile,snapshot_id))
                                    state_records.append({"unique_state_sha256":state_hash,"profile":profile,"knowledge_mode":state.knowledge_mode,"query_count":4*len(goals),"planning_graph_sha256":state.planning_graph.graph_sha256})
                        if step==50 or terminated: continue
                        _seed_all(seed*10000+step); _,_,done=env.step(torch.as_tensor(actions[step],dtype=torch.float32)); terminated=all(bool(value) for value in done)
        finally: env.close()
    mismatches=[row for row in rows if row["mismatch"]]
    _write_csv(output/"per_query_comparison.csv",rows); _write_csv(output/"mismatch_cases.csv",mismatches)
    def aggregate(field):
        result=[]
        for value in sorted({row[field] for row in rows}):
            group=[row for row in rows if row[field]==value]
            result.append({field:value,"query_count":len(group),"mismatch_count":sum(row["mismatch"] for row in group),"max_planning_cost_error":max(row["planning_cost_absolute_error"] for row in group),"max_physical_time_error":max(row["physical_time_absolute_error"] for row in group),"max_endpoint_error":max(row["endpoint_absolute_error"] for row in group)})
        return result
    _write_csv(output/"aggregate_by_profile.csv",aggregate("profile")); _write_csv(output/"aggregate_by_knowledge_mode.csv",aggregate("knowledge_mode"))
    summary={
        "schema":"bser.eq1.summary.v1","protocol_request_count":protocol_requests,"valid_protocol_request_count":valid_requests,"unique_state_count":len(seen),"query_count":len(rows),
        "reachability_mismatch_count":sum(row["reachable_mismatch"] for row in rows),"failure_reason_mismatch_count":sum(row["failure_reason_mismatch"] for row in rows),
        "path_sequence_mismatch_count":sum(row["path_cell_sequence_mismatch"] for row in rows),"connector_mismatch_count":sum(row["start_connector_mismatch"]+row["goal_connector_mismatch"] for row in rows),
        "component_mismatch_count":sum(row["component_mismatch"] for row in rows),"collision_validity_mismatch_count":sum(row["collision_validity_mismatch"] for row in rows),"tie_repeat_mismatch_count":sum(row["tie_repeat_mismatch"] for row in rows),
        "max_planning_cost_error":max((row["planning_cost_absolute_error"] for row in rows),default=0.0),"max_physical_time_error":max((row["physical_time_absolute_error"] for row in rows),default=0.0),"max_endpoint_error":max((row["endpoint_absolute_error"] for row in rows),default=0.0),
        "mismatch_count":len(mismatches),"passed":bool(rows) and not mismatches,
    }
    (output/"equivalence_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (output/"equivalence_summary.md").write_text(f"# BSER planner equivalence EQ1\n\nStatus: **{'PASS' if summary['passed'] else 'FAIL'}**\n\nCompared {len(rows)} queries over {len(seen)} unique planning states. Mismatches: {len(mismatches)}. Maximum planning-cost error: {summary['max_planning_cost_error']:.9g}; maximum physical-time error: {summary['max_physical_time_error']:.9g}; maximum endpoint error: {summary['max_endpoint_error']:.9g}.\n",encoding="utf-8")
    manifest={"schema":"bser.eq1.experiment.v1","profiles":list(profiles),"snapshot_steps":[0,10,25,50],"environment_max_steps":400,"query_sources":"four current agents","query_targets":["four navigation targets","all search candidates","all standby candidates","16 fixed-hash valid grid targets"],"state_records":state_records,"formal_training":False,"smoke":bool(smoke)}
    (output/"experiment_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return summary


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT); parser.add_argument("--smoke",action="store_true"); parser.add_argument("--profile",action="append",choices=PROFILES); parser.add_argument("--scenario-id"); args=parser.parse_args(); summary=run(args.output_dir,smoke=args.smoke,profiles_override=args.profile,scenario_id_override=args.scenario_id); print(json.dumps(summary,sort_keys=True)); raise SystemExit(0 if summary["passed"] else 1)


if __name__=="__main__": main()
