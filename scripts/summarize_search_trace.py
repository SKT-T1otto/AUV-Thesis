"""Summarize bounded diagnostic traces without inferring unobserved causes."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from scripts.analyze_search_failures import file_hash,read_json,distribution


def public_cell(point,mapping):
    coords=[]
    for p,o,h,n in zip(point,mapping["grid_origin"],mapping["grid_spacing"],mapping["grid_shape"]):
        u=(p-o)/h
        if u<0 or u>n:return None
        # Nearest-center convention, resolving exact ties toward lower index.
        coords.append(min(n-1,max(0,math.ceil(u)-1)))
    shape=mapping["grid_shape"]
    return (coords[0]*shape[1]+coords[1])*shape[2]+coords[2]


def segment_hits_box(start,end,obstacle):
    """Post-hoc truth audit only; never used by the running controller."""
    lower=[c-s/2 for c,s in zip(obstacle["center"],obstacle["size"])]
    upper=[c+s/2 for c,s in zip(obstacle["center"],obstacle["size"])]
    enter,leave=0.0,1.0
    for a,b,lo,hi in zip(start,end,lower,upper):
        delta=b-a
        if abs(delta)<1e-15:
            if not lo<=a<=hi:return False
        else:
            t0,t1=sorted(((lo-a)/delta,(hi-a)/delta))
            enter=max(enter,t0);leave=min(leave,t1)
            if enter>leave:return False
    return True


def occupied_voxel_intersections(start,end,mapping,margin=0.0):
    """Offline closed-volume audit of PUBLIC occupied voxels, not truth geometry."""
    shape=mapping["grid_shape"];spacing=mapping["grid_spacing"];origin=mapping["grid_origin"]
    hits=[]
    for index in mapping["occupied_cell_indices"]:
        xyz=(index//(shape[1]*shape[2]),(index//shape[2])%shape[1],index%shape[2])
        box=dict(center=[o+(c+0.5)*h for c,h,o in zip(xyz,spacing,origin)],
                 size=[h+2*margin for h in spacing])
        if segment_hits_box(start,end,box):hits.append(index)
    return hits


def summarize(directory,historical_run_dir=None):
    directory=Path(directory)
    identity=read_json(directory/"identity.json")
    summary=read_json(directory/"summary.json")
    rows=[json.loads(line) for line in (directory/"step_trace.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(rows)!=summary["physical_steps"]:raise ValueError("trace length does not match physical steps")
    if any(r["step_after"]!=i+1 or r["step_before"]!=i for i,r in enumerate(rows)):
        raise ValueError("trace has missing/duplicate/out-of-order steps")
    search_counts=Counter();all_counts=Counter();by_step=[];retained=[]
    for row in rows:
        counts=Counter()
        for q in row["queries"]["counts"]:
            all_counts[q["reason"]]+=q["count"]
            if q["caller"]=="generate_search_candidates":
                search_counts[q["reason"]]+=q["count"];counts[q["reason"]]+=q["count"]
        if counts:by_step.append(dict(step=row["step_after"],query_counts=dict(counts)))
        if "ATOMIC_REJECT_MISSING_SEARCH_ROUTE" in row["after"]["decision_reason"]:
            retained.append(dict(step=row["step_after"],
                same_allocation=row["before"]["allocation_hash"]==row["after"]["allocation_hash"]))
    collisions=[]
    for row in rows:
        for hit in row["collision_records"]:
            agent=hit["agent_id"];first_occupied=None;cell=None;cell_states=[]
            for earlier in rows:
                mapping=earlier["before"].get("live_public_map")
                if mapping is not None:
                    cell=public_cell(hit["first_hit_point"],mapping)
                    state=("out_of_bounds" if cell is None else "occupied" if cell in mapping["occupied_cell_indices"]
                           else "unknown" if cell in mapping["unknown_cell_indices"] else "known_free")
                    cell_states.append(dict(step=earlier["step_before"],state=state))
                    if cell in mapping["occupied_cell_indices"]:
                        if first_occupied is None:first_occupied=earlier["step_before"]
            tail=[dict(step=r["step_before"],speed=r["before"]["agents"][agent]["speed"],
                       velocity_to_tracking_angle_deg=r["before"]["agents"][agent]["velocity_to_tracking_angle_deg"],
                       cross_track_error=r["before"]["agents"][agent]["active_segment_cross_track_error"],
                       next_turn_angle_deg=r["before"]["agents"][agent]["next_turn_angle_deg"],
                       cached_map_revision=r["before"]["cached_map_revision"],
                       live_map_revision=r["before"]["live_map_revision"],
                       decision_reason=r["before"]["decision_reason"],
                       tracking_waypoint=r["before"]["agents"][agent]["tracking_waypoint"])
                  for r in rows[max(0,row["step_after"]-6):row["step_after"]]]
            collisions.append(dict(step=row["step_after"],agent_id=agent,public_collision_cell=cell,
                first_recorded_occupied_cell_step=first_occupied,
                collision_cell_public_state_history=cell_states,
                pre_collision_samples=tail,
                limitation="Occupied voxel timing is not exact obstacle-surface sensing time; trace does not establish a causal subtype."))
    result=dict(schema="ch3.search_trace_summary.v1",scenario_id=identity["scenario_id"],
                physical_steps=summary["physical_steps"],terminal=summary["terminal"],
                source_sha256=identity["source_before"]["sha256"],
                recorded_source_sha256=identity["recorded_source_sha256"],
                source_bytes_match=not identity["different_source_files"],
                observer_noninterference=summary["observer_noninterference"],
                search_candidate_query_counts=dict(search_counts),all_query_counts=dict(all_counts),
                search_candidate_queries_by_step=by_step,atomic_reject_retained_allocation=retained,
                fresh_snapshot_blocked_by_cooldown_steps=[r["step_after"] for r in rows
                    if r["after"].get("full_refresh") and r["after"]["decision_reason"]=="REJECT_EVENT_COOLDOWN"],
                searcher_motion_stall_proxy_agent_steps=sum(bool(m["motion_stall_proxy"])
                    for r in rows if r.get("search_transition",False) for m in r.get("agent_metrics",[]) if m["agent_id"]<3),
                any_searcher_motion_stall_proxy_steps=sum(any(m["agent_id"]<3 and m["motion_stall_proxy"]
                    for m in r.get("agent_metrics",[])) for r in rows if r.get("search_transition",False)),
                controller_map_lag_steps=sum(r["before"]["cached_map_revision"]!=r["before"]["live_map_revision"] for r in rows),
                collisions=collisions,telemetry=summary["telemetry"],
                sources={name:file_hash(directory/name) for name in ("identity.json","summary.json","step_trace.jsonl")},
                limitations=["One bounded prefix does not quantify the cause distribution of 100 historical episodes.",
                             "Query counts exclude initialization, which is stored separately.",
                             "Candidate reachability failures are not necessarily physical disconnection."])
    if historical_run_dir is not None:
        original=Path(historical_run_dir)
        episodes=read_json(original/"episodes.json")
        manifest=read_json(original/"evaluation_manifest.json")
        matches=[r for r in episodes if r["scenario_id"]==identity["scenario_id"] and r["scenario_seed"]==identity["scenario_seed"]]
        scenes=[s for s in manifest["scenarios"] if s["scenario_id"]==identity["scenario_id"] and s["scenario_seed"]==identity["scenario_seed"]]
        if len(matches)!=1 or len(scenes)!=1:raise ValueError("historical comparison needs one exact scene/seed match")
        old=matches[0];scene=scenes[0];new_hits=[h for r in rows for h in r["collision_records"]]
        errors=[]
        same_keys=[(h["agent_id"],h["obstacle_id"]) for h in new_hits]==[(h["agent_id"],h["obstacle_id"]) for h in old["collision_records"]]
        if new_hits and same_keys:
            for left,right in zip(new_hits,old["collision_records"]):
                errors.extend(abs(a-b) for field in ("position_before","candidate_position","first_hit_point") for a,b in zip(left[field],right[field]))
        result["historical_terminal_comparison"]=dict(
            terminal_step_matches=summary["terminal"] and summary["physical_steps"]==old["terminal_step"],
            collision_agent_obstacle_pairs_match=same_keys,
            max_collision_coordinate_difference=max(errors) if errors else None,
            limitation="Only exported terminal facts are compared; historical per-step trajectories were not saved.")
        audits=[]
        for row in rows:
            for hit in row["collision_records"]:
                agent=row["before"]["agents"][hit["agent_id"]]
                path=agent["planned_path"];obstacle=scene["obstacles"][hit["obstacle_id"]]
                intersecting=[i for i in range(len(path)-1) if segment_hits_box(path[i],path[i+1],obstacle)]
                audits.append(dict(agent_id=hit["agent_id"],obstacle_id=hit["obstacle_id"],
                                   planned_path_segment_indices_intersecting_truth=intersecting,
                                   cross_track_error_before_collision=agent["active_segment_cross_track_error"],
                                   speed_before_collision=agent["speed"],
                                   information_boundary="Post-hoc manifest truth; never passed to runtime guidance/control."))
                initial=rows[0]["before"]
                public_map=initial.get("live_public_map")
                initial_path=initial["agents"][hit["agent_id"]]["planned_path"]
                if public_map is not None and initial_path==path:
                    audits[-1]["initial_public_map_closed_voxel_audit"]=[
                        dict(segment_index=i,margin=margin,intersecting_occupied_voxels=
                             occupied_voxel_intersections(path[i],path[i+1],public_map,margin))
                        for i in intersecting for margin in (0.0,float(scene["planner_obstacle_clearance"]))]
        result["posthoc_planned_path_truth_audit"]=audits
        result["sources"].update({"historical_"+name:file_hash(original/name) for name in ("episodes.json","evaluation_manifest.json")})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--probe-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--historical-run-dir",type=Path,help="Optional offline terminal/scene-truth comparison; never read by controller")
    args=p.parse_args()
    if args.output.exists():raise ValueError("output already exists")
    result=summarize(args.probe_dir,args.historical_run_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("scenario_id","physical_steps","search_candidate_query_counts","controller_map_lag_steps")},ensure_ascii=True))


if __name__=="__main__":main()
