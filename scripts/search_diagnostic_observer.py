"""Opt-in read-only telemetry for existing CH3 runtimes.

No policy, environment, torch or numpy imports. QueryTap returns the exact
original object, calls the query once, and restores the class on exit. Use only
in a serial diagnostic process, never concurrently with another runtime.
"""
from __future__ import annotations

from collections import Counter, deque
from contextlib import AbstractContextManager
import math
import sys


def vector(value):
    if hasattr(value, "detach"):
        value=value.detach().cpu().tolist()
    result=[float(x) for x in value]
    if len(result)!=3 or not all(math.isfinite(x) for x in result):
        raise ValueError("expected finite 3D vector")
    return result


def norm(value):return math.sqrt(sum(x*x for x in value))
def subtract(a,b):return [x-y for x,y in zip(a,b)]
def distance(a,b):return norm(subtract(a,b))


def segment_distance(point,start,end):
    delta=subtract(end,start);d=sum(x*x for x in delta)
    fraction=max(0,min(1,sum(x*y for x,y in zip(subtract(point,start),delta))/d)) if d else 0
    return distance(point,[x+fraction*y for x,y in zip(start,delta)])


def angle(a,b):
    denominator=norm(a)*norm(b)
    return None if denominator<=1e-12 else math.degrees(math.acos(max(-1,min(1,sum(x*y for x,y in zip(a,b))/denominator))))


def plain(value):
    if hasattr(value,"value"):return value.value
    return str(value)


def endpoint_present(state,point,role):
    kind=str(role).lower().startswith("exec")
    return any(str(e.role).lower().startswith("exec")==kind and
               all(abs(float(a)-float(b))<=1e-12 for a,b in zip(e.point,point))
               for e in state.planning_graph.endpoint_connectors)


class QueryTap(AbstractContextManager):
    """Observe real path calls without re-querying, refreshing or changing results."""
    def __init__(self,service_class,max_samples=24):
        self.service_class=service_class;self.max_samples=max_samples
        self.counts=Counter();self.samples=[];self.original=None

    def __enter__(self):
        if getattr(self.service_class.query,"_search_diagnostic_tap",False):
            raise RuntimeError("query tap already active; serial use only")
        self.original=self.service_class.query
        owner=self
        def observed(service,start,goal,agent):
            caller=sys._getframe(1).f_code.co_name
            result=owner.original(service,start,goal,agent)
            state=service.state
            status="reachable" if result.reachable else str(result.failure_reason or "unspecified")
            key=(int(state.step),int(agent.agent_id),caller,status,int(state.map_revision))
            owner.counts[key]+=1
            if not result.reachable and len(owner.samples)<owner.max_samples:
                sample=dict(step=key[0],agent_id=key[1],caller=caller,reason=status,map_revision=key[4])
                for name,value in (("start",start),("goal",goal)):
                    try:
                        coordinate=vector(value)
                    except (TypeError,ValueError):
                        # The original query may consume a one-shot iterable.
                        # Telemetry must not turn its returned failure into an exception.
                        coordinate=None
                    sample[name]=coordinate
                    sample[name+"_endpoint_present"]=(endpoint_present(state,coordinate,agent.role)
                                                       if coordinate is not None else None)
                owner.samples.append(sample)
            return result
        observed._search_diagnostic_tap=True
        self.service_class.query=observed
        return self

    def drain(self):
        result=dict(counts=[dict(step=k[0],agent_id=k[1],caller=k[2],reason=k[3],map_revision=k[4],count=v)
                            for k,v in sorted(self.counts.items())],samples=self.samples)
        self.counts.clear();self.samples=[]
        return result

    def __exit__(self,*exception):
        self.service_class.query=self.original
        return False


def capture_runtime(runtime,live_public_state=None):
    """Sample existing public state and installed guidance; never advance a tracker.

    Occupancy is the controller's cached view, explicitly dated. Fresh sensing
    revision is public metadata only. No truth obstacles or target coordinates.
    """
    state=runtime.state;agents=runtime.env.get_agent_state();task=runtime.env.get_task_state()
    mapping=runtime.env.get_mapping_state()
    guidance=runtime.guidance
    rows=[]
    for agent in state.agents:
        i=agent.agent_id;assignment=guidance.assignment_for(i)
        tracking=runtime.bridge.path_tracker.snapshot(i)
        position=vector(agents.positions[i]);velocity=vector(agents.velocities[i])
        path=[vector(p) for p in assignment.planned_path]
        index=tracking.next_index
        start=path[max(0,index-1)] if path else position
        end=path[min(index,len(path)-1)] if path else vector(assignment.tracking_waypoint)
        target=vector(assignment.tracking_waypoint)
        next_point=path[index+1] if index+1<len(path) else None
        rows.append(dict(agent_id=i,role=agent.role,position=position,velocity=velocity,speed=norm(velocity),
            assignment_id=assignment.assignment_id,assignment_kind=assignment.assignment_kind,
            semantic_waypoint=vector(assignment.final_waypoint),tracking_waypoint=target,
            hold=bool(assignment.hold_state),reachable=bool(assignment.reachable),
            path_index=index,path_point_count=len(path),
            planned_path=path,
            tracking_distance=distance(position,target),
            active_segment_cross_track_error=segment_distance(position,start,end),
            velocity_to_tracking_angle_deg=angle(velocity,subtract(target,position)),
            next_turn_angle_deg=angle(subtract(end,start),subtract(next_point,end)) if next_point else None,
            remaining_path_length=runtime.bridge.path_tracker.remaining_path_length(i,position)))
    allocation=runtime.controller.current_allocation
    result=dict(step=int(task.step),found=bool(task.target_found),executor_knows_target=bool(task.executor_knows_target),
        agents=rows,allocation_hash=allocation.allocation_sha256,
        decision_reason=guidance.decision_reason,
        cached_map_revision=int(state.map_revision),live_map_revision=int(mapping.map_revision),
        cached_known_cells=int(sum(bool(v) for v in state.occupancy.known_mask)),
        cached_occupied_cells=int(sum(bool(v) for v in state.occupancy.occupied_mask)),
        last_full_refresh_step=int(runtime.provider._last_full_refresh_step),
        full_refresh=bool(runtime.provider.last_snapshot_was_full_refresh),
        replan_count=int(runtime.controller.replan_count),
        replan_steps=list(runtime.controller.replan_steps))
    if live_public_state is not None:
        result["live_public_map"]=dict(
            diagnostic_only=True,map_revision=int(live_public_state.map_revision),
            grid_shape=list(live_public_state.grid.shape),grid_origin=list(live_public_state.grid.origin),
            grid_spacing=list(live_public_state.grid.spacing),
            occupied_cell_indices=[i for i,v in enumerate(live_public_state.occupancy.occupied_mask) if v],
            unknown_cell_indices=[i for i,v in enumerate(live_public_state.occupancy.unknown_mask) if v])
    return result


class TransitionObserver:
    """Append physical transitions, with conservative explicitly named proxies."""
    def __init__(self,window=10,progress_epsilon=0.1,displacement_epsilon=0.1):
        if window<2:raise ValueError("window must be at least 2")
        self.window=window;self.progress_epsilon=progress_epsilon;self.displacement_epsilon=displacement_epsilon
        self.history={i:deque(maxlen=window+1) for i in range(4)}
        self.summary=Counter();self.agent_summary={i:Counter() for i in range(4)}

    def observe(self,before,after,queries,terminal_result=None):
        if after["step"]!=before["step"]+1:raise ValueError("one physical step per observation required")
        search=not before["found"]
        row=dict(step_before=before["step"],step_after=after["step"],search_transition=search,
                 before=before,after=after,queries=queries,agent_metrics=[],
                 collision_records=(terminal_result or {}).get("collision_records",[]))
        if search:self.summary["pre_found_exposure_steps"]+=1
        self.summary["query_count"]+=sum(x["count"] for x in queries["counts"])
        for q in queries["counts"]:
            if q["reason"]!="reachable":self.summary["query_failure:"+q["reason"]]+=q["count"]
        for a,b in zip(before["agents"],after["agents"]):
            if a["agent_id"]!=b["agent_id"]:raise ValueError("agent order changed")
            i=a["agent_id"];history=self.history[i]
            if not history:history.append(a)
            history.append(b)
            stable=len(history)==self.window+1 and len({x["assignment_id"] for x in history})==1
            progress=(history[0]["remaining_path_length"]-history[-1]["remaining_path_length"]) if stable else None
            displacement=distance(history[0]["position"],history[-1]["position"]) if stable else None
            stall=bool(stable and progress<self.progress_epsilon and displacement<self.displacement_epsilon
                       and a["reachable"] and not a["hold"])
            # This is a motion proxy only; target-search coverage is unavailable.
            metrics=dict(agent_id=i,displacement=distance(a["position"],b["position"]),
                         speed_change=b["speed"]-a["speed"],stable_assignment_window=stable,
                         window_remaining_path_progress=progress,window_net_displacement=displacement,
                         motion_stall_proxy=stall,
                         effective_search=None,target_search_coverage_increment=None)
            row["agent_metrics"].append(metrics)
            if search:
                self.agent_summary[i]["search_transitions"]+=1
                self.agent_summary[i]["hold_steps"]+=int(a["hold"])
                self.agent_summary[i]["motion_stall_proxy_steps"]+=int(stall)
        return row

    def result(self):
        return dict(schema="ch3.search_transition_observer.v1",counts=dict(self.summary),
                    agents={str(k):dict(v) for k,v in self.agent_summary.items()},
                    motion_stall_window_steps=self.window,progress_epsilon=self.progress_epsilon,
                    displacement_epsilon=self.displacement_epsilon,
                    effective_search_steps=None,
                    limitations=["Motion stall proxy is not search-coverage stagnation.",
                                 "Cached occupancy changes are not first sensor-observation timestamps.",
                                 "No causal collision subtype is assigned without a bounded intervention."])
