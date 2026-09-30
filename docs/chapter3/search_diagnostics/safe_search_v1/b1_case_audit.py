"""Read-only mechanism evidence from completed B1 traces; no source changes."""
from collections import Counter
import argparse
import math
from pathlib import Path

from chapter3_bser.experiments.safe_search_v1.run_b1_development import collect_completed_b1, arm_directory
from chapter3_bser.experiments.safe_search_v1.run_paired import write_json
from docs.chapter3.search_diagnostics.safe_search_v1.development_baseline_audit import Inputs
from docs.chapter3.search_diagnostics.safe_search_v1.development_trace_audit import trace_rows


def runs(steps):
    groups=[]
    for step in sorted(set(steps)):
        if groups and groups[-1][-1]+1==step:
            groups[-1].append(step)
        else:
            groups.append([step])
    return [dict(first=g[0],last=g[-1],steps=len(g)) for g in groups]


def inspect(trace, row):
    search=[t for t in trace if t['search_transition']]
    wait, starved, invalid, query_failures, atomic = {}, [], [], [], []
    for agent in range(3):
        steps=[]
        for t in search:
            a=t['before']['agents'][agent]
            m=next(x for x in t['agent_metrics'] if x['agent_id']==agent)
            if a['path_diagnostics'].get('tracker_completed') and m['motion_stall_proxy']:
                steps.append(t['step_before'])
        wait[str(agent)]=[r for r in runs(steps) if r['steps']>=10]
    for t in search:
        state=t['before']
        q=t['queries']['counts']
        search_queries=sum(x['count'] for x in q if x['caller']=='generate_search_candidates')
        completed=sum(a['path_diagnostics'].get('tracker_completed') is True for a in state['agents'][:3])
        if state['agents'][3]['reachable'] is False:
            invalid.append(t['step_before'])
            if not search_queries and completed==3:
                starved.append(t['step_before'])
        failed=sum(x['count'] for x in q if x['reason']=='no_start_connector' and x['caller']=='generate_search_candidates')
        if failed:
            query_failures.append(dict(step_before=t['step_before'],failed_search_queries=failed,
                cached_revision=state['cached_map_revision'],live_revision=state['live_map_revision'],
                decision_before=state['decision_reason'],decision_after=t['after']['decision_reason']))
        if t['after']['decision_reason'].startswith('ATOMIC_REJECT') and search_queries:
            atomic.append(dict(step_before=t['step_before'],step_after=t['step_after'],
                decision_after=t['after']['decision_reason'],
                search_queries=[x for x in q if x['caller']=='generate_search_candidates']))
    collision=[]
    for hit in row['episode_result']['collision_records']:
        agent=hit['agent_id']
        preceding_hold=[]
        for t in reversed(trace):
            if not t['before']['agents'][agent]['hold']:break
            preceding_hold.append(t)
        preceding_hold.reverse()
        seq=[]
        for t in trace[-6:]:
            a=t['before']['agents'][agent]
            seq.append(dict(step=t['step_before'],agent=agent,speed=a['speed'],hold=a['hold'],
                reachable=a['reachable'],position=a['position'],tracking_waypoint=a['tracking_waypoint'],
                velocity_to_tracking_angle_deg=a['velocity_to_tracking_angle_deg'],
                remaining_path_length=a['remaining_path_length'],
                decision_reason=t['before']['decision_reason'],queries=t['queries']['counts']))
        collision.append(dict(agent=agent,role='executor' if agent==3 else 'searcher',
            terminal_step=row['physical_steps'],found_step=row['found_step'],preceding_states=seq))
        collision[-1].update(consecutive_hold_steps_before_collision=len(preceding_hold),
            hold_start_step=preceding_hold[0]['step_before'] if preceding_hold else None,
            distance_during_final_hold=sum(math.dist(t['before']['agents'][agent]['position'],
                t['after']['agents'][agent]['position']) for t in preceding_hold))
    post=[t for t in trace if not t['search_transition']]
    c=row['complete_episode_row']
    return dict(index=row['original_episode_index'],variant=row['variant'],found_step=row['found_step'],
        reason=row['stop_reason'],physical_steps=row['physical_steps'],
        searcher_stall=row['searcher_motion_stall_proxy_agent_steps'],searcher_hold=row['searcher_hold_agent_steps'],
        completed_route_stall_runs=wait,executor_unreachable_runs=runs(invalid),
        all_three_completed_executor_unreachable_no_search_query_runs=runs(starved),
        search_no_start_connector_event_count=len(query_failures),
        search_no_start_connector_events=query_failures if len(query_failures)<=6 else query_failures[:3]+query_failures[-3:],
        collision=collision,
        atomic_rejection_query_step_count=len(atomic),
        atomic_rejection_samples=atomic if len(atomic)<=6 else atomic[:3]+atomic[-3:],
        post_found=dict(steps=len(post),executor_unreachable_steps=sum(t['before']['agents'][3]['reachable'] is False for t in post),
            executor_hold_steps=sum(t['before']['agents'][3]['hold'] for t in post),
            executor_movement=sum(math.dist(t['before']['agents'][3]['position'],t['after']['agents'][3]['position']) for t in post),
            decision_reason_state_labels=dict(Counter(t['after']['decision_reason'] for t in post)),
            recorded_metrics={k:c[k] for k in ('handoff_delay','executor_distance_to_target_at_found',
                'executor_min_distance_to_target','executor_final_distance_to_target','executor_path_unreachable_count',
                'executor_invalid_count','executor_replan_count','capture_contact_step_count','capture_hold_counter_max') if k in c}))


def audit_run(run,output):
    run,output=Path(run),Path(output)
    if output.exists():
        raise ValueError('Output must be new')
    inputs=Inputs();inputs.watch(Path(__file__))
    parent,rows=collect_completed_b1(run)
    inputs.watch(run/'episodes.json');inputs.watch(run/'identity.json')
    result=[]
    for row in rows:
        path=arm_directory(run,row['original_episode_index'],row['variant'])/f"episode_{row['original_episode_index']:04d}"/row['variant']/'step_trace.jsonl'
        inputs.watch(path)
        result.append(inspect(list(trace_rows(path)),row))
    ac=Path(parent['references']['B0_AC']['directory'])
    reference_rows=inputs.read(ac/'episodes.json');reference_collisions=[]
    for row in reference_rows:
        if row['stop_reason']!='obstacle_collision':continue
        i=row['original_episode_index']
        path=ac/f'scene_{i:04d}/episode_{i:04d}/V5/step_trace.jsonl';inputs.watch(path)
        case=inspect(list(trace_rows(path)),row)
        reference_collisions.append(dict(baseline='B0',index=i,variant='V5',
            collisions=[{k:c[k] for k in ('agent','terminal_step','consecutive_hold_steps_before_collision',
                'hold_start_step','distance_during_final_hold')} for c in case['collision']]))
    inputs.verify()
    write_json(output,dict(schema='ch3.safe_search.b1_case_audit.v1',complete=True,episode_count=60,cases=result,
        separately_retained_B0_V5_collision_reference=reference_collisions,
        source_hashes=inputs.hashes,note='State labels are not independent controller invocation counts. Concurrent symptoms do not prove event priority caused a failure.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit_run(a.run,a.output)
