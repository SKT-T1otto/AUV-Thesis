"""Bounded collection and separate offline scoring; no routes are installed.

Examples are in docs/chapter3_map_stp/shadow_diagnostic.md. Both commands
require new output directories and a frozen explicit diagnostic configuration.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from .io import ROOT, digest, load_snapshot, new_directory, plain, read, save_snapshot, sha, source_identity, write
from .routes import forecast, route_pool
from .scoring import METHODS, choose, compare_case, hypotheses, route_scores, swept_events


def config(path):
    c = read(path)
    if c.get('schema') != 'ch3.map_stp.shadow.v1' or any(c.get(k) is not False for k in
            ('training','checkpoint_loading','install_shadow_routes')):
        raise ValueError('only nontraining, checkpoint-free shadow configuration is supported')
    if (not 1 <= len(c['scenario_indices']) <= 8 or len(set(c['scenario_indices'])) != len(c['scenario_indices'])
            or not 0 <= max(c['snapshot_steps']) <= c['max_collection_steps'] <= 30
            or not 2 <= c['routes_per_agent'] <= 4 or not 1 <= c['via_limit'] <= 12
            or not 1 <= c['horizon_steps'] <= 80 or not 32 <= c['particles'] <= 1024
            or not 1 <= c['max_cost_ratio'] <= 1.5
            or not 0 <= c['discount_per_second'] <= .1
            or c['selection_seed'] == c['validation_seed']
            or c['target_speed_prior'] != [.25,.55]
            or c['target_direction_prior'] != 'isotropic_sphere'
            or c['condition_current_non_detection'] is not True):
        raise ValueError('diagnostic bounds/prior contract exceeded')
    return c


def collect(args, c):
    # Only this command can import the unchanged baseline runtime and scene data.
    import torch
    from chapter3_bser.experiments.bser_effect_v1.runtime import make_runtime
    from chapter3_bser.experiments.bser_final_v1.forecast import vehicle_constants
    from core.mapping.planning_state import extract_planning_state, planning_state_sha256
    torch.set_num_threads(1)
    manifest = read(ROOT/c['scenario_manifest']); baseline = read(ROOT/c['baseline_config'])
    out = new_directory(args.output_dir,'runs/map_stp_shadow_v1')
    identity = source_identity()
    write(out/'config.json',c); write(out/'source_identity.json',identity)
    result = dict(schema='ch3.map_stp.collection.v1',config_sha256=digest(c),
        inputs={c['scenario_manifest']:sha(ROOT/c['scenario_manifest']),c['baseline_config']:sha(ROOT/c['baseline_config'])},
        source_sha256=identity['source_sha256'], snapshots=[], episodes=[], environment_steps=0,
        training=False, checkpoint_loaded=False, actor_forward_calls=0, optimizer_update_count=0,
        shadow_route_installations=0, formal_evaluation=False)
    for index in c['scenario_indices']:
        runtime = None; started = time.perf_counter()
        try:
            scene = manifest['scenarios'][index]
            runtime = make_runtime(baseline,scene,arm='D2',seed=c['environment_seed_base']+index,episode_id=index)
            captured=[]
            while True:
                task = runtime.env.get_task_state()
                if runtime.step in c['snapshot_steps'] and not task.target_found and not runtime.terminal:
                    before = runtime.controller.current_allocation.allocation_sha256
                    rng_before = torch.get_rng_state().clone()
                    state = extract_planning_state(runtime.env)  # Never refresh the baseline provider.
                    assert before == runtime.controller.current_allocation.allocation_sha256
                    assert torch.equal(rng_before,torch.get_rng_state())
                    env = runtime.env.unwrapped
                    public = dict(dt=float(env.dt), bounds=env.space_size.detach().cpu().numpy().tolist(),
                        detect_eps_bias=float(env.detect_eps_bias),target_clearance=float(env.target_obstacle_clearance),
                        max_reflections=int(env.target_max_reflections_per_step),planner_clearance=float(env.planner_obstacle_clearance),
                        remaining_steps=int(env.max_steps-runtime.step))
                    goals = {str(item.agent_id):list(item.waypoint) for item in runtime.controller.current_allocation.search_assignments
                             if item.failure_reason is None}
                    sid = dict(scenario_index=index,scenario_id=scene['scenario_id'],
                               environment_seed=c['environment_seed_base']+index,step=runtime.step,
                               baseline_allocation_sha256=before)
                    payload = dict(schema='ch3.map_stp.public_snapshot.v1',identity=sid,state=state,public=public,goals=goals,
                        vehicles={str(i):asdict(vehicle_constants(env,i)) for i in state.searcher_ids},
                        state_sha256=planning_state_sha256(state))
                    filename=f'scene_{index:04d}_step_{runtime.step:03d}.json.gz'
                    save_snapshot(out/filename,payload)
                    result['snapshots'].append(dict(file=filename,sha256=sha(out/filename),**sid))
                    captured.append(runtime.step)
                if runtime.step >= c['max_collection_steps'] or runtime.terminal or task.target_found:
                    break
                runtime.advance(); result['environment_steps'] += 1
            diagnostic = runtime.controller_diagnostics()
            assert diagnostic['actor_forward_calls'] == diagnostic['optimizer_update_count'] == 0
            assert diagnostic['residual_action_max_abs'] == 0
            episode = dict(scenario_index=index,steps=runtime.step,captured_steps=captured,
                found=bool(task.target_found),terminal=runtime.terminal,
                stop='terminal' if runtime.terminal else 'found' if task.target_found else 'bounded_prefix',
                collection_seconds=time.perf_counter()-started)
            result['episodes'].append(episode)
            print(json.dumps(dict(event='collected',**episode)),flush=True)
        finally:
            if runtime is not None: runtime.close()
    if source_identity()['source_sha256'] != identity['source_sha256']:
        raise RuntimeError('source bytes changed during collection')
    write(out/'collection.json',result)
    print('Collection complete:',out,flush=True)


def score_snapshot(payload,c):
    from chapter3_bser.config import load_bser_phase1a1_config
    from chapter3_bser.detection_model import detection_probability
    from chapter3_bser.experiments.safe_search_v1.public_geometry import PublicGeometry, filter_planning_state
    from chapter3_bser.types import SearchCandidate
    from core.mapping.planning_state import planning_state_sha256
    state=payload['state']; original_hash=planning_state_sha256(state)
    public=payload['public']; settings=dict(c,planner_clearance=public['planner_clearance'])
    safe=filter_planning_state(state,public['planner_clearance'])
    h=min(c['horizon_steps'],public['remaining_steps'])
    geometry=PublicGeometry(state,public['planner_clearance'])
    groups=[]; paths=[]; generation=[]; agent_ids=[]; legacy=[]
    for i in state.searcher_ids:
        if str(i) not in payload['goals']:
            generation.append(dict(agent_id=i,failure='unassigned_or_failed',selected=0)); continue
        pool,diag=route_pool(safe,i,payload['goals'][str(i)],settings)
        accepted=[]; predictions=[]; references=[]
        for route in pool:
            prediction=forecast(route,state.agents[i],payload['vehicles'][str(i)],h)
            if not geometry.path_free(prediction):
                diag['forecast_geometry_reject']=diag.get('forecast_geometry_reject',0)+1; continue
            accepted.append(route); predictions.append(prediction)
            candidate=SearchCandidate(i,route.route_id,tuple(route.points[-1]),route.points,route.cells,
                                     route.length,route.planning_cost,route.physical_time,route.source)
            references.append(float(np.dot(state.target_belief.probabilities,
                detection_probability(candidate,state,load_bser_phase1a1_config()))))
        generation.append(dict(agent_id=i,**diag,forecast_accepted=len(accepted)))
        if accepted:
            groups.append(accepted); paths.append(predictions); agent_ids.append(i); legacy.append(references)
    row=dict(identity=payload['identity'],public_state_sha256=original_hash,
             filtered_graph_sha256=safe.planning_graph.graph_sha256,generation=generation,agents=[],team=None)
    if not groups: return row
    offset=payload['identity']['scenario_index']*1000+state.step
    selection,selection_identity=hypotheses(state,c,public,c['selection_seed']+offset)
    validation,validation_identity=hypotheses(state,c,public,c['validation_seed']+offset)
    row['selection_model']=selection_identity; row['validation_model']=validation_identity
    def events(target):
        stationary=np.broadcast_to(target[:,:1],target.shape)
        moving=[]; static=[]
        for i,predictions in zip(agent_ids,paths):
            r=state.agents[i].sensor_radius+public['detect_eps_bias']
            moving.append([swept_events(p,target,r) for p in predictions])
            static.append([swept_events(p,stationary,r) for p in predictions])
        return moving,static
    moving,static=events(selection); check_moving,check_static=events(validation)
    for pos,i in enumerate(agent_ids):
        scores=[route_scores(m,s,route.physical_time,public['dt'],c['discount_per_second'])
                for route,m,s in zip(groups[pos],moving[pos],static[pos])]
        check=[route_scores(m,s,route.physical_time,public['dt'],c['discount_per_second'])
               for route,m,s in zip(groups[pos],check_moving[pos],check_static[pos])]
        choices={method:choose(scores,method) for method in METHODS}
        row['agents'].append(dict(agent_id=i,goal=payload['goals'][str(i)],choices=choices,
            routes=[dict(route_id=r.route_id,source=r.source,points=r.points.tolist(),planning_cost=r.planning_cost,
                nominal_time=r.physical_time,length=r.length,forecast_sha256=digest(p),
                legacy_gaussian_reference=old,selection=s,validation=v)
                for r,p,old,s,v in zip(groups[pos],paths[pos],legacy[pos],scores,check)]))
    if tuple(agent_ids)==tuple(state.searcher_ids):
        row['team']=compare_case(groups,moving,static,check_moving,check_static,
            public['dt'],c,c['bootstrap_seed']+offset)
    assert planning_state_sha256(state)==original_hash
    return row


def summarize(rows,c):
    agents=[(r,a) for r in rows for a in r['agents'] if len(a['routes'])>=2]
    teams=[r for r in rows if r['team'] is not None and r['team']['combinations']>1]
    diverse_scenes=sorted({r['identity']['scenario_index'] for r in teams})
    summary=dict(snapshot_count=len(rows),agent_cases=sum(len(r['agents']) for r in rows),
        diverse_agent_cases=len(agents),team_snapshots=sum(r['team'] is not None for r in rows),
        diverse_team_snapshots=len(teams),diverse_scenarios=diverse_scenes,comparisons={})
    for method in METHODS[:-1]:
        deltas=np.array([r['team']['comparisons'][method]['validation_delta'] for r in teams])
        changed=[r['team']['comparisons'][method]['changed'] for r in teams]
        scene_means=[float(np.mean([r['team']['comparisons'][method]['validation_delta'] for r in teams
                                   if r['identity']['scenario_index']==scene])) for scene in diverse_scenes]
        summary['comparisons'][method]=dict(changed_teams=sum(changed),changed_fraction=float(np.mean(changed)) if changed else None,
            mean_validation_delta=float(deltas.mean()) if len(deltas) else None,
            scene_weighted_validation_delta=float(np.mean(scene_means)) if scene_means else None,
            positive_cases=int(np.sum(deltas>1e-12)),negative_cases=int(np.sum(deltas < -1e-12)),
            positive_fraction=float(np.mean(deltas>1e-12)) if len(deltas) else None,
            scene_mean_deltas=dict(zip(map(str,diverse_scenes),scene_means)),
            particle_supported_positive_cases=sum(r['team']['comparisons'][method]['particle_bootstrap95'][0]>0 for r in teams))
    gate=c['decision_gate']; comparison=summary['comparisons']['static']
    enough=(len(agents)>=gate['minimum_diverse_agent_cases'] and len(teams)>=gate['minimum_diverse_team_snapshots']
            and len(diverse_scenes)>=gate['minimum_scenarios'])
    promising=(enough and comparison['changed_fraction']>=gate['minimum_changed_team_fraction']
        and comparison['scene_weighted_validation_delta']>=gate['minimum_validation_absolute_gain']
        and comparison['positive_fraction']>=gate['minimum_positive_case_fraction'])
    summary['decision']=('continue_bounded_execution_validation' if promising else
                         'insufficient_route_opportunity' if not enough else 'do_not_integrate_yet')
    summary['decision_scope']='model-only mechanism screen; no real discovery performance claim'
    return summary


def score(args,c):
    source=Path(args.snapshots).resolve()
    if (ROOT/'runs/map_stp_shadow_v1').resolve() not in source.parents:
        raise ValueError('read only a public snapshot collection under runs/map_stp_shadow_v1')
    collection=read(source/'collection.json')
    if collection['config_sha256']!=digest(c): raise ValueError('collection/config mismatch')
    out=new_directory(args.output_dir,'docs/chapter3_map_stp')
    identity=source_identity()
    if identity['source_sha256']!=collection['source_sha256']:
        raise ValueError('source changed since public collection; recollect explicitly')
    write(out/'config.json',c); write(out/'source_identity.json',identity)
    write(out/'collection.json',collection)
    rows=[]
    for entry in collection['snapshots']:
        path=source/entry['file']
        if sha(path)!=entry['sha256']: raise ValueError('snapshot bytes changed')
        started=time.perf_counter()
        row=score_snapshot(load_snapshot(path),c)
        row['scoring_seconds']=time.perf_counter()-started
        rows.append(row)
        print(json.dumps(dict(event='scored',identity=row['identity'],seconds=row['scoring_seconds'],
                              pools=[len(a['routes']) for a in row['agents']])),flush=True)
    summary=summarize(rows,c)
    write(out/'cases.json',dict(schema='ch3.map_stp.shadow_cases.v1',cases=rows))
    write(out/'summary.json',summary)
    if source_identity()['source_sha256']!=identity['source_sha256']:
        raise RuntimeError('source changed during scoring')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('collect','score'))
    parser.add_argument('--config',default='configs/chapter3/map_stp_shadow_v1.json')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--snapshots')
    args=parser.parse_args(); c=config(args.config)
    if args.command=='score' and not args.snapshots: parser.error('--snapshots is required for offline scoring')
    (collect if args.command=='collect' else score)(args,c)


if __name__=='__main__': main()
