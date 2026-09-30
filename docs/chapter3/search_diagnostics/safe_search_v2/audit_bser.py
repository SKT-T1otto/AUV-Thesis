"""BSER decision audit. Existing evidence stays read-only.

Default: offline paired results. --initial-snapshots additionally constructs 20
initial public states and scores shadow decisions; it never advances physics.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import numpy as np

ROOT = Path.cwd()
HISTORY = ROOT / '3090结果/collision_terminal'
CURRENT = ROOT / 'runs/found_search_v2/development_20260929_v1'
NAMES = {'B0':'B0_search_prior_eval100_seed12729_v1','B1':'B1_bser_prior_eval100_seed12729_v1'}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def mean(values):
    return float(statistics.mean(values)) if values else None

def metric(r, name):
    if name == 'found': return int(r['found'])
    if name == 'success': return int(r['success'])
    if name == 'pre_collision':
        return int(r['first_collision_step'] is not None and (r['found_step'] is None or r['first_collision_step']<=r['found_step']))
    if name == 'score': return r['found_step'] if r['found_step'] is not None else 400
    raise ValueError(name)

def offline():
    data = {b: read(HISTORY/n/'episodes.json') for b,n in NAMES.items()}
    assert read(HISTORY/NAMES['B0']/'evaluation_manifest.json') == read(HISTORY/NAMES['B1']/'evaluation_manifest.json')
    assert read(HISTORY/NAMES['B0']/'identity.json')['comparable_inputs_sha256'] == read(HISTORY/NAMES['B1']/'identity.json')['comparable_inputs_sha256']
    paired = list(zip(data['B0'],data['B1']))
    assert len(paired)==100 and all(len(x)==100 for x in data.values())
    assert all(all(x[k]==y[k] for k in ('scenario_id','scenario_seed','environment_innovation_seed','episode_index')) for x,y in paired)
    assert all(r['task_protocol']=='collision_terminal_v1' and r['termination_reason'] in ('success','timeout','obstacle_collision') for es in data.values() for r in es)
    rng=np.random.default_rng(20260929); samples=rng.integers(0,100,(20000,100))
    effects={}
    for name in ('found','success','pre_collision','score'):
        d=np.array([metric(y,name)-metric(x,name) for x,y in paired],dtype=float)
        effects[name]=dict(B0=mean([metric(x,name) for x in data['B0']]),B1=mean([metric(x,name) for x in data['B1']]),
            delta=float(d.mean()),bootstrap95=np.percentile(d[samples].mean(axis=1),[2.5,97.5]).tolist())
    common=[(x,y) for x,y in paired if x['found'] and y['found']]
    conditional={}
    for name in ('found_step','executor_distance_at_handoff','executor_distance_to_target_at_found','found_to_target_received_steps','success'):
        values=[(x[name],y[name]) for x,y in common if x[name] is not None and y[name] is not None]
        conditional[name]=dict(n=len(values),B0=mean([x for x,y in values]),B1=mean([y for x,y in values]),delta=mean([y-x for x,y in values]))
    standby=[]
    for r in read(CURRENT/'episodes.json'):
        if r['arm']!='R0': continue
        path=CURRENT/r['artifact_dir']/'step_trace.jsonl'
        travelled=0.; goal_changes=0; previous=None; steps=near=0; last=None
        with path.open(encoding='utf-8') as f:
            for line in f:
                t=json.loads(line)
                if not t['search_transition']:continue
                a=t['before']['agents'][3]; z=t['after']['agents'][3]
                travelled+=float(np.linalg.norm(np.asarray(z['position'])-a['position']))
                target=a['semantic_waypoint']; gap=float(np.linalg.norm(np.asarray(a['position'])-target))
                steps+=1;near+=int(gap<.75)
                if previous is not None and previous!=target:goal_changes+=1
                previous=target;last=dict(step=t['step_before'],gap_to_standby=gap,speed=a['speed'])
        standby.append(dict(baseline=r['baseline'],index=r['original_episode_index'],found=r['found_within_budget'],
            pre_found_steps=steps,travelled=travelled,goal_changes=goal_changes,near_goal_steps=near,
            last_pre_found_state=last,pre_found_executor_collision=bool(r['pre_found_collision'] and 3 in r['episode_result']['first_collision_agent_ids'])))
    return dict(historical100=effects,found_gained=[y['episode_index'] for x,y in paired if y['found'] and not x['found']],
        found_lost=[x['episode_index'] for x,y in paired if x['found'] and not y['found']],common_found=conditional,
        common_found_caveat='Outcome-selected 32-scene subset; descriptive mechanism evidence only, not a causal treatment effect.',
        original100_pre_found_executor_collisions={b:[r['episode_index'] for r in es if metric(r,'pre_collision') and 3 in r['first_collision_agent_ids']] for b,es in data.items()},
        current_R0_standby=standby,source_hashes={f'{b}/{n}':sha(HISTORY/d/n) for b,d in NAMES.items() for n in ('episodes.json','identity.json','evaluation_manifest.json')})

def initial_snapshots():
    import torch
    from chapter3_bser.experiments.safe_search_v1.runtime import make_runtime
    from chapter3_bser.experiments.safe_search_v2.provenance import framework_sources, verify_sources
    from chapter3_bser.objective import build_objective_context,evaluate_objective,expected_detection_probability,response_diagnostics
    from chapter3_bser.greedy_solver import solve_fixed_standby_greedy,solve_joint_greedy
    torch.set_num_threads(1)
    before=framework_sources(); plan=read(CURRENT/'identity.json')
    assert plan['sources']==before
    manifest=read(HISTORY/NAMES['B0']/'evaluation_manifest.json'); config=read(ROOT/'configs/chapter3/hgr_train.json')
    outputs=[]
    for scene in plan['selected']:
        i=scene['original_episode_index']; runtime=None
        try:
            runtime=make_runtime(config,manifest['scenarios'][i],baseline='B1_bser_prior',variant='V4',seed=scene['environment_innovation_seed'],episode_id=i)
            state=runtime.state; allocator=runtime.controller.inner.allocator
            generated=allocator._generate_candidates(state)
            candidates,ys=generated.search_candidates,generated.standby_candidates
            context=build_objective_context(state,candidates,ys,allocator.config)
            fixed=next(y for y in ys if np.allclose(y.waypoint,state.agents[state.executor_id].position,rtol=0,atol=1e-10))
            pure=solve_fixed_standby_greedy(candidates,fixed,context,search_only=True)
            fixed_joint=solve_fixed_standby_greedy(candidates,fixed,context)
            joint=solve_joint_greedy(candidates,ys,context)
            y_for_pure=max(ys,key=lambda y:evaluate_objective(pure.selected,y,context))
            groups=[tuple(c for c in candidates if c.agent_id==agent) for agent in state.searcher_ids]
            exact=max(evaluate_objective(tuple(c for c in combo if c is not None),y,context)
                for combo in itertools.product(*[(None,*g) for g in groups]) for y in ys)
            def pack(selected,y):
                diagnostic=response_diagnostics(selected,y,context)
                return dict(selected=[c.candidate_id for c in selected],standby=y.candidate_id,standby_source=y.source,
                    standby_waypoint=list(y.waypoint),standby_move_time=y.physical_travel_time,
                    detection=expected_detection_probability(selected,context),objective=evaluate_objective(selected,y,context),
                    response_time=diagnostic.conditional_reachable_response_time if diagnostic.response_defined else None,
                    unreachable_mass_ratio=diagnostic.unreachable_detected_mass_ratio,
                    search_travel_times=[c.physical_travel_time for c in selected])
            installed=runtime.controller.inner.current_allocation
            assert sorted(c.candidate_id for c in joint.selected)==sorted(c.candidate_id for c in installed.search_assignments)
            assert tuple(joint.standby.waypoint)==tuple(installed.executor_assignment.target_region)
            assert runtime.step==0 and runtime.controller_diagnostics()['residual_steps_checked']==0
            outputs.append(dict(index=i,scenario_id=scene['scenario_id'],public_state_step=0,search_candidates=len(candidates),standby_candidates=len(ys),
                search_only_fixed=pack(pure.selected,fixed),weighted_fixed=pack(fixed_joint.selected,fixed),
                search_only_optimized_standby=pack(pure.selected,y_for_pure),joint=pack(joint.selected,joint.standby),
                exact_joint_objective=exact,greedy_relative_gap=(exact-joint.objective)/max(exact,1e-15)))
            print(f'Initial public snapshot {len(outputs)}/20, no physics steps',flush=True)
        finally:
            if runtime is not None: runtime.close()
    verify_sources(before)
    return dict(source_inventory_sha256=before['inventory']['sha256'],historical_provenance_records=before['historical_record_count'],
        states=outputs,environment_steps=0,training=False,checkpoint_loaded=False,
        caveat='Initialization-only, model-based shadow decisions. These are not rollout outcomes and do not establish performance or later-state behavior.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-snapshots',action='store_true')
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args(); out=Path(args.output_dir).resolve()
    allowed=(ROOT/'docs/chapter3/search_diagnostics/safe_search_v2').resolve()
    if allowed not in out.parents or out.exists():raise ValueError('Use a new evidence subdirectory under safe_search_v2 docs')
    result=dict(schema='ch3.bser.decision_audit.v1',offline=offline(),audit_script_sha256=sha(__file__))
    if args.initial_snapshots:result['initial_snapshots']=initial_snapshots()
    out.mkdir(parents=True)
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('Audit written:',out,flush=True)

if __name__=='__main__': main()
