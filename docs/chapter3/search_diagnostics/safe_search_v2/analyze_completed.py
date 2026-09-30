"""Read-only audit of completed R0-R4 development outputs; no simulator imports.

Run from repository root with: python -B -m
docs.chapter3.search_diagnostics.safe_search_v2.analyze_completed
"""
from pathlib import Path
import collections
import hashlib
import json
import numpy as np

from chapter3_bser.experiments.safe_search_v2.run_windows import (
    BASELINES, read_completed, summarize, file_hash, verify_sources)

RUN = Path('runs/found_search_v2/development_20260929_v1')
OUT = Path('docs/chapter3/search_diagnostics/safe_search_v2/results_20260929')

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def main():
    plan, rows, summary = (read(RUN / n) for n in ('identity.json','episodes.json','summary.json'))
    verify_sources(plan['sources'])
    assert all(file_hash(p) == h for p, h in plan['input_sha256'].items())
    expected = {(BASELINES[b], a, s['original_episode_index']) for b in plan['baselines']
                for a in plan['arms'] for s in plan['selected']}
    assert len(rows) == len(expected) == 200
    assert {(r['baseline'], r['arm'], r['original_episode_index']) for r in rows} == expected
    computed = summarize(rows, plan)
    assert computed['groups'] == summary['groups']
    assert computed['paired_vs_R0'] == summary['paired_vs_R0']
    historical_matches = {}
    for b, name, variant in [('B0', 'development_AC_20260928_v1', 'V5'),
                             ('B1', 'development_B1_remaining_20260929_v1', 'V4')]:
        historical = read(Path('runs/safe_search_v1') / name / 'episodes.json')
        historical = {r['original_episode_index']: r for r in historical if r['variant'] == variant}
        reference = [r for r in rows if r['baseline'] == BASELINES[b] and r['arm'] == 'R0']
        matches = sum(r['signature_sha256'] == historical[r['original_episode_index']]['signature_sha256'] for r in reference)
        assert matches == 20
        historical_matches[b] = dict(parent=variant, matching_full_episode_signatures=matches)
    diagnostics, traces = {}, {}
    for n, row in enumerate(rows, 1):
        b = next(b for b,v in BASELINES.items() if v == row['baseline'])
        a, idx = row['arm'], row['original_episode_index']
        scene = next(s for s in plan['selected'] if s['original_episode_index'] == idx)
        path = RUN / row['artifact_dir']
        checked = read_completed(path, plan, b, a, scene)
        assert dict(checked, artifact_dir=row['artifact_dir']) == row
        key = f'{b}_{a}_{idx:04d}'
        d = dict(status_counts=collections.Counter(), longest_tracking_stall={},
                 v2_counts=row['controller'].get('found_search_v2',{}).get('counts',{}),
                 collision_agents=row['episode_result']['first_collision_agent_ids'],
                 found_step=row['found_step'], stop_reason=row['stop_reason'],
                 pre_found_collision=row['pre_found_collision'],
                 physical_steps=row['physical_steps'], query_counts=row['telemetry']['counts'],
                 terminal_searchers=[], pre_found_motion_sha256=None,
                 stalled_tracking_speed_sum=0., stalled_tracking_count=0,
                 anchor_zero_error_hold_count=0, hold_diagnostic_count=0,
                 unique_observed_cells=row['search_coverage']['unique_observed_cells'])
        h = hashlib.sha256()
        streak, best, previous_target = collections.Counter(), {}, {}
        search_count = effective = holds = stalls = 0
        with (path/'step_trace.jsonl').open(encoding='utf-8') as f:
            for line in f:
                t = json.loads(line)
                if not t['search_transition']:
                    continue
                search_count += 1
                effective += t['search_coverage']['effective_observation_steps']
                before = t['before']
                h.update(json.dumps([[ag['position'], ag['velocity']] for ag in t['after']['agents']],
                                    sort_keys=True).encode())
                guidance = (before.get('found_search_v2') or {}).get('guidance',{})
                for ag in before['agents'][:3]:
                    i = str(ag['agent_id'])
                    gd = guidance.get(i,{})
                    status = gd.get('status','PARENT')
                    d['status_counts'][status] += 1
                    holds += int(ag['hold'])
                    stalls += int(next(m for m in t['agent_metrics'] if m['agent_id']==ag['agent_id'])['motion_stall_proxy'])
                    if gd.get('anchor') is not None:
                        d['hold_diagnostic_count'] += 1
                        d['anchor_zero_error_hold_count'] += int(gd['anchor_error'] < 1e-9)
                    target = ag['tracking_waypoint']
                    stationary = status in ('TRACKING', 'PARENT') and ag['speed'] < .02 and not ag['hold']
                    if stationary:
                        d['stalled_tracking_speed_sum'] += ag['speed']
                        d['stalled_tracking_count'] += 1
                    streak[i] = streak[i]+1 if stationary and target == previous_target.get(i) else int(stationary)
                    previous_target[i] = target
                    if streak[i] > best.get(i,{}).get('steps',0):
                        best[i] = dict(steps=streak[i], end_step=before['step'],
                                      tracking_distance=ag['tracking_distance'],speed=ag['speed'],
                                      assignment_id=ag['assignment_id'],path_index=ag['path_index'],
                                      path_point_count=ag['path_point_count'],position=ag['position'],target=target)
                d['terminal_searchers'] = [{k:ag[k] for k in ('agent_id','hold','speed','tracking_distance','path_index','path_point_count','position','tracking_waypoint')} for ag in t['after']['agents'][:3]]
        assert search_count == row['pre_found_exposure_steps']
        assert effective == row['effective_search_steps']
        assert holds == row['searcher_hold_agent_steps']
        assert stalls == row['searcher_motion_stall_proxy_agent_steps']
        d['longest_tracking_stall'] = best
        d['pre_found_motion_sha256'] = h.hexdigest()
        diagnostics[key] = d
        traces[key] = row['trace_sha256']
        if n % 20 == 0: print(f'Audited {n}/200', flush=True)
    group_d = {}
    for b in plan['baselines']:
        for a in plan['arms']:
            ds = [d for k,d in diagnostics.items() if k.startswith(f'{b}_{a}_')]
            counts, statuses, queries = collections.Counter(), collections.Counter(), collections.Counter()
            for d in ds:
                counts.update(d['v2_counts']); statuses.update(d['status_counts']); queries.update(d['query_counts'])
            group_d[f'{b}_{a}'] = dict(v2_counts=counts,status_counts=statuses,query_counts=queries,
                tracking_stationary_agent_steps=sum(d['stalled_tracking_count'] for d in ds),
                episodes_tracking_stationary_50_steps=sum(any(v['steps']>=50 for v in d['longest_tracking_stall'].values()) for d in ds),
                anchor_zero_error_hold_count=sum(d['anchor_zero_error_hold_count'] for d in ds),
                hold_diagnostic_count=sum(d['hold_diagnostic_count'] for d in ds),
                pre_found_executor_collision_episodes=sum(d['pre_found_collision'] and 3 in d['collision_agents'] for d in ds),
                unique_observed_cells_mean=float(np.mean([d['unique_observed_cells'] for d in ds])))
    paired = {}
    rng = np.random.default_rng(20260929)
    boot = rng.integers(0,20,size=(20000,20))
    indices = [s['original_episode_index'] for s in plan['selected']]
    lookup = {(r['baseline'],r['arm'],r['original_episode_index']):r for r in rows}
    for b in plan['baselines']:
        for a, ref in [('R1','R0'),('R2','R0'),('R3','R0'),('R4','R0'),('R2','R1'),('R3','R1'),('R4','R3'),('R4','R2')]:
            treatment = [lookup[(BASELINES[b],a,i)] for i in indices]
            control = [lookup[(BASELINES[b],ref,i)] for i in indices]
            p = dict(gained_found=[], lost_found=[], avoided_collision=[], added_collision=[],metrics={})
            for i,x,y in zip(indices,treatment,control):
                if x['found_within_budget'] and not y['found_within_budget']: p['gained_found'].append(i)
                if y['found_within_budget'] and not x['found_within_budget']: p['lost_found'].append(i)
                if y['pre_found_collision'] and not x['pre_found_collision']: p['avoided_collision'].append(i)
                if x['pre_found_collision'] and not y['pre_found_collision']: p['added_collision'].append(i)
            funcs = dict(found_pp=lambda r:100*int(r['found_within_budget']), collision_pp=lambda r:100*int(r['pre_found_collision']),
                         penalized_steps=lambda r:r['found_step'] if r['found_step'] is not None else 400)
            for metric,func in funcs.items():
                delta = np.array([func(x)-func(y) for x,y in zip(treatment,control)])
                p['metrics'][metric] = dict(delta=float(delta.mean()), percentile_bootstrap_95=np.percentile(delta[boot].mean(axis=1),[2.5,97.5]).tolist())
            common = [x['found_step']-y['found_step'] for x,y in zip(treatment,control) if x['found_step'] is not None and y['found_step'] is not None]
            p['common_found_n'] = len(common)
            p['common_found_step_delta_mean'] = float(np.mean(common)) if common else None
            p['identical_pre_found_motion_scenes'] = [i for i in indices if diagnostics[f'{b}_{a}_{i:04d}']['pre_found_motion_sha256']==diagnostics[f'{b}_{ref}_{i:04d}']['pre_found_motion_sha256']]
            paired[f'{b}_{a}_vs_{ref}'] = p
    result = dict(run=str(RUN),audit=dict(complete_episode_runs=200,independent_scenes=20,
        physical_steps=sum(r['physical_steps'] for r in rows), all_child_identities_and_trace_hashes_verified=True,
        source_and_input_hashes_verified=True, summary_recomputed=True,
        historical_reference_signature_matches=historical_matches,
        coverage_hold_and_stall_recomputed_from_all_search_transitions=True,
        found_collision_same_step=sum(r['found_step'] is not None and r['pre_found_collision'] for r in rows),
        input_files={n:file_hash(RUN/n) for n in ('identity.json','episodes.json','summary.json')},
        trace_sha256=traces),groups=computed['groups'],diagnostics=group_d,paired=paired,episodes=diagnostics,
        uncertainty='Paired percentile bootstrap, 20000 resamples of the 20 fixed development scenes, seed 20260929. Descriptive only; selection on reused development scenes and multiple comparisons are not corrected.')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('groups','diagnostics','paired')},ensure_ascii=False))

if __name__ == '__main__':
    main()
