"""Offline paired B1 transfer analysis. Requires all 60 real B1 terminals.

Historical B0 outputs keep their own identities. No simulator is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
import inspect
from pathlib import Path

from chapter3_bser.experiments.safe_search_v1 import analyze_development as stats
from chapter3_bser.experiments.safe_search_v1.run_b1_development import (
    B1_PLAN, VARIANTS, arm_directory, collect_completed_b1, reference_metadata)
from chapter3_bser.experiments.safe_search_v1.run_development import INDICES
from chapter3_bser.experiments.safe_search_v1.run_paired import ROOT, file_hash, write_json
from docs.chapter3.search_diagnostics.safe_search_v1.analyze_ac_development import episode_metrics, gate_result
from docs.chapter3.search_diagnostics.safe_search_v1.development_baseline_audit import Inputs
from docs.chapter3.search_diagnostics.safe_search_v1 import development_trace_audit as windows


def comparison(values, new, old):
    """Matched scene differences; never treat 60 arms as independent scenes."""
    result = {}
    for metric in values[INDICES[0], old]:
        complete = [(i, values[i, old][metric], values[i, new][metric]) for i in INDICES
                    if values[i, old][metric] is not None and values[i, new][metric] is not None]
        record = (stats.paired_bootstrap([b-a for _, a, b in complete]) if complete else
                  dict(mean_delta=None, bootstrap_95_percentile_ci=None, n_pairs=0))
        record.update(direction=f"{new} minus {old}", n_planned_pairs=20,
            excluded_zero_exposure_indices=[i for i in INDICES if i not in {r[0] for r in complete}])
        if metric in ("found", "success", "pre_found_collision"):
            record.update(gained_indices=[i for i,a,b in complete if a==0 and b==1],
                          lost_indices=[i for i,a,b in complete if a==1 and b==0])
        result[metric] = record
    return result


def diagnostics(trace, terminal):
    search = [r for r in trace if r['search_transition']]
    reasons = Counter(r['after']['decision_reason'] for r in search)
    no_candidates = 0
    stale = 0
    executor_invalid = 0
    for r in search:
        a = r['before']
        stale += a['cached_map_revision'] != a['live_map_revision']
        executor_invalid += a['agents'][3]['reachable'] is False
        if not any(q['caller']=='generate_search_candidates' for q in r['queries']['counts']):
            no_candidates += 1
    return dict(original_episode_index=terminal['original_episode_index'],
        variant=terminal['variant'], outcome=stats.outcome(terminal),
        found_step=terminal['found_step'], terminal_step=terminal['physical_steps'],
        stall_agent_steps=terminal['searcher_motion_stall_proxy_agent_steps'],
        hold_agent_steps=terminal['searcher_hold_agent_steps'],
        decision_reason_state_labels=dict(reasons), stale_map_before_steps=stale,
        executor_unreachable_before_steps=executor_invalid,
        steps_without_search_named_query=no_candidates,
        note='Decision labels can persist between decisions; absence of a query is not itself failure.')


def analyze(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new analysis output directory')
    audit = Inputs()
    audit.watch(Path(__file__))
    for helper in (stats.aggregate, windows.audit_trace, episode_metrics, gate_result, Inputs):
        audit.watch(Path(inspect.getfile(helper)))
    plan = audit.read(B1_PLAN)
    parent, b1_rows = collect_completed_b1(run)
    preflight=audit.read(ROOT/'docs/chapter3/search_diagnostics/safe_search_v1/b1_preflight_verification.json')
    if (preflight['passed'] is not True or preflight['v0_native_equal_at_every_step'] is not True
            or preflight['source_inventory_sha256']!=parent['sources_before']['inventory']['sha256']
            or preflight['recorded_arms']!=6 or preflight['original_indices']!=[3,28]):
        raise ValueError('B1 native equivalence preflight missing')
    prefix=ROOT/'runs/safe_search_v1/b1_mechanism_20260928_v1'
    for name in ('identity','episodes'):
        if audit.watch(prefix/(name+'.json'))!=preflight[name+'_sha256']:
            raise ValueError('B1 preflight evidence changed')
    refs = parent['references']
    d2, ac = (Path(refs[k]['directory']) for k in ('B0_D2','B0_AC'))
    if reference_metadata(d2, ac) != refs:
        raise ValueError('B0 historical reference has changed')
    source_review = audit.read(ROOT/'docs/chapter3/search_diagnostics/safe_search_v1/b1_source_diff_review.json')
    if (source_review['before_inventory_sha256'] != refs['B0_AC']['source_inventory_sha256']
            or source_review['after_inventory_sha256'] != parent['sources_before']['inventory']['sha256']
            or source_review['unchanged_existing_count'] != 336
            or source_review['runtime_and_controller_bytes_unchanged'] is not True):
        raise ValueError('B1 source comparability review missing')
    historical_analysis=audit.read(ROOT/'docs/chapter3/search_diagnostics/safe_search_v1/ac_results_20260928/analysis.json')
    old_comparability=historical_analysis['source_comparability']
    if (old_comparability['passed'] is not True or old_comparability['reviewed_exact_source_diff_passed'] is not True
            or historical_analysis['sources']['new_source_inventory_sha256']!=refs['B0_AC']['source_inventory_sha256']
            or historical_analysis['sources']['reference_source_inventory_sha256']!=refs['B0_D2']['source_inventory_sha256']):
        raise ValueError('Historical D2-to-AC comparability chain missing')
    historical_rows=audit.read(d2/'episodes.json')
    if {(c['original_episode_index'],c['variant']) for c in old_comparability['controls']}!={(3,'V3'),(28,'V4')}:
        raise ValueError('Both historical source controls are required')
    for c in old_comparability['controls']:
        i,v=c['original_episode_index'],c['variant']
        directory=ROOT/'runs/safe_search_v1/ac_source_controls_20260928_v1'/f'scene_{i:04d}_{v}'
        identity_path=directory/'identity.json'
        summary_path=directory/f'episode_{i:04d}'/v/'summary.json'
        if audit.watch(identity_path)!=c['identity_sha256'] or audit.watch(summary_path)!=c['summary_sha256']:
            raise ValueError('Historical source control evidence changed')
        control=audit.read(summary_path)
        old=next(r for r in historical_rows if r['original_episode_index']==i and r['variant']==v)
        if (control['signature_sha256']!=old['signature_sha256']
                or control['signature_sha256']!=c['ordered_signature_sha256']
                or control['physical_steps']!=400 or control['episode_result']!=old['episode_result']):
            raise ValueError('Historical source control no longer reconciles')
    for directory in (run,d2,ac):
        for name in ('identity.json','episodes.json','summary.json'):
            audit.watch(directory/name)
    old_rows = [r for r in audit.read(d2/'episodes.json') if r['variant'] in ('V0','V3')]
    old_rows += audit.read(ac/'episodes.json')
    results, values = {}, {}
    all_windows, all_diagnostics, compact_rows = {}, {}, []
    for baseline, rows in (('B0',old_rows), ('B1',b1_rows)):
        if len(rows)!=60 or {(r['original_episode_index'],r['variant']) for r in rows} != {(i,v) for i in INDICES for v in VARIANTS}:
            raise ValueError('Each baseline requires exactly the same 20 x 3 arms')
        traces, window_records, cases = {}, [], []
        for row in rows:
            i,v = row['original_episode_index'],row['variant']
            directory = arm_directory(run,i,v) if baseline=='B1' else (ac if v=='V5' else d2)/f'scene_{i:04d}'
            path = directory/f'episode_{i:04d}'/v/'step_trace.jsonl'
            audit.watch(path)
            trace = list(windows.trace_rows(path))
            window_records.append(windows.audit_trace(trace,row))
            hold = sum(a['hold'] for t in trace if t['search_transition'] for a in t['before']['agents'][:3])
            if hold != row['searcher_hold_agent_steps']:
                raise ValueError('Hold trace/summary mismatch')
            traces[i,v] = stats.trace_summary(path)
            cases.append(diagnostics(trace,row))
            compact_rows.append(dict(baseline=baseline, original_episode_index=i, variant=v,
                scenario_id=row['scenario_id'], scenario_seed=row['scenario_seed'],
                environment_innovation_seed=row['environment_innovation_seed'],
                outcome=stats.outcome(row), found_step=row['found_step'],
                physical_steps=row['physical_steps'], pre_found_exposure_steps=row['pre_found_exposure_steps'],
                metrics=episode_metrics(row,traces[i,v])))
        vals = {(r['original_episode_index'],r['variant']):episode_metrics(r,traces[r['original_episode_index'],r['variant']]) for r in rows}
        values[baseline] = vals
        summary = {v:stats.aggregate([r for r in rows if r['variant']==v],traces) for v in VARIANTS}
        for v,s in summary.items():
            s['stall_plus_hold_fraction'] = (s['searcher_motion_stall_proxy_agent_steps']+s['searcher_hold_agent_steps'])/(3*s['pre_found_exposure_steps'])
            ts = [t for (i,variant),t in traces.items() if variant==v]
            s['executor_pre_found_path_length'] = sum(t['pre_found_path_length_by_agent'].get('3',0) for t in ts)
            s['searcher_pre_found_path_length'] = sum(sum(t['pre_found_path_length_by_agent'].get(str(j),0) for j in range(3)) for t in ts)
            callers=Counter()
            for t in ts:
                callers.update(t['pre_found_query_callers'])
            s['pre_found_query_callers']=dict(callers)
            s['pre_found_collision_snapshots'] = [dict(original_episode_index=i,**c)
                for (i,variant),t in traces.items() if variant==v for c in t['pre_found_collision_snapshots']]
        results[baseline] = dict(variants=summary,comparisons={f'{a}_minus_{b}':comparison(vals,a,b) for a,b in (('V5','V3'),('V5','V0'),('V3','V0'))})
        all_windows[baseline] = {v:dict(all_outcomes=windows.aggregate([r for r in window_records if r['variant']==v]),
            no_found_timeouts=windows.aggregate([r for r in window_records if r['variant']==v and r['outcome']=='no_found_timeout'])) for v in VARIANTS}
        all_diagnostics[baseline] = cases
    by_key = {(r['baseline'],r['original_episode_index'],r['variant']):r for r in compact_rows}
    for i in INDICES:
        for v in VARIANTS:
            a,b=by_key['B0',i,v],by_key['B1',i,v]
            if any(a[k]!=b[k] for k in ('scenario_id','scenario_seed','environment_innovation_seed')):
                raise ValueError('B0/B1 scene or seed identity mismatch')
    transfer = {}
    for new,old in (('V5','V3'),('V5','V0'),('V3','V0')):
        metrics = {}
        for metric in ('found','success','pre_found_collision','stall_plus_hold_fraction','effective_observation_fraction'):
            diffs=[]
            for i in INDICES:
                a,b,c,d = [values[base][i,v][metric] for base,v in (('B1',new),('B1',old),('B0',new),('B0',old))]
                if all(x is not None for x in (a,b,c,d)):
                    diffs.append((a-b)-(c-d))
            metrics[metric] = stats.paired_bootstrap(diffs) if diffs else None
        transfer[f'{new}_minus_{old}'] = metrics
    gate=gate_result(results['B1']['variants'],source_comparability_passed=True)
    gate['meaning']='B1 prespecified descriptive development gate; not formal performance acceptance or statistical superiority.'
    audit.verify()
    result=dict(schema='ch3.safe_search.b1_transfer_analysis.v1',complete=True,independent_scenarios=20,
        new_b1_episodes=60,historical_b0_reference_episodes_used=60,variants=list(VARIANTS),
        results=results, transfer_effect_difference_B1_minus_B0=transfer, b1_gate=gate,
        scenario_rows=compact_rows,window_audit=all_windows,diagnostics=all_diagnostics,
        b1_physical_steps=sum(r['physical_steps'] for r in b1_rows),
        b1_source_inventory_sha256=parent['sources_before']['inventory']['sha256'],
        historical_B0_source_comparability=old_comparability,
        b1_preflight=preflight,
        source_verification_passed=True, source_hashes=audit.hashes,
        bootstrap=dict(replicates=plan['bootstrap_replicates'],seed=plan['bootstrap_seed'],unit='matched_original_scenario'),
        limitations=['Development scenes already used in B0 tuning; no independent generalization claim.',
            'Pooled rate changes and mean per-scene paired changes have different weights.',
            'Observation renewal and motion stall are diagnostic proxies, not detection probabilities.',
            'No training, checkpoint loading, HGR execution, heldout80 evaluation or performance acceptance.',
            'Intervals descriptive without multiplicity correction; transfer difference is effect heterogeneity, not mediation.'])
    output.mkdir(parents=True)
    write_json(output/'analysis.json',result)
    for baseline in ('B0','B1'):
        print(baseline,{v:dict(found=s['found_count'],success=s['success_count'],pre_collision=s['pre_found_collision_count'],
            stall_hold=s['stall_plus_hold_fraction'],effective=s['actual_belief_footprint']['effective_observation_fraction'],
            outcomes=s['outcome_partition']) for v,s in results[baseline]['variants'].items()},flush=True)
    print('B1_GATE',gate,flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    analyze(args.run,args.output)
