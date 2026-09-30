"""Offline completion and six-version pairing; requires all 60 new terminals."""
from __future__ import annotations
import argparse
from collections import Counter
import inspect as introspection
import math
from pathlib import Path

from chapter3_bser.experiments.safe_search_v1 import analyze_development as stats
from chapter3_bser.experiments.safe_search_v1.run_paired import ROOT, write_json, file_hash
from chapter3_bser.experiments.safe_search_v1.run_b1_development import collect_completed_b1, reference_metadata
from chapter3_bser.experiments.safe_search_v1.run_ac_development import read_reference, collect_completed_ac
from chapter3_bser.experiments.safe_search_v1.run_development import INDICES
from chapter3_bser.experiments.safe_search_v1.provenance import framework_sources
from docs.chapter3.search_diagnostics.safe_search_v1 import run_b1_remaining as supplement
from docs.chapter3.search_diagnostics.safe_search_v1.analyze_b1_development import comparison
from docs.chapter3.search_diagnostics.safe_search_v1.analyze_ac_development import episode_metrics
from docs.chapter3.search_diagnostics.safe_search_v1.b1_case_audit import inspect as inspect_case
from docs.chapter3.search_diagnostics.safe_search_v1.development_baseline_audit import Inputs
from docs.chapter3.search_diagnostics.safe_search_v1.development_trace_audit import trace_rows

VERSIONS = ('V0','V1','V2','V3','V4','V5')


def independent_counts(rows, summaries):
    """Recompute headlines directly, independently of aggregate/paired helpers."""
    checked = {}
    for variant in VERSIONS:
        arm = [r for r in rows if r['variant'] == variant]
        assert len(arm) == 20
        found = len([r for r in arm if r['complete_episode_row']['found']])
        success = len([r for r in arm if r['stop_reason'] == 'success'])
        collisions = len([r for r in arm if r['episode_result']['first_collision_step'] is not None and
            (r['found_step'] is None or r['episode_result']['first_collision_step'] <= r['found_step'])])
        exposure = sum(r['found_step'] if r['found_step'] is not None else r['physical_steps'] for r in arm)
        stallhold = sum(r['searcher_motion_stall_proxy_agent_steps'] + r['searcher_hold_agent_steps'] for r in arm)
        effective = sum(r['search_coverage']['effective_observation_steps'] for r in arm)
        s = summaries[variant]
        assert (found, success, collisions, exposure) == (s['found_count'],s['success_count'],s['pre_found_collision_count'],s['pre_found_exposure_steps'])
        assert abs(stallhold/(3*exposure)-s['stall_plus_hold_fraction']) < 1e-12
        assert abs(effective/exposure-s['actual_belief_footprint']['effective_observation_fraction']) < 1e-12
        checked[variant] = dict(found=found,success=success,pre_found_collisions=collisions,
            exposure=exposure,stall_and_hold_agent_steps=stallhold,effective_observation_steps=effective)
    return checked


def analyze(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new analysis directory')
    audit = Inputs()
    for obj in (analyze, supplement.collect, stats.aggregate, comparison, episode_metrics, inspect_case, Inputs):
        audit.watch(Path(introspection.getfile(obj)))
    plan = audit.read(supplement.SUPPLEMENT_PLAN)
    new_parent, new_rows = supplement.collect(run)
    old_run = Path(new_parent['reference_directory'])
    old_parent, old_rows = collect_completed_b1(old_run)
    if not (framework_sources() == new_parent['sources_before'] == old_parent['sources_before'] == old_parent['sources_after']):
        raise ValueError('B1 runtime source changed across groups')
    if new_parent['selected'] != old_parent['selected']:
        raise ValueError('B1 population changed')
    previous = audit.read(supplement.HERE/'b1_results_20260928/analysis.json')
    for path, sha in previous['source_hashes'].items():
        if audit.watch(path) != sha:
            raise ValueError('Previously reviewed B1/B0 evidence changed')
    refs = old_parent['references']
    d2, ac = (Path(refs[k]['directory']) for k in ('B0_D2','B0_AC'))
    if reference_metadata(d2, ac) != refs:
        raise ValueError('B0 reference binding differs')
    _, b0_d2_rows, _ = read_reference(d2)
    _, b0_ac_rows = collect_completed_ac(ac)
    for directory in (run,old_run,d2,ac):
        for name in ('identity.json','episodes.json','summary.json'):
            audit.watch(directory/name)
    results, values, compact, cases, numerical = {}, {}, [], [], {}
    comparisons = plan['comparisons'] + plan['secondary_comparisons']
    for baseline, rows in (('B1', old_rows+new_rows), ('B0', b0_d2_rows+b0_ac_rows)):
        if len(rows)!=120 or {(r['original_episode_index'],r['variant']) for r in rows}!={(i,v) for i in INDICES for v in VERSIONS}:
            raise ValueError('Expected exact 20 x 6 matrix per baseline')
        traces = {}
        for row in rows:
            i, v = row['original_episode_index'], row['variant']
            if baseline == 'B1':
                directory = supplement.arm_directory(run if v in supplement.VARIANTS else old_run, i, v)
            else:
                directory = (ac if v=='V5' else d2)/f'scene_{i:04d}'
            path = directory/f'episode_{i:04d}'/v/'step_trace.jsonl'
            audit.watch(path)
            traces[i,v] = stats.trace_summary(path)
            if traces[i,v]['physical_steps'] != row['physical_steps']:
                raise ValueError('Trace length mismatch')
            if baseline == 'B1':
                trace = list(trace_rows(path))
                hold = sum(a['hold'] for t in trace if t['search_transition'] for a in t['before']['agents'][:3])
                effective = sum(t['search_coverage']['effective_observation_steps'] for t in trace)
                if hold != row['searcher_hold_agent_steps'] or effective != row['search_coverage']['effective_observation_steps']:
                    raise ValueError('Coverage/Hold trace mismatch')
                c = inspect_case(trace,row)
                post_queries=Counter()
                for t in trace:
                    if not t['search_transition']:
                        for q in t['queries']['counts']:
                            post_queries[q['caller']+':'+q['reason']] += q['count']
                c['post_found']['query_callers'] = dict(post_queries)
                cases.append(c)
            compact.append(dict(baseline=baseline, index=i, variant=v,
                scenario_id=row['scenario_id'],scenario_seed=row['scenario_seed'],
                environment_innovation_seed=row['environment_innovation_seed'],
                outcome=stats.outcome(row),found_step=row['found_step'],physical_steps=row['physical_steps'],
                metrics=episode_metrics(row,traces[i,v])))
            print(f'B1_SIX_ANALYZED baseline={baseline} index={i} variant={v}',flush=True)
        vals={(r['original_episode_index'],r['variant']):episode_metrics(r,traces[r['original_episode_index'],r['variant']]) for r in rows}
        values[baseline] = vals
        summaries={v:stats.aggregate([r for r in rows if r['variant']==v],traces) for v in VERSIONS}
        for v,s in summaries.items():
            s['stall_plus_hold_fraction']=(s['searcher_motion_stall_proxy_agent_steps']+s['searcher_hold_agent_steps'])/(3*s['pre_found_exposure_steps'])
            queries=Counter()
            for (i,variant),t in traces.items():
                if variant==v:queries.update(t['pre_found_query_callers'])
            s['pre_found_query_callers']=dict(queries)
            if v in ('V0','V3','V5'):
                old=previous['results'][baseline]['variants'][v]
                for name in ('found_count','success_count','pre_found_collision_count','outcome_partition',
                             'pre_found_exposure_steps','stall_plus_hold_fraction','actual_belief_footprint','pre_found_query_reasons'):
                    if s[name]!=old[name]:raise ValueError('Previous analysis no longer reconciles: '+name)
        results[baseline]=dict(variants=summaries,comparisons={f'{a}_minus_{b}':comparison(vals,a,b) for a,b in comparisons})
        numerical[baseline]=independent_counts(rows,summaries)
    indexed={(r['baseline'],r['index'],r['variant']):r for r in compact}
    for i in INDICES:
        for v in VERSIONS:
            a,b=indexed['B0',i,v],indexed['B1',i,v]
            assert all(a[k]==b[k] for k in ('scenario_id','scenario_seed','environment_innovation_seed'))
    transfer={}
    for new,old in comparisons:
        transfer[f'{new}_minus_{old}']={}
        for metric in ('found','success','pre_found_collision','stall_plus_hold_fraction','effective_observation_fraction'):
            diffs=[]
            for i in INDICES:
                x=[values[base][i,v][metric] for base,v in (('B1',new),('B1',old),('B0',new),('B0',old))]
                if all(v is not None for v in x):diffs.append(x[0]-x[1]-x[2]+x[3])
            transfer[f'{new}_minus_{old}'][metric]=stats.paired_bootstrap(diffs) if diffs else None
    # Exploratory trace explanation, separate from the prespecified effect sizes.
    pair_mechanisms=[]
    for i,old,new in ((0,'V1','V2'),(3,'V5','V4')):
        data=[]
        for variant in (old,new):
            base=run if variant in supplement.VARIANTS else old_run
            path=supplement.arm_directory(base,i,variant)/f'episode_{i:04d}'/variant/'step_trace.jsonl'
            audit.watch(path)
            data.append(list(trace_rows(path)))
        guidance=None;physical=None
        for t,u in zip(*data):
            changed=[j for j in range(4) if any(t['after']['agents'][j][k]!=u['after']['agents'][j][k]
                for k in ('hold','reachable','semantic_waypoint','tracking_waypoint'))]
            if changed and guidance is None:
                guidance=dict(step=t['step_after'],agents=changed,
                    observations=[dict(variant=v,decision_after=x['after']['decision_reason'],
                        cached_map_revision_after=x['after']['cached_map_revision'],
                        live_map_revision_after=x['after']['live_map_revision'],queries=x['queries']['counts'],
                        agents=[{k:a[k] for k in ('agent_id','hold','reachable','semantic_waypoint','tracking_waypoint','position')}
                                for a in x['after']['agents']]) for v,x in ((old,t),(new,u))])
            delta=max(math.dist(x['position'],y['position']) for x,y in zip(t['after']['agents'],u['after']['agents']))
            if delta>1e-7 and physical is None:
                physical=dict(step=t['step_after'],max_agent_position_difference=delta,tolerance=1e-7)
        pair_mechanisms.append(dict(index=i,old=old,new=new,first_guidance_difference=guidance,
            first_physical_difference=physical,common_prefix_steps=min(map(len,data)),
            interpretation_limit='Exploratory observed ordering; not a proof that one isolated event alone caused the final outcome.'))
    audit.verify()
    output.mkdir(parents=True)
    result=dict(schema='ch3.safe_search.b1_six_versions.v1',complete=True,new_b1_episodes=60,
        retained_b1_episodes=60,combined_b1_episodes=120,independent_scenarios=20,
        new_physical_steps=sum(r['physical_steps'] for r in new_rows),
        combined_b1_physical_steps=sum(r['physical_steps'] for r in new_rows+old_rows),
        results=results,scenario_rows=compact,transfer_effect_difference_B1_minus_B0=transfer,
        source_inventory_sha256=new_parent['sources_before']['inventory']['sha256'],
        program_failures=[],source_verification_passed=True,source_hashes=audit.hashes,
        bootstrap=dict(replicates=10000,seed=20260928,unit='paired_scene'),
        limitations=['All20 scenes are reused development scenes; not independent formal thesis evaluation.',
            'Descriptive paired intervals without multiplicity correction; n=20, not120 independent scenes.',
            'Pooled rates differ in weighting from mean per-scene paired changes.',
            'Effective observation is coverage renewal, not target detection probability.',
            'No training, checkpoint loading, HGR, heldout80 or performance acceptance.'])
    write_json(output/'analysis.json',result)
    write_json(output/'cases.json',dict(complete=True,episodes=120,cases=cases))
    write_json(output/'pair_mechanisms.json',dict(cases=pair_mechanisms,exploratory=True))
    write_json(output/'numeric_audit.json',dict(passed=True,method='Independent complete-episode counters and denominator recomputation',checks=numerical))
    print('B1_SIX_ANALYSIS_COMPLETE',flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();analyze(a.run,a.output)
