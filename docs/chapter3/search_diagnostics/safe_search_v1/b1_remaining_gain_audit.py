"""Exploratory positive-case evidence, separately from frozen paired effects."""
from pathlib import Path
import math
from chapter3_bser.experiments.safe_search_v1.run_paired import ROOT, write_json
from docs.chapter3.search_diagnostics.safe_search_v1.development_baseline_audit import Inputs
from docs.chapter3.search_diagnostics.safe_search_v1.development_trace_audit import trace_rows

HERE=Path(__file__).resolve().parent
OUT=HERE/'b1_remaining_results_20260929'


def audit():
    target=OUT/'gain_cases.json'
    if target.exists():raise ValueError('Output must be new')
    inputs=Inputs();inputs.watch(__file__)
    analysis=inputs.read(OUT/'analysis.json')
    assert analysis['complete'] and analysis['new_b1_episodes']==60
    roots={'V5':ROOT/'runs/safe_search_v1/development_B1_20260928_v1',
           'V4':ROOT/'runs/safe_search_v1/development_B1_remaining_20260929_v1'}
    rows={}
    for variant,root in roots.items():
        p=root/'episodes.json';data=inputs.read(p)
        assert inputs.hashes[str(p.resolve())]==analysis['source_hashes'][str(p.resolve())]
        rows[variant]={r['original_episode_index']:r for r in data if r['variant']==variant}
    cases=[]
    for i in (63,67,77):
        traces={};arms={}
        assert rows['V5'][i]['found_step'] is None and rows['V4'][i]['found_step'] is not None
        for variant,root in roots.items():
            p=root/f'scene_{i:04d}_{variant}/episode_{i:04d}/{variant}/step_trace.jsonl'
            assert inputs.watch(p)==analysis['source_hashes'][str(p.resolve())]
            trace=list(trace_rows(p));traces[variant]=trace
            recovery=[dict(step=t['step_after'],reason=t['after']['decision_reason'],
                found_after=t['after']['found'],
                agents_after=[{k:a[k] for k in ('agent_id','semantic_waypoint','hold','reachable')} for a in t['after']['agents']])
                for t in trace if t['after']['decision_reason']=='SAFE_SEARCH_RECOVERY' and
                                 t['before']['decision_reason']!='SAFE_SEARCH_RECOVERY']
            r=rows[variant][i]
            arms[variant]=dict(found_step=r['found_step'],terminal_step=r['physical_steps'],
                reason=r['stop_reason'],pre_found_exposure_steps=r['pre_found_exposure_steps'],
                safe_counts=r['controller']['safe_search']['counts'],observed_recovery_transitions=recovery)
        difference=None
        for t,u in zip(traces['V5'],traces['V4']):
            distance=max(math.dist(a['position'],b['position']) for a,b in zip(t['after']['agents'],u['after']['agents']))
            if distance>1e-7:
                difference=dict(step=t['step_after'],max_agent_position_difference=distance);break
        cases.append(dict(index=i,arms=arms,first_physical_difference=difference))
    inputs.verify()
    write_json(target,dict(complete=True,exploratory=True,cases=cases,source_hashes=inputs.hashes,
        limitations=['Cases selected after observing gains, not an independent effect estimate.',
            'Raw counts cover different pre-Found exposure lengths and are not direct query-efficiency comparisons.',
            'Recovery timing and final outcome are observed; one event is not proven to explain the whole episode.']))
    print('Positive-case audit passed')


if __name__=='__main__':audit()
