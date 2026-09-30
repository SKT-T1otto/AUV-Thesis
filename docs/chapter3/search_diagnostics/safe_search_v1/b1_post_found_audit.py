"""Reconcile post-Found executor failure from complete B1 V5 traces."""
import argparse
from collections import Counter
from pathlib import Path

from chapter3_bser.experiments.safe_search_v1.run_b1_development import collect_completed_b1,arm_directory
from chapter3_bser.experiments.safe_search_v1.run_paired import write_json
from docs.chapter3.search_diagnostics.safe_search_v1.development_baseline_audit import Inputs
from docs.chapter3.search_diagnostics.safe_search_v1.development_trace_audit import trace_rows


def audit(run,output):
    run,output=Path(run),Path(output)
    if output.exists():raise ValueError('Output must be new')
    inputs=Inputs();inputs.watch(Path(__file__));inputs.watch(run/'episodes.json')
    _,rows=collect_completed_b1(run);result=[]
    for row in rows:
        if row['variant']!='V5' or row['found_step'] is None:continue
        i=row['original_episode_index'];path=arm_directory(run,i,'V5')/f'episode_{i:04d}/V5/step_trace.jsonl'
        inputs.watch(path);post=[t for t in trace_rows(path) if not t['search_transition']]
        counts=Counter();first_failure=None
        for t in post:
            for q in t['queries']['counts']:counts[q['caller']+':'+q['reason']]+=q['count']
            if first_failure is None:
                samples=[q for q in t['queries']['samples'] if q['agent_id']==3 and q['reason']=='no_start_connector']
                if samples:
                    first_failure=dict(step_before=t['step_before'],step_after=t['step_after'],
                        cached_map_revision=t['after']['cached_map_revision'],live_map_revision=t['after']['live_map_revision'],
                        decision_after=t['after']['decision_reason'],query_samples=samples,
                        executor_reachable_before=t['before']['agents'][3]['reachable'],
                        executor_reachable_after=t['after']['agents'][3]['reachable'])
        result.append(dict(index=i,found_step=row['found_step'],outcome=row['stop_reason'],
            post_found_steps=len(post),unreachable_steps=sum(t['before']['agents'][3]['reachable'] is False for t in post),
            query_callers=dict(counts),first_no_start_connector=first_failure))
    timeouts=[r for r in result if r['outcome']=='timeout']
    total=sum(r['post_found_steps'] for r in timeouts);unreachable=sum(r['unreachable_steps'] for r in timeouts)
    inputs.verify()
    write_json(output,dict(schema='ch3.safe_search.b1_post_found_audit.v1',complete=True,variant='V5',
        found_episode_count=len(result),post_found_timeout_count=len(timeouts),episodes=result,
        timeout_post_found_steps=total,timeout_unreachable_steps=unreachable,
        timeout_unreachable_fraction=unreachable/total,source_hashes=inputs.hashes,
        scope='All B1 V5 Found episodes; only transitions whose before state is Found. Query samples use recorded public endpoints.',
        limitation='The recorded no_start_connector and absent start endpoint establish planning-connection failure; they do not establish physical goal unreachability or test a repair.'))
    print('POST_FOUND_TIMEOUT_UNREACHABLE',unreachable,total,unreachable/total)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.run,a.output)
