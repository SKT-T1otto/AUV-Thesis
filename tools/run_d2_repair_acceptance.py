"""Repeatable, bounded implementation experiments. Default invocation is a plan.

Exit 0: plan only, or all requested tests AND acceptance gates passed.
Exit 1: source drift, test failure/error/skip, missing evidence or timeout.
Exit 2: executed mechanism checks pass, but full acceptance remains blocked.
No result from this tool establishes training performance.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import unittest
import uuid

ROOT=Path(__file__).resolve().parents[1]
MATRIX=ROOT/'tests/d2_repair_v2_experiment.json'


def write_json(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2)
        stream.write('\n')


def matrix():
    value=json.loads(MATRIX.read_text(encoding='utf-8'))
    ids=[s['id'] for s in value['stages']]
    if value['schema']!='ch3.d2.repair_test_experiment.v1' or len(ids)!=len(set(ids)):
        raise ValueError('invalid experiment matrix')
    for stage in value['stages']:
        if not stage['tests'] or any(not t.startswith('tests.') for t in stage['tests']):
            raise ValueError('only explicit repository test cases are permitted')
        if type(stage['expected_tests']) is not int or stage['expected_tests']<1:
            raise ValueError('expected test count must be explicit')
    return value


def source_check():
    from chapter3_bser.experiments.d2_repair_v2.provenance import verify_reviewed_changes
    from tools.verify_d2_repair_v2_source import verify
    return dict(identity=verify_reviewed_changes(),independent=verify())


def test_identity():
    paths=list((ROOT/'tests').rglob('*.py'))+list((ROOT/'tests/fixtures').glob('*.py.txt'))+[MATRIX,Path(__file__).resolve()]
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def git_info():
    result={}
    for name,args in [('head',['rev-parse','HEAD']),('branch',['branch','--show-current']),('status',['status','--short','--branch'])]:
        p=subprocess.run(['git','-c','core.fsmonitor=false']+args,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
        result[name]=dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
    return result


def select_stages(quick=False):
    return [s for s in matrix()['stages'] if not quick or not s['full_only']]


def output_directory(requested=None):
    path=(Path(requested) if requested else ROOT/'tmp/d2_repair_acceptance'/
          (datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])).resolve()
    if ROOT/'tmp' not in path.parents:
        raise ValueError('test evidence must use a NEW directory below this repository tmp/')
    if path.exists():
        raise FileExistsError('refusing to overwrite previous evidence: '+str(path))
    return path


class RecordedResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs): super().__init__(*args,**kwargs); self.successes=[]
    def addSuccess(self,test): super().addSuccess(test); self.successes.append(test.id())


def child(stage,output):
    destination=output/(stage['id']+'.json')
    if destination.exists(): raise FileExistsError(destination)
    import numpy
    import torch
    torch.set_num_threads(1)
    started=time.perf_counter()
    suite=unittest.defaultTestLoader.loadTestsFromNames(stage['tests'])
    result=unittest.TextTestRunner(verbosity=2,resultclass=RecordedResult).run(suite)
    passed=(result.wasSuccessful() and not result.skipped and result.testsRun==stage['expected_tests'])
    record=dict(stage=stage['id'],command=[sys.executable]+sys.argv,tests_run=result.testsRun,
        expected_tests=stage['expected_tests'],successes=result.successes,
        failures=[dict(test=t.id(),traceback=tb) for t,tb in result.failures],
        errors=[dict(test=t.id(),traceback=tb) for t,tb in result.errors],
        skipped=[dict(test=t.id(),reason=r) for t,r in result.skipped],
        exit_code=0 if passed else 1,wall_seconds=time.perf_counter()-started,
        environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
            torch=torch.__version__,numpy=numpy.__version__,device='cpu',threads=torch.get_num_threads(),
            variables={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','AUV_REPAIR_TEST_EVIDENCE')}),
        is_ci=False,performance_evidence=False)
    write_json(destination,record)
    return record['exit_code']


def run_stage(stage,output):
    command=[sys.executable,'-m','tools.run_d2_repair_acceptance','--_stage',stage['id'],'--output',str(output)]
    env=os.environ.copy()
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): env[name]='1'
    env['PYTHONIOENCODING']='utf-8'
    env['AUV_REPAIR_TEST_EVIDENCE']=str(output/'artifacts')
    started=time.perf_counter(); timeout=False
    with (output/(stage['id']+'.log')).open('x',encoding='utf-8') as log:
        try:
            p=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
            code=p.returncode
        except subprocess.TimeoutExpired:
            timeout=True; code=1
            log.write('\nSTAGE TIMEOUT: 900 seconds; process terminated.\n')
    report=output/(stage['id']+'.json')
    valid=False
    if report.is_file():
        data=json.loads(report.read_text(encoding='utf-8'))
        valid=(data['exit_code']==0 and data['tests_run']==stage['expected_tests'] and not data['skipped'])
    return dict(stage=stage['id'],command=command,exit_code=code if code else (0 if valid else 1),
        report=report.name,log=stage['id']+'.log',wall_seconds=time.perf_counter()-started,timeout=timeout)


def acceptance_gates():
    gates=[]
    from tools.ch3_baselines.framework_provenance import framework_sources
    from chapter3_bser.experiments.d2_repair_v1.provenance import verify_reviewed_changes
    for name,check in [('legacy_baseline_source',framework_sources),('legacy_repair_v1_source',verify_reviewed_changes)]:
        try:
            check()
        except Exception as exc:
            gates.append(dict(gate=name,passed=False,error=repr(exc),traceback=traceback.format_exc()))
        else: gates.append(dict(gate=name,passed=True))
    golden=ROOT/'experiments/chapter3/e0_core_migration/golden_trace_manifest.json'
    gates.append(dict(gate='historical_e0',passed=False,file_present=golden.is_file(),
        reason='Historical golden data missing' if not golden.is_file() else 'Golden equivalence not executed by this bounded test runner',
        replacement_generated=False))
    return gates


def artifact_oracles(output,quick):
    folder=output/'artifacts'
    rows=json.loads((folder/'negative_controls.json').read_text(encoding='utf-8'))
    if ({r['defect'] for r in rows}!=set(matrix()['negative_controls']) or
            not all(r['old_control_detected'] is True and r['current_repair_passed'] is True for r in rows)):
        raise AssertionError('negative controls did not cover every declared defect')
    details=dict(negative_control_checks=len(rows),distinct_defects=len({r['defect'] for r in rows}))
    if not quick:
        runtime=json.loads((folder/'runtime_chain.json').read_text(encoding='utf-8'))
        checkpoints=json.loads((folder/'scheduler_checkpoint.json').read_text(encoding='utf-8'))
        if {(r['algorithm'],r['repair']) for r in runtime}!={('b2',False),('b3',False),('b2',True),('b3',True)}:
            raise AssertionError('missing B2/B3 legacy or repaired runtime evidence')
        required={'pre_found','discovery','waiting_handoff','executor_informed','frozen_searchers','success','obstacle_collision','timeout'}
        for r in runtime:
            if (r['update_rounds']>4 or not required<=set(r['coverage']) or
                    not all(1<=s['steps']<=32 for s in r['scenarios']) or
                    not r['rollout_replay_alignment'] or not r['team_reward_exact'] or
                    r['reward_weights_changed'] or r['networks_changed']):
                raise AssertionError('runtime evidence violates budget or semantic contract')
            if r['repair'] and any(v!=0 for v in r['frozen_actor_deltas']):
                raise AssertionError('frozen Actor changed')
        if len(checkpoints)!=2 or {r['algorithm'] for r in checkpoints}!={'b2','b3'}:
            raise AssertionError('missing independent B2/B3 scheduler evidence')
        for r in checkpoints:
            if (r['environment_steps']!=256 or r['episodes']!=8 or r['update_rounds']!=2 or
                    r['actor_steps']!=[2]*4 or r['critic_steps']!=16 or
                    not all(r[k] for k in ('default_hyperparameters_unchanged','checkpoint_source_checked',
                       'default_formal_load_rejected','heldout_evaluation_complete','source_drift_save_rejected'))):
                raise AssertionError('scheduler/checkpoint evidence disagrees')
        details.update(runtime_models=len(runtime),runtime_steps=sum(r['physical_steps'] for r in runtime),
                       scheduler_models=len(checkpoints),scheduler_steps=sum(r['environment_steps'] for r in checkpoints))
    return details


def execute(output,quick=False):
    output.mkdir(parents=True,exist_ok=False)
    before_tests=test_identity(); started=time.perf_counter()
    snapshot=dict(git=git_info(),test_files=before_tests,matrix=matrix(),command=[sys.executable]+sys.argv,
                  is_ci=False,outputs_modified=False,user_models_loaded=False,long_training=False)
    write_json(output/'before.json',snapshot)
    stages=[]; source_before=None; source_after=None; error=None; oracles=None
    try:
        source_before=source_check()
        write_json(output/'source_before.json',source_before)
        for stage in select_stages(quick):
            if source_check()!=source_before: raise ValueError('source identity changed before '+stage['id'])
            print('START '+stage['id'],flush=True)
            record=run_stage(stage,output); stages.append(record)
            print('FINISH '+stage['id']+' exit='+str(record['exit_code']),flush=True)
        source_after=source_check()
        if source_after!=source_before or test_identity()!=before_tests:
            raise ValueError('production or test identity changed during verification')
        if all(s['exit_code']==0 for s in stages): oracles=artifact_oracles(output,quick)
    except Exception as exc:
        error=dict(error=repr(exc),traceback=traceback.format_exc())
    gates=acceptance_gates()  # Read-only; never start the rejected legacy paths.
    passed=bool(stages) and all(s['exit_code']==0 for s in stages) and error is None and oracles is not None
    complete=passed and not quick
    ready=complete and all(g['passed'] for g in gates)
    code=1 if not passed else (0 if ready else 2)
    status='REJECTED' if not passed else ('MECHANISM_VERIFIED' if complete else 'PARTIAL_CHECKS_PASSED')
    result=dict(schema='ch3.d2.repair_test_results.v1',status=status,executed=True,quick=quick,
        mechanism_checks_passed=passed,full_mechanism_suite_complete=complete,formal_training_allowed=False,
        performance_passed=False,exit_code=code,stages=stages,source_before=source_before,source_after=source_after,
        error=error,acceptance_gates=gates,artifact_oracles=oracles,
        not_run=[s['id'] for s in matrix()['stages'] if s['id'] not in {x['stage'] for x in stages}],
        wall_seconds=time.perf_counter()-started,is_ci=False,
        evidence_sha256={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in output.rglob('*') if p.is_file() and p.suffix in ('.json','.log')})
    write_json(output/'results.json',result)
    print(json.dumps(dict(status=status,exit_code=code,report=str(output/'results.json'),
                         mechanism_checks_passed=passed,formal_training_allowed=False)),flush=True)
    return code


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Run bounded tests with newly initialized temporary models')
    parser.add_argument('--quick',action='store_true',help='Omit runtime and scheduler tests; report them as NOT RUN')
    parser.add_argument('--output',help='New directory below repository tmp/; existing paths are rejected')
    parser.add_argument('--_stage',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args._stage:
        stage=next((s for s in matrix()['stages'] if s['id']==args._stage),None)
        if stage is None or not args.output: raise ValueError('unknown internal test stage')
        path=Path(args.output).resolve()
        if ROOT/'tmp' not in path.parents or not path.is_dir(): raise ValueError('invalid internal evidence directory')
        return child(stage,path)
    output=output_directory(args.output)
    if not args.execute:
        print(json.dumps(dict(executed=False,output=str(output),stages=select_stages(args.quick),
                              bounds=matrix()['bounds'],performance_evidence=False),indent=2))
        return 0
    return execute(output,quick=args.quick)


if __name__=='__main__': raise SystemExit(main())
