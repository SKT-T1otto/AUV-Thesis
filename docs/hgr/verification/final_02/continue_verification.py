"""Recover only complete method logs after the user interrupted the runner.

This is the exact evidence-recovery recipe, not production implementation.
The source tree, discovery list, method identities, counts and success footer
must all match. Original child exit codes were lost with the runner; they are
recorded as unavailable rather than invented. Missing modules run normally.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import platform
import re
import sys
import time
import unittest

import numpy as np
import torch
from tools import verify_collision_terminal as verify


def main():
    output = verify.ROOT / 'docs/hgr/verification/final_02'
    continuation = output / 'continuation'
    continuation.mkdir(exist_ok=False)
    started, timer = verify.utc(), time.perf_counter()
    before = json.loads((output / 'tree_before.json').read_text(encoding='utf-8'))
    if verify.tree_identity() != before:
        raise RuntimeError('interrupted-run source tree changed; cannot reuse any prior results')
    modules = ['.'.join(p.relative_to(verify.ROOT).with_suffix('').parts)
               for p in sorted((verify.ROOT / 'tests').rglob('test_*.py'))]
    discovered = [t.id() for t in verify.flatten(unittest.defaultTestLoader.loadTestsFromNames(modules))]
    prior_discovery = json.loads((output / 'current/discovery.json').read_text(encoding='utf-8'))
    if sorted(discovered) != sorted(prior_discovery) or len(discovered) != len(set(discovered)):
        raise RuntimeError('discovery changed or contains duplicate methods')
    recovered = []
    for path in sorted((output / 'current').glob('*.log')):
        content = path.read_text(encoding='utf-8')
        identities = [cls + '.' + method for method, cls in re.findall(
            r'^(\S+) \(([^)]+)\) \.\.\. ok\s*$', content, re.M)]
        expected = [name for name in discovered if name.startswith(path.stem + '.')]
        counts = re.findall(r'^Ran (\d+) tests? in ([0-9.]+)s\s+OK\s*(?:\n|$)', content, re.M)
        if (not expected or sorted(identities) != sorted(expected) or len(counts) != 1
                or int(counts[0][0]) != len(expected)):
            raise RuntimeError('incomplete or mismatched retained module log: ' + str(path))
        recovered.append(dict(module=path.stem, passed=True, tests_run=len(expected),
            passed_methods=len(expected), ran_ids=identities, success_ids=identities,
            failure_records=[], error_records=[], skipped_records=[],
            unsuccessful_method_count=0, subtest_failure_records=0,
            exit_code=None, process_exit_code_available=False,
            evidence_origin='complete_retained_unittest_log_exact_method_match',
            log_path=path.relative_to(verify.ROOT).as_posix(),
            log_sha256=verify.sha(path.read_bytes())))
    completed = {r['module'] for r in recovered}
    missing = [m for m in modules if m not in completed]
    verify.dump(continuation / 'recovery_inventory.json', dict(
        source_tree_sha256=before['sha256'], source_tree_unchanged=True,
        completed_modules=len(recovered), completed_methods=sum(r['tests_run'] for r in recovered),
        remaining_modules=missing, discovered_methods=len(discovered),
        original_process_exit_codes_available=False,
        reason='User interrupted the original runner; all processes stopped. Only exact complete OK logs are reused.'))
    verify.dump(continuation / 'recovered_methods.json', recovered)
    fresh = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(verify.module_job, name, continuation / name) for name in missing]
        for future in as_completed(jobs):
            result = future.result()
            fresh.append(result)
            verify.dump(continuation / 'new_methods.json', fresh)
            print(f"continued {len(fresh)}/{len(missing)} {result['module']}: "
                  f"{result['passed_methods']}/{result['tests_run']} exit={result['exit_code']}", flush=True)
    combined = recovered + fresh
    ran = [name for r in combined for name in r.get('ran_ids', [])]
    current_passed = (all(r['passed'] for r in recovered)
        and all(r['passed'] and r['exit_code'] == 0 for r in fresh)
        and sorted(ran) == sorted(discovered))
    verify.dump(output / 'current/methods.json', combined)
    current = dict(discovered=len(discovered), tests_run=sum(r['tests_run'] for r in combined),
        passed_methods=sum(r['passed_methods'] for r in combined),
        failure_records=[v for r in combined for v in r.get('failure_records', [])],
        error_records=[v for r in combined for v in r.get('error_records', [])],
        skipped_records=[v for r in combined for v in r.get('skipped_records', [])],
        unsuccessful_method_count=sum(r.get('unsuccessful_method_count', 0) for r in combined),
        subtest_failure_records=sum(r.get('subtest_failure_records', 0) for r in combined),
        blocked=0, exact_discovery_execution_match=sorted(ran) == sorted(discovered),
        recovered_modules=len(recovered), fresh_modules=len(fresh),
        recovered_process_exit_codes_available=False,
        modules=[{k: r.get(k) for k in ('module', 'passed', 'exit_code', 'evidence_origin')} for r in combined],
        passed=current_passed, continued_after_user_interruption=True)
    verify.dump(output / 'current/summary.json', current)
    reports = dict(current=current)
    for name in ('legacy', 'historical', 'golden'):
        child = output / name
        child.mkdir(exist_ok=False)
        suite_started = verify.utc()
        try:
            result = getattr(verify, name)(child)
        except FileNotFoundError as exc:
            result = dict(passed=False, status='blocked_missing_historical_input', blocked=1, missing_input=str(exc))
        except Exception:
            import traceback
            result = dict(passed=False, status='runner_error', traceback=traceback.format_exc())
        result.update(started_utc=suite_started, ended_utc=verify.utc())
        verify.dump(child / 'summary.json', result)
        reports[name] = result
        print(f"{name}: passed={result['passed']} status={result.get('status', 'completed')}", flush=True)
    after = verify.tree_identity()
    verify.dump(output / 'tree_after.json', after)
    report = dict(context='latest_recorded_local_verification_not_CI',
        original_command='python -B -m tools.verify_collision_terminal --suite all --workers 4 --output-dir docs/hgr/verification/final_02',
        continuation_recipe='docs/hgr/verification/final_02/continue_verification.py',
        head=verify.git('rev-parse', 'HEAD').stdout.decode().strip(), suite='all',
        source_tree_sha256=before['sha256'], tree_unchanged=before == after,
        platform=platform.platform(), python=sys.version, executable=sys.executable,
        dependencies=dict(torch=torch.__version__, numpy=np.__version__),
        continuation_started_utc=started, ended_utc=verify.utc(), continuation_wall_seconds=time.perf_counter()-timer,
        continued_after_user_interruption=True, recovered_modules=len(recovered), fresh_modules=len(fresh),
        recovered_process_exit_codes_available=False,
        current_suite_passed=current_passed, legacy_compatibility_passed=reports['legacy']['passed'],
        historical_suite_status=reports['historical'].get('status', 'completed'),
        golden_status=reports['golden'].get('status', 'completed'),
        all_required_verification_passed=all(r['passed'] for r in reports.values()) and before == after,
        passed=all(r['passed'] for r in reports.values()) and before == after)
    report['exit_code'] = 0 if report['passed'] else 1
    verify.dump(output / 'summary.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
