"""Fixed-tree local current/historical/legacy/golden verification.

regression is a compatibility alias for current; all includes all four suites.
Historical inputs are read only. Every invocation requires a new output path.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = '93a9c8fb53857051390265e3035061bf05402e25'
EXCLUDED = {'outputs', 'runs', 'checkpoints', '.git', '__pycache__', '.pytest_cache'}


def utc(): return datetime.now(timezone.utc).isoformat()
def sha(data): return hashlib.sha256(data).hexdigest()
def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
def git(*args, check=True):
    return subprocess.run(['git', '-c', 'safe.directory='+ROOT.as_posix(), *args], cwd=ROOT,
                          check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
def tracked_and_new():
    return sorted(set(git('ls-files', '--cached', '--others', '--exclude-standard', '-z').stdout.decode().split('\0'))-{''})
def source_files():
    for name in tracked_and_new():
        path = ROOT/name
        if (not path.is_file() or EXCLUDED.intersection(Path(name).parts)
                or name.startswith('docs/collision_terminal/maintenance/')
                or path.suffix in {'.log', '.pt', '.pth', '.pyc'}): continue
        if path.suffix in {'.py', '.json', '.ps1', '.bat', '.sh', '.yml', '.yaml', '.toml', '.csv'} or path.name in {'AGENTS.md', '.gitignore', '.gitattributes'}:
            yield name
def tree_identity():
    files = {name: sha((ROOT/name).read_bytes()) for name in source_files()}
    return {'sha256': sha(json.dumps(files, sort_keys=True).encode()), 'files': files}
def flatten(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite): yield from flatten(test)
        else: yield test


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.methods, self.success_ids = [], []
    def startTest(self, test):
        self.methods.append(test.id()); super().startTest(test)
    def addSuccess(self, test):
        self.success_ids.append(test.id()); super().addSuccess(test)


def run_methods(names, output):
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    ids = [t.id() for t in flatten(suite)]
    if not ids or len(ids) != len(set(ids)): raise ValueError('zero or duplicate discovered test methods')
    started = utc()
    with (output/'unittest.log').open('w', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordingResult).run(suite)
    report = {'discovered': len(ids), 'discovered_ids': ids, 'tests_run': result.testsRun,
        'ran_ids': result.methods, 'passed_methods': len(result.success_ids), 'success_ids': result.success_ids,
        'failure_records': [{'id': t.id(), 'traceback': tb} for t, tb in result.failures],
        'error_records': [{'id': t.id(), 'traceback': tb} for t, tb in result.errors],
        'skipped_records': [{'id': t.id(), 'reason': why} for t, why in result.skipped],
        'unsuccessful_method_count': result.testsRun-len(result.success_ids)-len(result.skipped),
        'subtest_failure_records': sum(isinstance(t, unittest.case._SubTest) for t, _ in result.failures+result.errors),
        'blocked': 0, 'started_utc': started, 'ended_utc': utc(),
        'passed': bool(ids) and result.wasSuccessful() and not result.skipped}
    dump(output/'methods.json', report)
    return report


def module_job(module, output):
    with tempfile.TemporaryDirectory(prefix='auv-verify-module-') as temporary:
        scratch = Path(temporary)
        command = [sys.executable, '-B', '-m', 'tools.verify_collision_terminal', '--_module', module, '--output-dir', str(scratch)]
        with (scratch/'process.log').open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
        path = scratch/'methods.json'
        report = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {
            'passed': False, 'tests_run': 0, 'passed_methods': 0, 'error_records': [{'id': module, 'traceback': 'child process did not write methods.json'}]}
        with Path(str(output)+'.log').open('w', encoding='utf-8') as stream:
            for name in ('unittest.log', 'process.log'):
                if (scratch/name).is_file(): stream.write((scratch/name).read_text(encoding='utf-8'))
        report.update(command=command, exit_code=result.returncode, module=module)
    return report


def current(output, workers):
    modules = ['.'.join(p.relative_to(ROOT).with_suffix('').parts) for p in sorted((ROOT/'tests').rglob('test_*.py'))]
    discovered = [t.id() for t in flatten(unittest.defaultTestLoader.loadTestsFromNames(modules))]
    if not discovered or len(discovered) != len(set(discovered)): raise ValueError('zero or duplicate current methods')
    dump(output/'discovery.json', discovered)
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(module_job, name, output/name) for name in modules]
        for future in as_completed(jobs):
            r = future.result(); results.append(r)
            print(f"{len(results)}/{len(modules)} {r['module']}: {r['passed_methods']}/{r['tests_run']} exit={r['exit_code']}", flush=True)
    ran = [name for r in results for name in r.get('ran_ids', [])]
    dump(output/'methods.json', results)
    return {'discovered': len(discovered), 'tests_run': sum(r['tests_run'] for r in results),
        'passed_methods': sum(r['passed_methods'] for r in results),
        'failure_records': [v for r in results for v in r.get('failure_records', [])],
        'error_records': [v for r in results for v in r.get('error_records', [])],
        'skipped_records': [v for r in results for v in r.get('skipped_records', [])],
        'subtest_failure_records': sum(r.get('subtest_failure_records', 0) for r in results),
        'unsuccessful_method_count': sum(r.get('unsuccessful_method_count', 0) for r in results),
        'blocked': 0, 'exact_discovery_execution_match': sorted(ran) == sorted(discovered),
        'modules': [{'module': r['module'], 'passed': r['passed'], 'exit_code': r['exit_code']} for r in results],
        'passed': all(r['passed'] and r['exit_code'] == 0 for r in results) and sorted(ran) == sorted(discovered)}


def historical(output):
    modules = ['.'.join(p.relative_to(ROOT).with_suffix('').parts) for p in sorted((ROOT/'historical_verification').glob('test_*.py'))]
    tests = list(flatten(unittest.defaultTestLoader.loadTestsFromNames(modules)))
    runnable, blocked = [], []
    for test in tests:
        missing = []
        if 'Phase1AV1FrozenTest' in test.id():
            manifest = json.loads((ROOT/'docs/chapter3_bser/phase1a1/phase1a_v1_freeze_before.json').read_text())
            missing = [REFERENCE+':'+r['path'] for r in manifest['files'] if git('cat-file', '-e', REFERENCE+':'+r['path'], check=False).returncode]
        elif 'E0DeliveryTests' in test.id():
            missing = [str(ROOT/'experiments/chapter3/e0_equivalence'/n) for n in ('equivalence_summary.json', 'per_trajectory_results.csv') if not (ROOT/'experiments/chapter3/e0_equivalence'/n).is_file()]
        elif 'Phase1CV2OverlayTests' in test.id():
            path = ROOT/'docs2/phase1c_v2_design/overlay_manifest.json'
            if not path.is_file(): missing = [str(path)]
        if missing: blocked.append({'id': test.id(), 'status': 'blocked_missing_historical_input', 'missing': missing})
        else: runnable.append(test.id())
    report = run_methods(runnable, output) if runnable else {'tests_run': 0, 'passed_methods': 0, 'passed': False}
    report.update(discovered=len(tests), discovered_ids=[t.id() for t in tests], blocked=len(blocked), blocked_records=blocked,
                  status='blocked_missing_historical_input' if blocked else ('passed' if report['passed'] else 'failed'))
    report['passed'] = report['passed'] and not blocked
    return report


def export_reference(destination):
    if git('cat-file', '-e', REFERENCE+'^{commit}', check=False).returncode:
        git('fetch', '--no-tags', 'origin', REFERENCE, check=False)
    archive = git('archive', REFERENCE, check=False)
    if archive.returncode: raise FileNotFoundError('blocked_missing_fixed_reference_commit: '+REFERENCE)
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as source:
        for member in source.getmembers():
            if not (destination/member.name).resolve().is_relative_to(destination.resolve()) or member.issym() or member.islnk():
                raise ValueError('unexpected archive path/link')
        source.extractall(destination)


def legacy(output):
    baseline = json.loads((ROOT/'tests/legacy_baseline.json').read_text())
    assert baseline['reference_commit'] == REFERENCE
    if git('cat-file', '-e', REFERENCE+':core/env/task_protocol.py', check=False).returncode == 0:
        raise ValueError('reference unexpectedly includes collision-terminal protocol')
    reports = {}
    with tempfile.TemporaryDirectory(prefix='auv-fixed-legacy-') as temporary:
        for label in ('reference', 'current'):
            target = Path(temporary)/label; target.mkdir()
            if label == 'reference': export_reference(target)
            else:
                for name in tracked_and_new():
                    path = ROOT/name
                    if not path.is_file() or EXCLUDED.intersection(Path(name).parts) or name.startswith('docs/collision_terminal/maintenance/'): continue
                    (target/name).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target/name)
            shutil.copyfile(ROOT/'tests/legacy_behavior_probe.py', target/'_legacy_probe.py')
            result_path = output/(label+'.json')
            command = [sys.executable, '-E', '-s', '-B', '-m', '_legacy_probe', str(result_path)]
            env = {k: v for k, v in os.environ.items() if not k.upper().startswith('PYTHON')}
            with (output/(label+'.log')).open('w', encoding='utf-8') as stream:
                result = subprocess.run(command, cwd=target, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
            reports[label] = {'command': command, 'exit_code': result.returncode, 'source_root': str(target)}
            if result.returncode: return {'passed': False, 'processes': reports, 'status': 'probe_failed'}
        left, right = [json.loads((output/(label+'.json')).read_text(encoding='utf-8')) for label in ('reference', 'current')]
        isolation_left, isolation_right = left.pop('isolation'), right.pop('isolation')
        assert isolation_left['root'] != isolation_right['root']
        ml, mr = isolation_left['project_modules'], isolation_right['project_modules']
        additions = sorted(set(mr)-set(ml))
        identity = right.pop('protocol_identity')
        from core.env.task_protocol import protocol_identity
        identity_valid = identity == protocol_identity({}) and 'protocol_identity' not in left
        module_valid = set(ml) <= set(mr) and set(additions) <= set(baseline['allowed_new_import_modules']) and all(ml[k] == mr[k] for k in ml)
        differences = [key for key in sorted(left.keys() | right.keys()) if left.get(key) != right.get(key)]
        return {'passed': not differences and identity_valid and module_valid, 'reference_commit': REFERENCE,
            'processes': reports, 'comparison': baseline['comparison'], 'different_top_level_fields': differences,
            'legacy_identity_valid': identity_valid, 'dependency_isolation_valid': module_valid,
            'added_import_modules': additions, 'discovered': 1, 'tests_run': 1,
            'bounded_steps_per_source': len(left['transitions']),
            'reference_payload_sha256': sha((output/'reference.json').read_bytes()),
            'current_payload_sha256': sha((output/'current.json').read_bytes())}


def golden_profile(profile):
    import torch
    torch.set_num_threads(1)
    from tools.run_core_golden_e0 import _run_profile
    return _run_profile(profile)
def golden(output):
    from tools.run_core_golden_e0 import GOLDEN_PATH, PROFILES
    if not GOLDEN_PATH.is_file():
        return {'status': 'blocked_missing_historical_input', 'missing_input': str(GOLDEN_PATH),
                'trajectories': 0, 'passed_trajectories': 0, 'blocked': 1, 'passed': False}
    rows, mismatches = [], []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for result in executor.map(golden_profile, PROFILES):
            rows.extend(result['rows']); mismatches.extend(result['mismatches'])
    dump(output/'trajectories.json', rows)
    return {'trajectories': len(rows), 'passed_trajectories': sum(r['passed'] for r in rows),
            'mismatches': mismatches, 'passed': len(rows) == 60 and all(r['passed'] for r in rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('current', 'regression', 'historical', 'legacy', 'golden', 'all'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--_module', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
    import torch
    import numpy as np
    torch.set_num_threads(1)
    if args._module:
        return 0 if run_methods([args._module], output)['passed'] else 1
    if not args.suite or args.workers < 1: parser.error('suite and positive workers required')
    output.mkdir(parents=True, exist_ok=False)
    started, timer = utc(), time.perf_counter()
    before = tree_identity(); dump(output/'tree_before.json', before)
    selected = ('current', 'legacy', 'historical', 'golden') if args.suite == 'all' else (('current' if args.suite == 'regression' else args.suite),)
    reports = {}
    for name in selected:
        child = output/name; child.mkdir(); suite_started = utc()
        try: result = current(child, args.workers) if name == 'current' else globals()[name](child)
        except FileNotFoundError as exc:
            result = {'passed': False, 'status': 'blocked_missing_historical_input', 'blocked': 1, 'missing_input': str(exc)}
        except Exception:
            import traceback
            result = {'passed': False, 'status': 'runner_error', 'traceback': traceback.format_exc()}
        result.update(started_utc=suite_started, ended_utc=utc())
        dump(child/'summary.json', result); reports[name] = result
        print(f"{name}: passed={result['passed']} status={result.get('status', 'completed')}", flush=True)
    after = tree_identity(); dump(output/'tree_after.json', after)
    report = {'context': 'latest_recorded_local_verification_not_CI',
        'command': [sys.executable, '-m', 'tools.verify_collision_terminal', *sys.argv[1:]],
        'head': git('rev-parse', 'HEAD').stdout.decode().strip(), 'suite': args.suite,
        'source_tree_sha256': before['sha256'], 'tree_unchanged': before == after,
        'platform': platform.platform(), 'python': sys.version, 'executable': sys.executable,
        'dependencies': {'torch': torch.__version__, 'numpy': np.__version__},
        'started_utc': started, 'ended_utc': utc(), 'wall_seconds': time.perf_counter()-timer,
        'current_suite_passed': reports.get('current', {}).get('passed'),
        'legacy_compatibility_passed': reports.get('legacy', {}).get('passed'),
        'historical_suite_status': reports.get('historical', {}).get('status', 'not_run'),
        'golden_status': reports.get('golden', {}).get('status', 'passed' if reports.get('golden', {}).get('passed') else 'not_run'),
        'all_required_verification_passed': len(reports) == 4 and all(r['passed'] for r in reports.values()) and before == after,
        'passed': all(r['passed'] for r in reports.values()) and before == after}
    report['exit_code'] = 0 if report['passed'] else 1
    dump(output/'summary.json', report); print(json.dumps(report, indent=2), flush=True)
    return report['exit_code']


if __name__ == '__main__': raise SystemExit(main())
