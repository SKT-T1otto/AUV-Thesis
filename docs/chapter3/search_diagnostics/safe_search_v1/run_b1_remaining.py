"""Frozen B1 V1/V2/V4 supplement; invokes the existing reviewed CLI unchanged.

This is experiment orchestration, not a simulator/controller implementation.
Own script and plan bytes are pinned separately from the unchanged 338-file
runtime inventory. Historical V0/V3/V5 and all B0 evidence remain read-only.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import sys
import time

from chapter3_bser.experiments.safe_search_v1.run_paired import ROOT, read_json, write_json, file_hash, output_directory, summarize
from chapter3_bser.experiments.safe_search_v1.run_development import INDICES, SEED, PLAN, validate_development_inputs, _run_child
from chapter3_bser.experiments.safe_search_v1.run_b1_development import collect_completed_b1, ZERO_FIELDS
from chapter3_bser.experiments.safe_search_v1.run_ac_development import validate_trace
from chapter3_bser.experiments.safe_search_v1.provenance import framework_sources, verify_sources

VARIANTS = ('V1', 'V2', 'V4')
BASELINE = 'B1_bser_prior'
OPTIONS = {
    'V1': dict(planning_state_consistency=True, failure_policy=False, path_safety=False),
    'V2': dict(planning_state_consistency=True, failure_policy=True, path_safety=False),
    'V4': dict(planning_state_consistency=True, failure_policy=True, path_safety=True),
}
HERE = Path(__file__).resolve().parent
SUPPLEMENT_PLAN = HERE / 'b1_remaining_plan_20260929.json'


def arm_directory(output, index, variant):
    return Path(output) / f'scene_{index:04d}_{variant}'


def child_command(manifest, config, output, index, variant):
    if index not in INDICES or variant not in VARIANTS:
        raise ValueError('Unplanned scene or variant')
    return [sys.executable, '-B', '-m', 'chapter3_bser.experiments.safe_search_v1.run_paired',
            '--manifest', str(manifest), '--config', str(config), '--output-dir', str(output),
            '--baseline', BASELINE, '--variants', variant, '--episode-indices', str(index),
            '--steps', '400', '--seed', str(SEED), '--full-episodes', '--search-coverage']


def validate_terminal(row, scene, variant):
    fields = {k: scene[k] for k in ('original_episode_index', 'scenario_id', 'scenario_seed', 'environment_innovation_seed')}
    fields.update(baseline=BASELINE, variant=variant, terminal=True,
                  full_episode_requested=True, full_episode_completed=True)
    if any(row.get(k) != v for k, v in fields.items()):
        raise ValueError('Terminal identity/completion mismatch')
    if any(row.get(k) is not True for k in ('terminal', 'full_episode_requested', 'full_episode_completed')):
        raise ValueError('Real terminal required')
    steps, found = row.get('physical_steps'), row.get('found_step')
    if (type(steps) is not int or not 1 <= steps <= 400 or
            found is not None and (type(found) is not int or not 0 <= found <= steps) or
            row.get('found_within_budget') is not (found is not None)):
        raise ValueError('Invalid step or Found count')
    result, reason = row.get('episode_result', {}), row.get('stop_reason')
    collision = result.get('first_collision_step')
    if (reason not in ('success', 'timeout', 'obstacle_collision') or
            reason != result.get('termination_reason') or result.get('terminal_step') != steps or
            result.get('task_protocol') != 'collision_terminal_v1' or
            reason == 'timeout' and steps != 400 or result.get('success') is not (reason == 'success') or
            reason == 'success' and found is None or
            (reason == 'obstacle_collision') != (collision is not None) or
            collision is not None and (type(collision) is not int or collision != steps)):
        raise ValueError('Inconsistent task outcome')
    exposure = found if found is not None else steps
    pre_collision = collision is not None and (found is None or collision <= found)
    if row.get('pre_found_exposure_steps') != exposure or row.get('pre_found_collision') is not pre_collision:
        raise ValueError('Inconsistent pre-Found outcome')
    names = ('searcher_motion_stall_proxy_agent_steps', 'searcher_hold_agent_steps')
    if any(type(row.get(k)) is not int or row[k] < 0 for k in names) or sum(row[k] for k in names) > 3 * exposure:
        raise ValueError('Invalid disjoint motion counts')
    complete = row.get('complete_episode_row') or {}
    expected = dict(method='ch3_baseline_bser_prior', reference_runtime_method='ch3_baseline_bser_prior',
        found=found is not None, found_step=found, success=reason == 'success', actual_length=steps,
        episode_length=steps, terminal_step=steps, termination_reason=reason,
        scenario_id=scene['scenario_id'], scenario_seed=scene['scenario_seed'],
        environment_innovation_seed=scene['environment_innovation_seed'],
        evaluation_episode_index=scene['original_episode_index'], optimizer_update_count=0,
        training_update=False, first_collision_step=collision)
    if any(k not in complete or complete[k] != v for k, v in expected.items()):
        raise ValueError('Complete episode evidence differs')
    controller = row.get('controller', {})
    if any(k not in controller or controller[k] != 0 for k in ZERO_FIELDS):
        raise ValueError('Nonzero learned action/residual/update')
    if controller.get('safe_search', {}).get('options') != OPTIONS[variant]:
        raise ValueError('Requested options not executed')
    coverage = row.get('search_coverage') or {}
    if coverage.get('available') is not True or coverage.get('pre_found_exposure_steps') != exposure:
        raise ValueError('Complete measured coverage required')


def validate_child(directory, scene, variant, parent):
    directory = Path(directory)
    if (directory / 'failure.json').exists():
        raise ValueError('Child program failure')
    identity, summary, rows = (read_json(directory / n) for n in ('identity.json', 'summary.json', 'episodes.json'))
    expected = dict(schema='ch3.safe_search.paired_identity.v1', baseline=BASELINE, variants=[variant],
        seed=SEED, steps=400, task_horizon=400, full_episodes=True, training=False, checkpoint_loaded=False,
        formal_thesis_evaluation=False, residual_source='zeros_4x3', selected=[scene], search_coverage_enabled=True,
        input_sha256=parent['child_input_sha256'], sources_before=parent['sources_before'],
        sources_after=parent['sources_before'], source_and_input_verification_passed=True)
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError('Child source/input/request mismatch')
    if (len(rows) != 1 or summary.get('all_requested_runs_recorded') is not True or
            summary.get('source_and_input_verification_passed') is not True or set(summary.get('variants', {})) != {variant}):
        raise ValueError('Incomplete child')
    row = rows[0]
    validate_terminal(row, scene, variant)
    arm = directory / f"episode_{scene['original_episode_index']:04d}" / variant
    if row != read_json(arm / 'summary.json'):
        raise ValueError('Arm/root row mismatch')
    if any(summary['variants'][variant].get(k) != v for k, v in
           dict(n_expected=1, n_recorded=1, n_terminal=1, full_episode_evaluation_complete=True).items()):
        raise ValueError('Incomplete arm counts')
    return row, validate_trace(arm / 'step_trace.jsonl', row)


def collect(output):
    output = Path(output)
    parent = read_json(output / 'identity.json')
    if ((output / 'failure.json').exists() or parent.get('experiment_complete') is not True or
            parent.get('source_and_input_verification_passed') is not True or
            parent.get('sources_before') != parent.get('sources_after') or
            parent.get('variants') != list(VARIANTS) or parent.get('planned_episode_runs') != 60 or
            parent.get('options') != OPTIONS or parent.get('baseline') != BASELINE or
            parent.get('seed') != SEED or parent.get('task_horizon') != 400 or
            [s['original_episode_index'] for s in parent['selected']] != list(INDICES)):
        raise ValueError('Invalid or incomplete supplement')
    rows, hashes = [], {}
    for scene in parent['selected']:
        for variant in VARIANTS:
            i = scene['original_episode_index']
            row, sha = validate_child(arm_directory(output, i, variant), scene, variant, parent)
            rows.append(row)
            hashes[f'{i}:{variant}'] = sha
    if rows != read_json(output / 'episodes.json') or hashes != parent['trace_sha256']:
        raise ValueError('Root rows or trace hashes differ')
    for path, sha in parent['input_sha256'].items():
        if file_hash(Path(path)) != sha:
            raise ValueError('Supplement input changed: ' + path)
    return parent, rows


def run(output, workers):
    if type(workers) is not int or not 1 <= workers <= 20:
        raise ValueError('Workers must be 1..20')
    plan = read_json(SUPPLEMENT_PLAN)
    manifest, config, reference = (ROOT / plan[k] for k in ('manifest', 'config', 'b1_reference'))
    old, old_rows = collect_completed_b1(reference)
    sources = framework_sources()
    selected = validate_development_inputs(read_json(manifest), read_json(config), read_json(PLAN))
    required = dict(variants=list(VARIANTS), options=OPTIONS, indices=list(INDICES), seed=SEED,
        max_steps=400, planned_episode_runs=60, training=False, checkpoint_loaded=False,
        source_inventory_sha256=sources['inventory']['sha256'])
    if any(plan.get(k) != v for k, v in required.items()):
        raise ValueError('Frozen supplement plan mismatch')
    if old['sources_before'] != sources or old['sources_after'] != sources or old['selected'] != selected or len(old_rows) != 60:
        raise ValueError('Reference runtime/scene identity mismatch')
    for relative, sha in plan['reference_hashes'].items():
        if file_hash(ROOT / relative) != sha:
            raise ValueError('Historical input changed')
    inputs = [manifest, config, PLAN, SUPPLEMENT_PLAN, Path(__file__), *(ROOT / p for p in plan['reference_hashes'])]
    hashes = {str(p.resolve()): file_hash(p) for p in inputs}
    output = output_directory(output, inputs)
    if output == reference or reference in output.parents:
        raise ValueError('Do not write inside historical evidence')
    parent = dict(schema='ch3.safe_search.b1_remaining_identity.v1', baseline=BASELINE,
        variants=list(VARIANTS), options=OPTIONS, selected=selected, seed=SEED, task_horizon=400,
        full_episodes=True, search_coverage_enabled=True, planned_episode_runs=60, independent_scenarios=20,
        max_physical_steps=24000, training=False, checkpoint_loaded=False, formal_thesis_evaluation=False,
        performance_passed=None, heldout_scenarios_executed=0, workers=workers,
        input_sha256=hashes, child_input_sha256={str(p.resolve()): file_hash(p) for p in (manifest, config)},
        sources_before=sources, python=sys.version, reference_directory=str(reference), trace_sha256={})

    def verify():
        verify_sources(sources)
        if any(file_hash(Path(p)) != sha for p, sha in hashes.items()):
            raise RuntimeError('Frozen inputs/orchestrator changed during execution')

    output.mkdir(parents=True)
    (output / 'logs').mkdir()
    write_json(output / 'identity.json', parent)
    pending = deque((scene, v) for scene in selected for v in VARIANTS)
    completed, failures, records = [], [], {}
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            running = {}
            while pending or running:
                while pending and len(running) < workers and not failures:
                    verify()
                    scene, variant = pending.popleft()
                    i = scene['original_episode_index']
                    cmd = child_command(manifest, config, arm_directory(output, i, variant), i, variant)
                    future = executor.submit(_run_child, cmd, output / 'logs' / f'scene_{i:04d}_{variant}.log')
                    running[future] = (scene, variant)
                    print(f'B1_REMAINING_START index={i} variant={variant}', flush=True)
                if not running:
                    break
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    scene, variant = running.pop(future)
                    i = scene['original_episode_index']
                    try:
                        process = future.result()
                        if process['returncode']:
                            raise RuntimeError(f"Child exit {process['returncode']}")
                        verify()
                        row, sha = validate_child(arm_directory(output, i, variant), scene, variant, parent)
                        records[i, variant] = row
                        parent['trace_sha256'][f'{i}:{variant}'] = sha
                        completed.append([i, variant])
                        print(f'B1_REMAINING_COMPLETE index={i} variant={variant} arms={len(completed)}/60', flush=True)
                    except Exception as exc:
                        failures.append(dict(index=i, variant=variant, exception=type(exc).__name__, message=str(exc)))
                        print(f'B1_REMAINING_FAILURE {i} {variant}: {exc}', flush=True)
                write_json(output / 'progress.json', dict(completed_pairs=sorted(completed),
                    running_pairs=sorted([s['original_episode_index'], v] for s, v in running.values()),
                    unstarted_pairs=[[s['original_episode_index'], v] for s, v in pending],
                    program_failures=failures, planned_episode_runs=60, experiment_complete=False))
        if failures or len(records) != 60:
            raise RuntimeError('Incomplete experiment; no performance verdict')
        verify()
        rows = [records[i, v] for i in INDICES for v in VARIANTS]
        parent.update(sources_after=framework_sources(), source_and_input_verification_passed=True, experiment_complete=True)
        final = summarize(rows, VARIANTS, 20, True)
        final.update(experiment_complete=True, planned_episode_runs=60, program_failures=[],
                     wall_seconds=time.perf_counter()-started, formal_thesis_evaluation=False)
        write_json(output / 'episodes.json', rows)
        write_json(output / 'summary.json', final)
        write_json(output / 'identity.json', parent)
        collect(output)
        write_json(output / 'progress.json', dict(completed_pairs=sorted(completed), running_pairs=[],
            unstarted_pairs=[], program_failures=[], planned_episode_runs=60, experiment_complete=True))
        print(f"B1_REMAINING_FINISHED scenes=20 arms=60 steps={sum(r['physical_steps'] for r in rows)}", flush=True)
    except BaseException as exc:
        write_json(output / 'failure.json', dict(exception=type(exc).__name__, message=str(exc),
            completed_pairs=sorted(completed), program_failures=failures, experiment_complete=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--execute', action='store_true', required=True)
    args = parser.parse_args()
    run(args.output_dir, args.workers)
