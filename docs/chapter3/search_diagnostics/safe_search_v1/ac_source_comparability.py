"""Full-episode unchanged-arm replay audit for the additive A+C experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from chapter3_bser.experiments.safe_search_v1.run_paired import (
    ROOT, output_directory, read_json, write_json, file_hash)
from chapter3_bser.experiments.safe_search_v1.run_development import (
    collect_completed, _run_child, BASELINE, SEED)
from chapter3_bser.experiments.safe_search_v1.provenance import framework_sources, verify_sources


PLAN = ROOT / 'docs/chapter3/search_diagnostics/safe_search_v1/ac_experiment_plan.json'


def run(manifest, config, reference, output, review):
    manifest, config, reference, review = map(lambda p: Path(p).resolve(), (manifest, config, reference, review))
    old_identity, old_rows = collect_completed(reference)
    plan = read_json(PLAN)
    assert file_hash(reference / 'identity.json') == plan['reference_identity_sha256']
    assert file_hash(reference / 'episodes.json') == plan['reference_episodes_sha256']
    sources = framework_sources()
    review_data = read_json(review)
    assert review_data['review_passed'] is True
    assert review_data['new_source_inventory_sha256'] == sources['inventory']['sha256']
    assert review_data['reference_source_inventory_sha256'] == old_identity['sources_before']['inventory']['sha256']
    frozen = {str(p): file_hash(p) for p in (manifest, config, PLAN, review)}
    output = output_directory(output, (manifest, config, PLAN, review, reference))
    output.mkdir(parents=True)
    controls = plan['source_comparability']['old_arm_replay_controls']
    write_json(output / 'request.json', dict(plan=plan, input_sha256=frozen, sources=sources))

    def one(control):
        index, variant = control['original_episode_index'], control['variant']
        target = output / f'scene_{index:04d}_{variant}'
        command = [sys.executable, '-B', '-m', 'chapter3_bser.experiments.safe_search_v1.run_paired',
            '--manifest', str(manifest), '--config', str(config), '--output-dir', str(target),
            '--baseline', BASELINE, '--variants', variant, '--episode-indices', str(index),
            '--steps', '400', '--seed', str(SEED), '--full-episodes', '--search-coverage']
        result = _run_child(command, output / f'scene_{index:04d}_{variant}.log')
        if result['returncode']:
            raise RuntimeError(f'control {index}/{variant} returned {result["returncode"]}')
        identity = read_json(target / 'identity.json')
        rows = read_json(target / 'episodes.json')
        previous = next(r for r in old_rows if r['original_episode_index'] == index and r['variant'] == variant)
        if len(rows) != 1:
            raise ValueError('control must have exactly one terminal arm')
        current = rows[0]
        expected_selection = next(x for x in old_identity['selected'] if x['original_episode_index'] == index)
        assert identity['selected'] == [expected_selection]
        assert identity['sources_before'] == identity['sources_after'] == sources
        assert identity['source_and_input_verification_passed'] is True
        assert identity['seed'] == SEED and identity['variants'] == [variant]
        assert identity['full_episodes'] is True and identity['steps'] == 400
        assert identity['search_coverage_enabled'] is True
        assert identity['training'] is False and identity['checkpoint_loaded'] is False
        assert read_json(target / 'resolved_config.json') == read_json(reference / f'scene_{index:04d}' / 'resolved_config.json')
        assert read_json(target / 'evaluation_manifest.json') == read_json(reference / f'scene_{index:04d}' / 'evaluation_manifest.json')
        assert current['terminal'] is True and current['full_episode_completed'] is True
        assert current['full_episode_requested'] is True
        fields = ('physical_steps', 'found_step', 'found_within_budget', 'pre_found_collision',
                  'stop_reason', 'signature_sha256')
        assert isinstance(current['signature_sha256'], str) and len(current['signature_sha256']) == 64
        mismatched = [k for k in fields if current[k] != previous[k]]
        semantic_row = lambda row: {k: v for k, v in row['complete_episode_row'].items() if k != 'wall_seconds'}
        if semantic_row(current) != semantic_row(previous):
            mismatched.append('complete_episode_row_except_wall_seconds')
        if mismatched:
            raise ValueError(f'control {index}/{variant} differs: {mismatched}')
        trace = target / f'episode_{index:04d}' / variant / 'step_trace.jsonl'
        assert sum(1 for line in trace.open(encoding='utf-8') if line.strip()) == current['physical_steps']
        print(f'AC_CONTROL_EQUAL index={index} variant={variant} steps={current["physical_steps"]}', flush=True)
        return dict(original_episode_index=index, variant=variant, replay_directory=str(target),
            physical_steps=current['physical_steps'], physical_signature_sha256=current['signature_sha256'],
            equal_ordered_signature_digest=True, equal_key_outcome=True,
            identity_sha256=file_hash(target / 'identity.json'), summary_sha256=file_hash(target / 'summary.json'))

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(one, controls))
        verify_sources(sources)
        assert all(file_hash(Path(name)) == sha for name, sha in frozen.items())
        result = dict(schema='ch3.safe_search.ac_source_comparability.v1', passed=True,
            reviewed_exact_source_diff_passed=True, new_source_inventory_sha256=sources['inventory']['sha256'],
            reference_source_inventory_sha256=old_identity['sources_before']['inventory']['sha256'],
            review_file=str(review), review_sha256=file_hash(review), controls=results,
            plan_sha256=file_hash(PLAN), training=False, checkpoint_loaded=False,
            controls_included_in_new_V5_denominator=False)
        write_json(output / 'comparability.json', result)
        print('AC_SOURCE_COMPARABILITY_PASS', flush=True)
    except BaseException as exc:
        write_json(output / 'failure.json', dict(exception_type=type(exc).__name__, message=str(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'config', 'reference-dir', 'output-dir', 'review'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run(args.manifest, args.config, args.reference_dir, args.output_dir, args.review)


if __name__ == '__main__':
    main()
