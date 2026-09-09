"""Tiny synthetic worker contracts; never the canonical smoke or real checkpoint."""
import copy
import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.experiments.phase1c_prrac.beds import BEDSEpisodeAdapter, DIAGNOSTIC_FILES, IDENTITY_FIELDS, validate_beds, write_diagnostics
from chapter3_bser.integration.control_context import AgentAssignmentContextV1, ExecutorAssignmentContextV1, BSERControlContextV1
from tests.bser_test_utils import synthetic_state
from tests.prrac_evaluation_support import worker_jobs, write_checkpoint


def config():
    result = evaluator._load_config(evaluator.ROOT/'configs/chapter3/bser_phase1c_beds.json')
    result['max_steps'] = 4
    result['standby_diagnostics_enabled'] = False
    return result


def job_fixture(path):
    torch.manual_seed(991)
    job = worker_jobs(write_checkpoint(path), 1)[0]
    job['config'] = config()
    job['scenario']['max_steps'] = 4
    job['checkpoint_info'].update(checkpoint_runtime_revision='dynamic_public_intercept_v3_atomic_continuity',
                                  evaluation_runtime_revision='dynamic_public_intercept_v3_atomic_continuity',
                                  runtime_integration_mode='native', execution_variant='B1_ATOMIC_LAST_VALID',
                                  search_recovery_variant='S2A1_C2_LOCAL_CONNECTOR')
    return job


class BEDSEvaluationTests(unittest.TestCase):
    def test_terminal_ranking_is_reported_without_claiming_an_action(self):
        allocator = SimpleNamespace(ranking_diagnostics=[dict(step=4, candidate_count=2,
            mean_time_discount=.8, top_candidate_changed=True)], candidate_diagnostics=[])
        adapter = BEDSEpisodeAdapter(config(), 'synthetic', 0, allocator)
        record = adapter.payload()['early_discovery_diagnostics.csv'][0]
        self.assertEqual(record['step'], 4)
        self.assertEqual(record['candidate_count'], 2)
        self.assertFalse(record['action_applied'])
        self.assertEqual(len(adapter.payload()['early_discovery_diagnostics.csv']), 1)

    def test_existing_sidecar_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            retained = output/'executor_standby_diagnostics.csv'
            retained.write_text('retained\n', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                evaluator.run_evaluation(config_path=evaluator.ROOT/'configs/chapter3/bser_phase1c_beds.json',
                                         output_dir=output, beds_variant_override='full')
            self.assertEqual(retained.read_text(encoding='utf-8'), 'retained\n')

    def test_config_defaults_and_four_combinations(self):
        cfg = config()
        self.assertFalse(cfg['early_discovery']['enabled'])
        self.assertFalse(cfg['executor_standby']['enabled'])
        for early, standby in ((False, False), (True, False), (False, True), (True, True)):
            cfg['early_discovery']['enabled'], cfg['executor_standby']['enabled'] = early, standby
            self.assertEqual(validate_beds(cfg)['early_discovery']['enabled'], early)
        cfg['search_value_guidance']['enabled'] = True
        with self.assertRaises(ValueError):
            validate_beds(cfg)

    def test_guidance_only_executor_changes_and_found_returns_original(self):
        state = synthetic_state()
        assignments = tuple(AgentAssignmentContextV1(a.agent_id, a.role, 'search', 'original',
                            a.position, (), a.position, a.position, False, True, False) for a in state.agents)
        executor = ExecutorAssignmentContextV1(3, 'original', state.agents[3].position, (),
                                                state.agents[3].position, state.agents[3].position, False, True, False)
        guidance = BSERControlContextV1('bser.control_context.v1', 'v1:test', 'test', 0, 'SEARCH', assignments, executor, 'test')
        cfg = config()
        cfg['executor_standby']['enabled'] = True
        adapter = BEDSEpisodeAdapter(cfg, 'synthetic', 0)
        from tests.test_beds_safe_standby import navigation
        adapter.navigation = navigation(state)
        modified = adapter.prepare_guidance(guidance, state)
        for before, after in zip(guidance.agent_assignments[:3], modified.agent_assignments[:3]):
            self.assertIs(before, after)
        self.assertEqual(modified.schema_version, guidance.schema_version)
        self.assertEqual(modified.executor_assignment.source, 'BEDS_SAFE_STANDBY')
        found_state = replace(state, target_found=True)
        self.assertIs(adapter.prepare_guidance(guidance, found_state), guidance)
        actions = torch.rand(4, 3)
        self.assertIs(adapter.before_action(actions, found_state), actions)

    def test_off_worker_matches_prechange_HEAD_bit_for_bit(self):
        root = evaluator.ROOT
        def old_module(path):
            source = subprocess.check_output(['git', '-c', 'safe.directory='+root.as_posix(), 'show', 'HEAD:'+path], cwd=root, text=True, encoding='utf-8')
            module = ModuleType('chapter3_bser.experiments.phase1c_prrac._beds_baseline')
            module.__file__ = str(root/path)
            # Evaluate the frozen prechange evaluator; it has no dataclass definitions.
            exec(compile(source, module.__file__, 'exec'), module.__dict__)
            return module
        old = old_module('chapter3_bser/experiments/phase1c_prrac/evaluate_prrac_checkpoints.py')
        with tempfile.TemporaryDirectory() as directory:
            job = job_fixture(Path(directory)/'synthetic.pt')
            absent = copy.deepcopy(job)
            absent['config'].pop('early_discovery')
            absent['config'].pop('executor_standby')
            expected = old._evaluate_episode_job(absent)
            current_absent = evaluator._evaluate_episode_job(absent)
            disabled = evaluator._evaluate_episode_job(job)
            self.assertEqual(evaluator._canonical_json(expected), evaluator._canonical_json(current_absent))
            self.assertEqual(evaluator._canonical_json(expected), evaluator._canonical_json(disabled))

    def test_all_active_worker_combinations_and_serializable_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            base = job_fixture(Path(directory)/'synthetic.pt')
            for early, standby in ((True, False), (False, True), (True, True)):
                job = copy.deepcopy(base)
                job['config']['early_discovery']['enabled'] = early
                job['config']['executor_standby']['enabled'] = standby
                job['config']['standby_diagnostics_enabled'] = standby
                result = evaluator._evaluate_episode_job(job)
                self.assertFalse(evaluator._contains_tensor(result))
                self.assertEqual(set(result['beds_diagnostics']), set(DIAGNOSTIC_FILES))
                records = result['beds_diagnostics']['executor_standby_diagnostics.csv']
                self.assertEqual(bool(records), standby)
                if standby:
                    self.assertGreaterEqual(records[0]['executor_action_norm'], 0)
                    self.assertLessEqual(records[0]['executor_action_norm'], np.sqrt(3)+1e-6)
                    if records[0]['hold_active']:
                        self.assertEqual(records[0]['executor_action_norm'], 0)
                    else:
                        self.assertTrue(records[0]['route_available'])
                output = Path(directory)/f'{early}-{standby}'
                output.mkdir()
                for name, rows in result['beds_diagnostics'].items():
                    write_diagnostics(output, name, rows)
                    with (output/name).open(newline='', encoding='utf-8') as handle:
                        parsed = list(csv.DictReader(handle))
                    self.assertEqual(len(parsed), len(rows))
                    # Resume replays must not duplicate sidecar rows.
                    write_diagnostics(output, name, parsed+rows)
                    with (output/name).open(newline='', encoding='utf-8') as handle:
                        self.assertEqual(len(list(csv.DictReader(handle))), len(rows))

    def test_smoke_cli_selects_variant_without_running_it(self):
        with patch.object(evaluator, 'run_evaluation', return_value={}) as run:
            evaluator.main(['--config', str(evaluator.ROOT/'configs/chapter3/bser_phase1c_beds.json'),
                            '--beds-variant', 'full', '--episodes', '10'])
        self.assertEqual(run.call_args.kwargs['beds_variant_override'], 'full')
        self.assertEqual(run.call_args.kwargs['episodes_override'], 10)


if __name__ == '__main__':
    unittest.main()
