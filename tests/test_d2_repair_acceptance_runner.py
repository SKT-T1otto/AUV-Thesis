"""The acceptance harness must fail closed and preserve earlier evidence."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

from tools import run_d2_repair_acceptance as runner


class AcceptanceRunner(unittest.TestCase):
    def test_matrix_covers_all_defects_and_expensive_stages_are_explicit(self):
        value=runner.matrix()
        self.assertEqual(len(set(value['negative_controls'])),13)
        self.assertEqual(len(runner.select_stages(False)),7)
        self.assertEqual({s['id'] for s in runner.select_stages(False)}-
                         {s['id'] for s in runner.select_stages(True)},
                         {'runtime_chain','scheduler_checkpoint'})
        self.assertEqual(value['bounds']['multiagent_update_rounds_per_model'],4)
        self.assertEqual(value['bounds']['environment_steps_per_scene'],32)

    def test_default_plan_never_creates_output_or_starts_tests(self):
        target=runner.ROOT/'tmp'/('uncreated_repair_plan_'+uuid.uuid4().hex)
        stream=io.StringIO()
        with patch.object(runner,'run_stage',side_effect=AssertionError('test launched')),redirect_stdout(stream):
            self.assertEqual(runner.main(['--output',str(target)]),0)
        self.assertFalse(target.exists())
        self.assertFalse(json.loads(stream.getvalue())['executed'])

    def test_existing_output_and_escape_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'tmp',prefix='repair_runner_guard_') as temporary:
            marker=Path(temporary)/'preserve.txt'; marker.write_text('keep')
            with self.assertRaises(FileExistsError): runner.output_directory(temporary)
            self.assertEqual(marker.read_text(),'keep')
        with self.assertRaises(ValueError): runner.output_directory(runner.ROOT/'outputs'/'forbidden_test_run')

    def test_source_failure_prevents_every_model_stage(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'tmp',prefix='repair_runner_guard_') as temporary:
            target=Path(temporary)/'run'
            with patch.object(runner,'source_check',side_effect=ValueError('source rejected')), \
                 patch.object(runner,'acceptance_gates',return_value=[]), \
                 patch.object(runner,'git_info',return_value={}), \
                 patch.object(runner,'run_stage') as run,redirect_stdout(io.StringIO()):
                self.assertEqual(runner.execute(target),1)
            run.assert_not_called()
            result=json.loads((target/'results.json').read_text(encoding='utf-8'))
            self.assertEqual(result['status'],'REJECTED')
            self.assertIn('source rejected',result['error']['traceback'])
            self.assertFalse(result['formal_training_allowed'])

    def test_failed_child_cannot_be_reported_as_passed(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'tmp',prefix='repair_runner_guard_') as temporary:
            target=Path(temporary)/'run'
            with patch.object(runner,'source_check',return_value={'id':'same'}), \
                 patch.object(runner,'acceptance_gates',return_value=[]), \
                 patch.object(runner,'git_info',return_value={}), \
                 patch.object(runner,'select_stages',return_value=[{'id':'failed'}]), \
                 patch.object(runner,'run_stage',return_value={'stage':'failed','exit_code':1}), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(runner.execute(target),1)
            result=json.loads((target/'results.json').read_text(encoding='utf-8'))
            self.assertFalse(result['mechanism_checks_passed'])
            self.assertFalse(result['full_mechanism_suite_complete'])

    def test_test_identity_drift_invalidates_otherwise_successful_child(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'tmp',prefix='repair_runner_guard_') as temporary:
            target=Path(temporary)/'run'
            with patch.object(runner,'source_check',return_value={'id':'same'}), \
                 patch.object(runner,'acceptance_gates',return_value=[]), \
                 patch.object(runner,'git_info',return_value={}), \
                 patch.object(runner,'test_identity',side_effect=[{'test':'before'},{'test':'after'}]), \
                 patch.object(runner,'select_stages',return_value=[{'id':'ok'}]), \
                 patch.object(runner,'run_stage',return_value={'stage':'ok','exit_code':0}), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(runner.execute(target),1)
            result=json.loads((target/'results.json').read_text(encoding='utf-8'))
            self.assertEqual(result['status'],'REJECTED')
            self.assertIn('identity changed',result['error']['traceback'])


if __name__=='__main__': unittest.main(verbosity=2)
