"""Reject wrong options, truncated episodes and nonzero learned actions."""
import copy
from pathlib import Path
import tempfile
import unittest
from docs.chapter3.search_diagnostics.safe_search_v1 import run_b1_remaining as s
from tests.test_safe_search_b1_development import write_child


class RemainingContract(unittest.TestCase):
    def setUp(self):
        self.scene = dict(original_episode_index=0, scenario_id='fixture', scenario_seed=1,
                          environment_innovation_seed=12729)
        with tempfile.TemporaryDirectory() as directory:
            self.row = write_child(Path(directory)/'arm', self.scene, 'V5',
                                   dict(child_input_sha256={}, sources_before={}))
        self.row['variant'] = 'V4'
        self.row['controller']['safe_search']['options'] = copy.deepcopy(s.OPTIONS['V4'])

    def test_valid_and_exact_variants(self):
        s.validate_terminal(self.row, self.scene, 'V4')
        for v in s.VARIANTS:
            command=s.child_command('m','c','o',97,v)
            self.assertEqual(command[command.index('--variants')+1],v)
            self.assertIn('--full-episodes',command)
        for i,v in ((1,'V4'),(97,'V5')):
            with self.assertRaises(ValueError): s.child_command('m','c','o',i,v)

    def test_wrong_method_or_options_rejected(self):
        r=copy.deepcopy(self.row);r['complete_episode_row']['method']='ch3_baseline_search_prior'
        with self.assertRaises(ValueError):s.validate_terminal(r,self.scene,'V4')
        r=copy.deepcopy(self.row);r['controller']['safe_search']['options']['failure_policy']=False
        with self.assertRaises(ValueError):s.validate_terminal(r,self.scene,'V4')

    def test_learned_or_physical_residual_rejected(self):
        for name in s.ZERO_FIELDS:
            r=copy.deepcopy(self.row);r['controller'][name]=1
            with self.subTest(name=name),self.assertRaises(ValueError):s.validate_terminal(r,self.scene,'V4')

    def test_partial_and_short_timeout_rejected(self):
        for key in ('terminal','full_episode_requested','full_episode_completed'):
            r=copy.deepcopy(self.row);r[key]=False
            with self.subTest(key=key),self.assertRaises(ValueError):s.validate_terminal(r,self.scene,'V4')
        r=copy.deepcopy(self.row);r['stop_reason']='timeout';r['episode_result'].update(termination_reason='timeout',success=False)
        with self.assertRaises(ValueError):s.validate_terminal(r,self.scene,'V4')

    def test_wrong_seed_or_exposure_rejected(self):
        for key,value in (('environment_innovation_seed',12730),('pre_found_exposure_steps',0),
                          ('searcher_hold_agent_steps',4),('found_within_budget',False)):
            r=copy.deepcopy(self.row);r[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):s.validate_terminal(r,self.scene,'V4')


if __name__=='__main__':unittest.main()
