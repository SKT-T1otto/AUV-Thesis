"""Bounded real AUV acceptance. No injected Found, rewards, labels or summaries."""
import copy
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch

from chapter3_bser.experiments.hgr.train import load_config, load_checkpoint, COST_FIELDS, DEFAULT_CONFIG, Trainer
from chapter3_bser.experiments.hgr.runtime import collect_trajectory, continue_branch, DecisionSnapshot, MissionRuntime
from chapter3_bser.models.hgr.policy import HandoffPolicy, weights_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'tests/fixtures/hgr/integration_config.json'


def spawned_continuation(snapshot, policy_config, policy_state):
    torch.set_num_threads(1)
    policy=HandoffPolicy(policy_config); policy.load_state_dict(policy_state,strict=True)
    return continue_branch(snapshot,policy,keep_records=True)


def run_entry(arguments):
    result=subprocess.run([sys.executable,'-B','-m','chapter3_bser.experiments.hgr.cli',*map(str,arguments)],
                          cwd=ROOT,env=os.environ.copy(),capture_output=True,text=True,timeout=900)
    if result.returncode:
        raise AssertionError(result.stdout+'\n'+result.stderr)
    return result


class HGRIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_boundary_snapshot_exact_continuation_and_spawn(self):
        config=load_config(CONFIG)
        scenario=json.loads((ROOT/config['scenario_manifest']).read_text(encoding='utf-8'))['scenarios'][0]
        torch.manual_seed(9); policy=HandoffPolicy(config['policy'])
        trajectory=collect_trajectory(config,scenario,policy,seed=321)
        self.assertIsNotNone(trajectory['tau'])
        self.assertGreater(trajectory['tau'],trajectory['summary']['found_step'])
        self.assertEqual(trajectory['summary']['handoff_decision_step'],trajectory['tau'])
        self.assertEqual(trajectory['summary']['actual_length'],config['max_steps'])
        snapshot=trajectory['snapshot']
        with tempfile.TemporaryDirectory(prefix='hgr-snapshot-') as temporary:
            path=Path(temporary)/'boundary.snapshot'; snapshot.save(path)
            loaded=DecisionSnapshot.load(path); self.assertEqual(loaded.sha256,snapshot.sha256)
            restored=MissionRuntime.restore(loaded)
            self.assertEqual(restored.step,trajectory['tau'])
            self.assertEqual(restored.env.reward_accounting.steps,trajectory['tau'])
            self.assertEqual(restored.env.reward_adapter.contact_bonus_count,0)
            self.assertEqual(restored.provider.cached.step,trajectory['tau'])
            self.assertEqual(restored.env.get_mapping_state().obstacle_knowledge_mode,'online_unknown')
            np.testing.assert_array_equal(restored.features(),trajectory['features'])
            restored.close()
            local=continue_branch(loaded,policy,keep_records=True)
            with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn')) as pool:
                spawned=pool.submit(spawned_continuation,loaded,config['policy'],policy.state_dict()).result(timeout=180)
        original=trajectory['records'][trajectory['tau']:]
        for result in (local,spawned):
            self.assertEqual(result['steps'],len(original))
            self.assertEqual(result['terminal_step'],config['max_steps'])
            self.assertEqual(result['termination_reason'],'timeout')
            for expected,actual in zip(original,result['records']):
                for key in ('observations','actions','latents','next_observations'):
                    if key=='latents':
                        self.assertEqual(actual[key][:3],[None]*3)
                        np.testing.assert_array_equal(actual[key][3],expected[key][3])
                    else:
                        np.testing.assert_array_equal(actual[key],expected[key])
                self.assertEqual(actual['team_reward'],expected['team_reward'])
                self.assertEqual(actual['dones'],expected['dones'])
        for record in trajectory['records']:
            self.assertEqual(record['final_reward_by_agent'],[record['team_reward']]*4)
            self.assertEqual(record['reward_transform_applied_count'],1)
        self.assertTrue(any(r['team_reward'] != 0 for r in original))
        self.assertTrue(all(not any(r['dones']) for r in trajectory['records'][:-1]))

    def test_production_two_cycles_resume_and_matching_baseline_entries(self):
        with tempfile.TemporaryDirectory(prefix='hgr-production-') as temporary:
            base=Path(temporary)/'collision_terminal'
            output=base/'hgr'
            run_entry(['train','--config',CONFIG,'--output-dir',output])
            summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
            cycles=json.loads((output/'cycles.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['completed_cycles'],2)
            self.assertEqual(summary['actual_total_environment_steps'],sum(summary['costs'][k] for k in COST_FIELDS))
            self.assertGreater(summary['costs']['snapshot_restore_count'],0)
            for row in cycles:
                self.assertEqual(row['handoff_count'],1)
                self.assertNotEqual(row['phi0_hash'],row['phi1_hash'])
                self.assertNotEqual(row['theta_hash'],row['theta_after_hash'])
                self.assertGreater(row['g0_norm'],0)
                self.assertGreater(row['predictor_fit']['updates'],0)
                self.assertGreater(row['prediction_norm'],0)
                self.assertGreater(row['correction_norm'],0)
                self.assertEqual(row['draws'][0]['index'],row['draws'][1]['index'])
                self.assertEqual(len(set(row['suffix_dataset_ids']+row['pilot_dataset_ids']+row['main_dataset_ids'])),3)
            branches=json.loads((output/'branches.json').read_text(encoding='utf-8'))
            streams=[row[side]['random_stream_id'] for row in branches for side in ('old','new')]
            self.assertEqual(len(streams),len(set(streams)))
            checkpoint=output/'checkpoints/hgr_main_000001_cycle_000001.pt'
            resumed=base/'resumed'
            run_entry(['resume','--config',CONFIG,'--checkpoint',checkpoint,'--output-dir',resumed])
            final=load_checkpoint(summary['latest_checkpoint'])
            again=load_checkpoint(json.loads((resumed/'summary.json').read_text(encoding='utf-8'))['latest_checkpoint'])
            self.assertEqual(final['theta_hash'],again['theta_hash'])
            self.assertEqual(final['phi_hash'],again['phi_hash'])
            self.assertEqual(final['costs'],again['costs'])
            self.assertFalse(final['prefixes_valid'])
            self.assertTrue(final['cycle_complete'])
            for method in ('stochastic_direct_mc','direct_boundary_corrected'):
                directory=base/method
                run_entry(['train','--config',CONFIG,'--algorithm',method,'--total-main-trajectories','1','--output-dir',directory])
                row=json.loads((directory/'cycles.json').read_text(encoding='utf-8'))[0]
                self.assertGreater(row['g0_norm'],0)
                self.assertGreater(row['parameter_change_norm'],0)
                if method=='direct_boundary_corrected':
                    self.assertGreater(row['costs']['correction_new_suffix_steps'],0)
                    self.assertEqual(row['costs']['old_reference_suffix_steps'],0)
                else:
                    self.assertEqual(row['costs']['snapshot_restore_count'],0)
            # Fixed-policy evaluation uses the same legal scenario only as a
            # bounded integration fixture; it is never performance evidence.
            manifest=json.loads((ROOT/'tests/fixtures/hgr/handoff_manifest.json').read_text(encoding='utf-8'))
            manifest['scenarios'][0]['scenario_split']='validation'
            manifest_path=base/'evaluation_manifest.json'; manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
            run_entry(['evaluate','--checkpoint',summary['latest_checkpoint'],'--output-dir',base/'evaluate',
                       '--episodes','1','--manifest',manifest_path])
            evaluation=json.loads((base/'evaluate/summary.json').read_text(encoding='utf-8'))
            self.assertFalse(evaluation['training_update'])
            self.assertEqual(evaluation['reward_objective'],'team_mean_v1')
            self.assertEqual(evaluation['policy_mode'],'stochastic')
            if os.environ.get('AUV_HGR_EVIDENCE_DIR'):
                evidence=Path(os.environ['AUV_HGR_EVIDENCE_DIR']); evidence.mkdir(parents=True,exist_ok=True)
                for name in ('cycles.json','branches.json','episodes.json','summary.json','config.json'):
                    shutil.copyfile(output/name,evidence/name)
                for method in ('stochastic_direct_mc','direct_boundary_corrected','resumed','evaluate'):
                    destination=evidence/method;destination.mkdir(exist_ok=True)
                    for name in ('summary.json','cycles.json'):
                        if (base/method/name).is_file():
                            shutil.copyfile(base/method/name,destination/name)
                if os.environ.get('AUV_HGR_CHECKPOINT_DIR'):
                    checkpoint_dir=Path(os.environ['AUV_HGR_CHECKPOINT_DIR']);checkpoint_dir.mkdir(parents=True,exist_ok=True)
                    for path in (output/'checkpoints').glob('*.pt'):
                        target=checkpoint_dir/path.name
                        if target.exists():
                            raise FileExistsError(target)
                        shutil.copyfile(path,target)
                    (evidence/'retained_checkpoint.json').write_text(json.dumps(dict(
                        path=str(checkpoint_dir/Path(summary['latest_checkpoint']).name),
                        theta_hash=final['theta_hash'],phi_hash=final['phi_hash'],
                        source_sha256=final['source_identity']['sha256'],
                        purpose='bounded integration only; not performance evidence',
                        resume_parameters_identical=True)),encoding='utf-8')

    def test_no_handoff_cycle_retains_old_gradient_and_budget_completes_cycle(self):
        with tempfile.TemporaryDirectory(prefix='hgr-no-handoff-') as temporary:
            directory=Path(temporary)
            manifest=json.loads((ROOT/'tests/fixtures/hgr/handoff_manifest.json').read_text(encoding='utf-8'))
            for field in ('target_position','target_initial_position'):
                manifest['scenarios'][0][field]=[18.,18.,6.]
            scenario_path=directory/'manifest.json'; scenario_path.write_text(json.dumps(manifest),encoding='utf-8')
            import hashlib
            config=load_config(CONFIG)
            config.update(scenario_manifest=str(scenario_path),scenario_manifest_sha256=hashlib.sha256(scenario_path.read_bytes()).hexdigest(),max_steps=2,total_main_trajectories=5,max_total_environment_steps=1)
            config_path=directory/'config.json'; config_path.write_text(json.dumps(config),encoding='utf-8')
            output=directory/'collision_terminal/no_handoff'
            run_entry(['train','--config',config_path,'--output-dir',output])
            summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
            row=json.loads((output/'cycles.json').read_text(encoding='utf-8'))[0]
            self.assertEqual(row['N'],1);self.assertEqual(row['no_handoff_count'],1)
            self.assertFalse(row['hgr_boundary_mechanism_active'])
            self.assertEqual(row['delta_g_norm'],0.)
            self.assertGreater(row['g0_norm'],0.)
            self.assertEqual(summary['completed_cycles'],1)
            self.assertEqual(summary['costs']['snapshot_restore_count'],0)
            self.assertGreater(summary['budget_overshoot_steps'],0)
            self.assertTrue(row['complete'])

    def test_generated_m20_scenario_reaches_real_production_collector(self):
        config=load_config(DEFAULT_CONFIG)
        config['max_steps']=1
        config['policy']['actor'].update(hidden_dim=8,expert_hidden_dim=8)
        with tempfile.TemporaryDirectory(prefix='hgr-generated-m20-') as temporary:
            trainer=Trainer(config,Path(temporary)/'collision_terminal/generated')
            _,seed,scenario=trainer.scenario('bounded_generated_m20')
            self.assertEqual(scenario['scenario_profile'],'M20_MOVING_UNKNOWN_MULTI')
            self.assertEqual(scenario['obstacle_knowledge_mode'],'online_unknown')
            self.assertGreaterEqual(len(scenario['obstacles']),2)
            trajectory=collect_trajectory(config,scenario,trainer.policy,seed=seed)
            self.assertEqual(len(trajectory['records']),1)
            self.assertTrue(all(trajectory['records'][-1]['dones']))
            self.assertEqual(trajectory['summary']['reward_objective'],'team_mean_v1')


if __name__ == '__main__':
    unittest.main()
