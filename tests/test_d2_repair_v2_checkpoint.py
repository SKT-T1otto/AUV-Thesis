import copy
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from chapter3_bser.experiments.baselines.common.checkpoint import digest,state_digest
from chapter3_bser.experiments.d2_repair_v2.train import Trainer,load_for_evaluation
from chapter3_bser.experiments.d2_repair_v2.evaluate import evaluate
from tests.d2_v2_test_utils import ROOT,config,evidence_dir
from tests.d2_v2_scenarios import scenario

class ActualSchedulerCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old=torch.get_num_threads(); cls.addClassCleanup(torch.set_num_threads,old); torch.set_num_threads(1)

    def test_real_default_scheduler_checkpoint_identity_and_heldout_eval(self):
        results=[]
        for label in ('b2','b3'):
            with tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary); train_scene=scenario('normal',32)
                manifest=dict(scenario_profile=train_scene['scenario_profile'],scenario_split='train',scenario_role='train',scenario_count=1,scenarios=[train_scene])
                path=root/'train.json'; path.write_text(json.dumps(manifest),encoding='utf-8')
                c=config(label); c.update(max_steps=32,total_main_trajectories=8,max_total_environment_steps=256,
                    scenario_manifest=str(path),scenario_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                trainer=Trainer(c,root/'collision_terminal'/'train',mechanism=True)
                summary=trainer.run()
                self.assertEqual(summary['environment_steps'],256)
                self.assertEqual(summary['optimizer_updates'],2)
                self.assertEqual(summary['actor_optimizer_steps'],[2]*4)
                model,loaded,payload=load_for_evaluation(summary['latest_checkpoint'],allow_mechanism=True)
                self.assertEqual(payload['global_step'],256); self.assertTrue(payload['mechanism_only'])
                self.assertFalse(payload['formal_use_allowed'])
                self.assertEqual(state_digest(model.training_state_dict()),payload['model_state_sha256'])
                with self.assertRaises(ValueError): load_for_evaluation(summary['latest_checkpoint'])
                before=state_digest(trainer.model.training_state_dict())
                with patch('chapter3_bser.experiments.d2_repair_v2.provenance.verify_reviewed_changes',side_effect=ValueError('source drift')):
                    with self.assertRaises(ValueError): trainer.save('blocked')
                self.assertFalse((trainer.output/'checkpoint_blocked.pt').exists())
                self.assertEqual(before,state_digest(trainer.model.training_state_dict()))
                held=scenario('normal',32,seed=41729)
                held.update(scenario_split='eval',scenario_role='eval',target_position=[17.,16.,2.],target_initial_position=[17.,16.,2.])
                evaluation=root/'eval.json'; evaluation.write_text(json.dumps(dict(scenarios=[held])),encoding='utf-8')
                evaluate(summary['latest_checkpoint'],evaluation,root/'collision_terminal'/'eval',mechanism=True)
                rows=json.loads((root/'collision_terminal/eval/episodes.json').read_text(encoding='utf-8'))
                self.assertEqual(len(rows),1); self.assertTrue(rows[0]['trajectory_complete'])
                bad=copy.deepcopy(payload); bad['schema']='ch3.d2.implementation_repair.v1'
                bad_path=root/'legacy.pt'; torch.save(bad,bad_path)
                with self.assertRaises(ValueError): load_for_evaluation(bad_path,allow_mechanism=True)
                results.append(dict(algorithm=label,environment_steps=summary['environment_steps'],episodes=summary['episodes'],
                    default_hyperparameters_unchanged=True,update_rounds=summary['optimizer_updates'],actor_steps=summary['actor_optimizer_steps'],
                    critic_steps=summary['critic_optimizer_steps'],checkpoint_source_checked=True,
                    default_formal_load_rejected=True,heldout_evaluation_complete=True,source_drift_save_rejected=True))
                del trainer,model; gc.collect()
        out=evidence_dir()/'scheduler_checkpoint.json'
        out.write_bytes((json.dumps(results,indent=2)+'\n').encode())
