import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from tools.verify_d2_repair_v2_source import verify as independent_verify
from chapter3_bser.experiments.d2_repair_v2.provenance import verify_reviewed_changes,training_admission
from chapter3_bser.experiments.d2_repair_v2.train import Trainer,load_for_evaluation
from tests.d2_v2_test_utils import ROOT,config

class Provenance(unittest.TestCase):
    def test_complete_source_and_independent_review(self):
        self.assertTrue(independent_verify()["source_passed"])
        self.assertEqual(verify_reviewed_changes()["parent_commit"],"6538c3e6a67e96d56940ab09fcff15cd9dc7a7d8")
        data=json.loads((ROOT/'docs/provenance/d2_repair_v2_evolution.json').read_text(encoding='utf-8'))
        self.assertEqual(len(data['unchanged_files']),302)
        self.assertEqual(len(json.loads((ROOT/'docs/provenance/ch3_to_core_migration_manifest.json').read_text(encoding='utf-8'))['records']),27)

    def test_any_unregistered_production_drift_is_rejected(self):
        data=json.loads((ROOT/'docs/provenance/d2_repair_v2_evolution.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name in list(data['profiles']['windows_existing']['files'])+['docs/provenance/d2_repair_v2_evolution.json']:
                target=root/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/name,target)
            verify_reviewed_changes(root)
            for name in ('core/algorithms/agents.py','chapter3_bser/experiments/d2_repair_v2/model.py','configs/chapter3/d2_repair_v2/b2_train.json'):
                path=root/name; saved=path.read_bytes(); path.write_bytes(saved+b'\n# unreviewed\n')
                with self.assertRaises(ValueError): verify_reviewed_changes(root)
                path.write_bytes(saved)
            (root/'core/unreviewed.py').write_text('pass\n')
            with self.assertRaises(ValueError): verify_reviewed_changes(root)

    def test_admission_rejects_formal_run_without_all_required_evidence(self):
        with self.assertRaises(ValueError): training_admission(config=config())
        with tempfile.TemporaryDirectory() as temporary:
            target=Path(temporary)/'collision_terminal'/'blocked'
            with self.assertRaises(ValueError): Trainer(config(),target)
            self.assertFalse(target.exists())
        c=config(); c.update(max_steps=32,total_main_trajectories=1,max_total_environment_steps=32)
        self.assertIn('review_sha256',training_admission(mechanism=True,config=c))
        with patch('chapter3_bser.experiments.d2_repair_v2.train.verify_reviewed_changes',side_effect=ValueError('source drift')):
            with self.assertRaisesRegex(ValueError,'source drift'): load_for_evaluation('not-opened.pt')

    def test_check_only_creates_no_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'collision_terminal'/'plan'
            p=subprocess.run([__import__('sys').executable,'-m','chapter3_bser.experiments.d2_repair_v2.train',
                '--config','configs/chapter3/d2_repair_v2/b2_train.json','--output',str(output)],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            result=json.loads(p.stdout); self.assertFalse(result['executed']); self.assertFalse(result['admitted'])
            self.assertFalse(output.exists())
