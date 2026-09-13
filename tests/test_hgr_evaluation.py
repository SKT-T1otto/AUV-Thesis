"""Entry hardening: unit faults plus real fixed-version production behavior.

Repeated scenes and reduced networks below are explicitly interface fixtures,
never held-out performance data. No retained user checkpoint is changed.
"""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from chapter3_bser.experiments.hgr import cli, evaluation as ev
from chapter3_bser.experiments.hgr.provenance import fresh_source_identity, validate_source_identity, require_source_match
from chapter3_bser.experiments.hgr.train import Trainer, load_config, load_checkpoint, validate_config, digest, validated_output, write_json
from chapter3_bser.experiments.phase1c_prrac.task_metrics import aggregate_task_outcomes
from chapter3_bser.models.hgr.policy import weights_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'tests/fixtures/hgr/integration_config.json'
REFERENCE = '0fed8e1650f2747bb58f17bc0b375b69870ea0b5'
NEW_ROW_FIELDS = {'evaluation_episode_index', 'policy_sampling_seed', 'contact_episode',
                  'hold_episode', 'episode_length', 'wall_seconds', 'optimizer_update_count', 'failure_stage'}


def fixture_manifest(count=1):
    original = json.loads((ROOT/'tests/fixtures/hgr/handoff_manifest.json').read_text(encoding='utf-8'))
    scenes = []
    for i in range(count):
        scene = copy.deepcopy(original['scenarios'][0])
        scene.update(scenario_id=f'entry_fixture_{i}', scenario_split='validation', scenario_role='validation')
        scenes.append(scene)
    return dict(schema='test.hgr.entry_fixture.v1',
                purpose='Repeated training scene for interface verification only; no performance evidence', scenarios=scenes)


def trajectory(config, scenario, reason='timeout', found=False):
    """Synthetic ONLY for metrics/fault injection tests; real test is separate."""
    row = {k: config[k] for k in ('task_protocol', 'collision_detection_revision', 'terminal_reward_revision',
                                 'reward_objective', 'source_reward_revision')}
    row.update(scenario_id=scenario['scenario_id'], scenario_seed=scenario['scenario_seed'],
               gamma=config['rl']['gamma'], actual_length=2, success=reason == 'success',
               safe_success=reason == 'success', found=found, terminated=True, truncated=False,
               termination_reason=reason, collision_episode=reason == 'obstacle_collision',
               first_collision_step=2 if reason == 'obstacle_collision' else None,
               first_collision_agent_ids=[0, 1, 3] if reason == 'obstacle_collision' else [],
               first_collision_phase='after_found' if found and reason == 'obstacle_collision' else None,
               found_step=1 if found else None, handoff_event_step=None, handoff_decision_step=None,
               team_discounted_return=1.95, team_undiscounted_return=2., source_individual_returns=[2.]*4,
               capture_contact_step_count=1 if found else 0, capture_full_hold_step_count=0)
    return dict(summary=row, records=[{'team_reward':1., 'final_reward_by_agent':[1.]*4} for _ in range(2)])


def adapt(config, scenario, **kwargs):
    return ev.episode_row(trajectory(config, scenario, **kwargs), config, scenario, index=0,
                         sampling_seed=3, policy_mode='stochastic', checkpoint=Path('fixture.pt'),
                         policy_hash='fixture', wall_seconds=.1)


class HGREvaluationUnitTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)
        self.scenario = fixture_manifest()['scenarios'][0]

    def test_shared_task_metrics_and_real_clock_adapter(self):
        rows = [adapt(self.config, self.scenario, reason='success', found=True),
                adapt(self.config, self.scenario, reason='obstacle_collision', found=True),
                adapt(self.config, self.scenario)]
        expected = aggregate_task_outcomes(rows, expected_episodes=3)
        resolved = ev.resolved_config(self.config, 3, 17, 'stochastic')
        summary = ev.evaluation_summary(rows, resolved, finalized=True, wall_seconds=1.)
        for key, value in expected.items():
            self.assertEqual(summary[key], value, key)
        self.assertEqual(summary['n_success']+summary['n_collision_failure']+summary['n_timeout'], 3)
        self.assertEqual(summary['actual_environment_steps'], 6)
        self.assertEqual(summary['first_collision_by_role'], {'Searcher':1, 'Executor':1})
        self.assertEqual(summary['first_collision_by_agent'], {'0':1,'1':1,'3':1})
        self.assertEqual(summary['success_if_found_denominator'], 2)
        self.assertEqual(summary['mean_team_undiscounted_return'], 2.)
        self.assertEqual(summary['actual_training_updates'], 0)
        zero = ev.evaluation_summary([rows[-1]], {**resolved, 'episodes':1}, finalized=True, wall_seconds=0.)
        self.assertIsNone(zero['success_if_found_rate'])
        incomplete = ev.evaluation_summary(rows, resolved, finalized=False, wall_seconds=1.)
        self.assertFalse(incomplete['evaluation_complete'])
        self.assertIsNone(incomplete['safe_success_rate'])
        self.assertIsNone(incomplete['mean_team_discounted_return'])

    def test_resolved_config_preserves_existing_optional_environment_defaults(self):
        omitted=copy.deepcopy(self.config)
        for field in ('collision_terminal_reward','collision_detection_revision','terminal_reward_revision',
                      'base_candidate','source_reward_revision'):
            omitted.pop(field)
        validate_config(omitted)
        self.assertEqual(ev.resolved_config(omitted,2,17,'stochastic'),
                         ev.resolved_config(self.config,2,17,'stochastic'))
        resolved=ev.resolved_config(self.config,2,17,'stochastic')
        unused={'lr_actor','lr_critic','gamma','batch_size','replay_size','training_episodes',
                'pilot_episodes','sigma_hold_episodes'}
        self.assertFalse(unused.intersection(resolved['environment_config']))
        self.assertNotIn('experiment',resolved['phase1b_config'])
        self.assertEqual(resolved['gamma'],self.config['rl']['gamma'])
        self.assertEqual(resolved['environment_config']['max_steps'],self.config['max_steps'])
        self.assertEqual(len(resolved['phase1b_reference_config_sha256']),64)

    def test_boolean_missing_contradictory_unknown_and_clock(self):
        value = trajectory(self.config, self.scenario)
        args = dict(index=0, sampling_seed=3, policy_mode='stochastic', checkpoint='fixture', policy_hash='x', wall_seconds=0.)
        for key in ('success','safe_success','found','collision_episode','truncated'):
            value['summary'][key]='false'
        value['summary']['terminated']='true'
        self.assertIs(ev.episode_row(value, self.config, self.scenario, **args)['found'], False)
        for updates in ({'found':None}, {'success':True}, {'termination_reason':'aborted'},
                        {'actual_length':3}, {'gamma':.5}, {'scenario_seed':123},
                        {'source_individual_returns':[float('nan')]*4}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                changed=copy.deepcopy(value); changed['summary'].update(updates)
                ev.episode_row(changed, self.config, self.scenario, **args)
        value['summary'].pop('capture_contact_step_count')
        self.assertIsNone(ev.episode_row(value, self.config, self.scenario, **args)['contact_episode'])

    def test_source_inventory_aggregate_and_difference(self):
        identity = fresh_source_identity()
        validate_source_identity(identity)
        for update in ({'files':{}}, {'sha256':'0'*64}, {'files':{'../core/x.py':'0'*64}}, {'files':{'.':'0'*64}},
                       {'implementation_version':'unknown'}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_source_identity({**identity, **update})
        changed = copy.deepcopy(identity)
        changed['files']['core/env/uav_env.py']='0'*64
        changed['sha256']=hashlib.sha256(json.dumps(changed['files'], sort_keys=True).encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, 'core/env/uav_env.py'):
            require_source_match(changed, identity)

    def test_manifest_identity_validation_canonical_order_mode_and_pairing(self):
        source=fresh_source_identity()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'scenes.json'; original=fixture_manifest(2)
            write_json(path, original)
            selected=ev.evaluation_manifest(self.config,2,17,path,source)
            self.assertEqual(selected['scenarios'],original['scenarios'])
            self.assertEqual(selected['origin']['original_schema'],original['schema'])
            resolved=ev.resolved_config(self.config,2,17,'stochastic')
            identity=ev.evaluation_identity(resolved,selected,'checkpoint','policy',source)
            path.write_text(json.dumps(original, separators=(',',':')), encoding='utf-8')
            compact=ev.evaluation_manifest(self.config,2,17,path,source)
            self.assertNotEqual(selected['origin']['original_file_sha256'],compact['origin']['original_file_sha256'])
            self.assertEqual(identity['sha256'],ev.evaluation_identity(resolved,compact,'checkpoint','policy',source)['sha256'])
            for changes in ({'output_dir':'different', 'total_main_trajectories':9876, 'prefix_lr':.8},):
                self.assertEqual(resolved, ev.resolved_config({**self.config,**changes},2,17,'stochastic'))
            alternative={**resolved,'method':'ch3_stochastic_direct_mc','algorithm':'stochastic_direct_mc'}
            paired=ev.evaluation_identity(alternative,selected,'other-checkpoint','other-policy',source)
            self.assertEqual(identity['pairing_sha256'],paired['pairing_sha256'])
            self.assertNotEqual(identity['sha256'],paired['sha256'])
            for changes in ({'policy_mode':'deterministic_mean'}, {'gamma':.8}):
                other=ev.evaluation_identity({**resolved,**changes},selected,'checkpoint','policy',source)
                self.assertNotEqual(identity['sha256'],other['sha256'])
                self.assertNotEqual(identity['pairing_sha256'],other['pairing_sha256'])
            for scenes in (list(reversed(original['scenarios'])), [{**original['scenarios'][0], 'target_velocity':[0.,0.,0.]},original['scenarios'][1]]):
                write_json(path,{**original,'scenarios':scenes})
                altered=ev.evaluation_manifest(self.config,2,17,path,source)
                self.assertNotEqual(identity['sha256'],ev.evaluation_identity(resolved,altered,'checkpoint','policy',source)['sha256'])
            bad_values=[[], [self.scenario,self.scenario]]
            for change in ({'scenario_id':None},{'scenario_seed':None},{'scenario_split':'train'},
                           {'scenario_profile':'unsupported'},{'max_steps':1}):
                bad_values.append([{**self.scenario,**change},original['scenarios'][1]])
            for scenes in bad_values:
                write_json(path,{**original,'scenarios':scenes})
                with self.subTest(scenes=scenes), self.assertRaises(ValueError):
                    ev.evaluation_manifest(self.config,2,17,path,source)
            generated=ev.evaluation_manifest(self.config,2,17,None,source)
            self.assertEqual(generated['selected_count'],2)
            self.assertTrue(all(s['obstacles'] for s in generated['scenarios']))
            self.assertEqual(generated['origin']['generator_seed'],17)
            self.assertEqual(generated['selected_content_sha256'],digest(generated['scenarios']))

    def test_output_preflight_all_methods_apis_cli_and_initialization_routes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            legal=root/'collision_terminal/empty'; legal.mkdir(parents=True)
            self.assertEqual(validated_output(self.config,legal),legal.resolve())
            self.assertEqual(validated_output(self.config,root/'collision_terminal/new'),(root/'collision_terminal/new').resolve())
            self.assertFalse((root/'collision_terminal/new').exists())
            occupied=root/'collision_terminal/occupied'; occupied.mkdir(); (occupied/'keep.txt').write_text('keep')
            paths=[root/'missing', root/'not_collision_terminal/path', root/'collision_terminal/../escaped', occupied]
            with patch('chapter3_bser.experiments.hgr.train.HandoffPolicy') as policy, patch.object(ev,'collect_trajectory') as collect:
                for algorithm in ('hgr','stochastic_direct_mc','direct_boundary_corrected'):
                    config={**self.config,'method':'ch3_'+algorithm,'algorithm':algorithm}
                    config_path=root/'config.json'; write_json(config_path,config)
                    for path in paths:
                        for kwargs in ({},{'resume':'unused.pt'},{'mean_initialization':'unused.pt'}):
                            with self.subTest(algorithm=algorithm,path=path,kwargs=kwargs), self.assertRaises((ValueError,FileExistsError)):
                                Trainer(config,path,**kwargs)
                        for command in ('train','resume','mean'):
                            args=['resume' if command=='resume' else 'train','--config',str(config_path),'--output-dir',str(path)]
                            if command!='train': args+=['--checkpoint' if command=='resume' else '--mean-initialization','unused.pt']
                            with self.assertRaises((ValueError,FileExistsError)):
                                cli.main(args)
                        with patch.object(ev,'file_sha256',return_value='fixture'), patch.object(ev,'load_checkpoint',return_value={'config':config}):
                            with self.assertRaises((ValueError,FileExistsError)):
                                ev.evaluate('unused.pt',path)
                            with self.assertRaises((ValueError,FileExistsError)):
                                cli.main(['evaluate','--checkpoint','unused.pt','--output-dir',str(path)])
                policy.assert_not_called(); collect.assert_not_called()
            self.assertEqual((occupied/'keep.txt').read_text(),'keep')
            self.assertFalse((root/'escaped').exists()); self.assertFalse((root/'missing').exists())


PROBE = r'''
import json, sys
from pathlib import Path
import numpy as np
import torch
from chapter3_bser.experiments.hgr import cli
from chapter3_bser.experiments.hgr.runtime import MissionRuntime
torch.set_num_threads(1)
def normalize(value):
    if isinstance(value, np.ndarray):
        return {'dtype':str(value.dtype),'shape':list(value.shape),'bytes':value.tobytes().hex()}
    if isinstance(value, (np.floating, float)): return {'float_hex':float(value).hex()}
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, dict): return {str(k):normalize(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [normalize(v) for v in value]
    return value
steps=[]
original=MissionRuntime.advance
def observed(self, policy, **kwargs):
    result=original(self,policy,**kwargs)
    steps.append(normalize(dict(record=result,task_result=self.env.get_episode_result(),
        found_step=self.found_step,handoff_event_step=self.handoff_event_step,handoff_decision_step=self.handoff_decision_step)))
    return result
MissionRuntime.advance=observed
checkpoint, manifest, output, mode, trace=sys.argv[1:]
code=cli.main(['evaluate','--checkpoint',checkpoint,'--manifest',manifest,'--output-dir',output,
               '--episodes','1','--seed','839','--policy-mode',mode])
Path(trace).write_text(json.dumps(steps,sort_keys=True),encoding='utf-8')
raise SystemExit(code)
'''


class HGREvaluationProductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary=tempfile.TemporaryDirectory(prefix='hgr-entry-production-')
        cls.root=Path(cls.temporary.name)
        cls.config=load_config(CONFIG)
        cls.config.update(total_main_trajectories=1)
        trainer=Trainer(cls.config,cls.root/'collision_terminal/train')
        result=trainer.run()  # real complete bounded cycle, never invented checkpoint counters
        cls.checkpoint=Path(result['latest_checkpoint'])
        cls.manifest=cls.root/'validation_fixture.json'; write_json(cls.manifest,fixture_manifest(2))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def output(self,name):
        return self.root/'collision_terminal'/name

    def test_checkpoint_integrity_rng_and_source_rejection(self):
        before=ev.file_sha256(self.checkpoint)
        state=torch.get_rng_state().clone()
        payload=load_checkpoint(self.checkpoint)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        for field,change in [('source_identity',{'files':{}}), ('source_identity',{'sha256':'0'*64}),
                             ('config',{'algorithm':'invalid'}), ('config',{'rl':{'gamma':.7}})]:
            bad=copy.deepcopy(payload); bad[field].update(change)
            path=self.root/'bad.pt'; torch.save(bad,path)
            with self.subTest(field=field,change=change),self.assertRaises(ValueError):
                ev.evaluate(path,self.output('rejected'),episodes=1,manifest=self.manifest)
            self.assertFalse(self.output('rejected').exists())
        bad=copy.deepcopy(payload)
        name=next(iter(bad['source_identity']['files']))
        bad['source_identity']['files'][name]='0'*64
        bad['source_identity']['sha256']=hashlib.sha256(json.dumps(bad['source_identity']['files'],sort_keys=True).encode()).hexdigest()
        torch.save(bad,self.root/'bad.pt')
        with self.assertRaisesRegex(ValueError,'production source mismatch'):
            ev.evaluate(self.root/'bad.pt',self.output('rejected'),episodes=1,manifest=self.manifest)
        self.assertEqual(before,ev.file_sha256(self.checkpoint))

        for update in ({'schema':'unknown'}, {'architecture_version':'unknown'},
                       {'method':'ch3_stochastic_direct_mc'}, {'algorithm':'stochastic_direct_mc'}, {'gamma':.8}):
            bad=copy.deepcopy(payload); bad.update(update); torch.save(bad,self.root/'bad.pt')
            with self.subTest(update=update), self.assertRaises(ValueError): load_checkpoint(self.root/'bad.pt')
        for value in (float('nan'),123.):
            bad=copy.deepcopy(payload)
            tensor=next(iter(bad['policy'].values())); tensor.fill_(value)
            torch.save(bad,self.root/'bad.pt')
            with self.subTest(value=value),self.assertRaises(ValueError): load_checkpoint(self.root/'bad.pt')

    def test_generated_evaluation_persists_initial_plan_actual_config_and_random_streams(self):
        output=self.output('generated_plan')
        seen=[]
        def collector(config,scenario,policy,**kwargs):
            progress=json.loads((output/'evaluation_progress.json').read_text(encoding='utf-8'))
            summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertFalse(progress['evaluation_complete']); self.assertFalse(summary['evaluation_complete'])
            self.assertIsNone(summary['safe_success_rate'])
            self.assertEqual(progress['n_valid_episodes'],len(seen))
            self.assertFalse(policy.training)
            self.assertTrue(all(not p.requires_grad for p in policy.parameters()))
            seen.append((copy.deepcopy(scenario),kwargs['seed']))
            return trajectory(config,scenario)
        rng=torch.get_rng_state().clone()
        with patch.object(ev,'collect_trajectory',collector):
            result=ev.evaluate(self.checkpoint,output,episodes=2,seed=413,policy_mode='deterministic_mean')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(result['evaluation_complete']); self.assertEqual(result['actual_environment_steps'],4)
        manifest=json.loads((output/'evaluation_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['scenarios'],[s for s,_ in seen]); self.assertEqual([s for _,s in seen],[413,414])
        self.assertEqual(manifest['origin']['generator_seed'],413)
        resolved=json.loads((output/'resolved_evaluation_config.json').read_text(encoding='utf-8'))
        self.assertEqual(resolved['episodes'],2); self.assertEqual(resolved['seed'],413)
        self.assertEqual(resolved['policy_mode'],'deterministic_mean')
        self.assertEqual(resolved['max_steps'],8); self.assertEqual(resolved['gamma'],.95)
        self.assertNotIn('output_dir',resolved); self.assertNotIn('total_main_trajectories',resolved)
        self.assertIn('execution_runtime',resolved)
        self.assertTrue((output/'evaluation_identity.json').is_file())

    def test_failures_interruptions_preserve_prefix_and_nonzero_cli(self):
        for kind in ('RuntimeError','KeyboardInterrupt'):
            output=self.output('failure_'+kind)
            # Run the actual CLI in a child process; only the collector fails.
            script="""import sys
from unittest.mock import patch
from chapter3_bser.experiments.hgr import cli, evaluation as ev
from tests.test_hgr_evaluation import trajectory
count=0
def collector(config,scenario,policy,**kwargs):
    global count
    count+=1
    if count==2: raise ERROR('injected interface failure')
    return trajectory(config,scenario)
with patch.object(ev,'collect_trajectory',collector):
    raise SystemExit(cli.main(sys.argv[1:]))
""".replace('ERROR',kind)
            result=subprocess.run([sys.executable,'-B','-c',script,'evaluate','--checkpoint',str(self.checkpoint),
                    '--manifest',str(self.manifest),'--output-dir',str(output),'--episodes','2'],cwd=ROOT,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
            failure=json.loads((output/'evaluation_failure.json').read_text(encoding='utf-8'))
            rows=json.loads((output/'episodes.json').read_text(encoding='utf-8'))
            self.assertEqual(len(rows),1); self.assertEqual(summary['n_valid_episodes'],1)
            self.assertFalse(summary['evaluation_complete']); self.assertEqual(summary['n_expected_episodes'],2)
            self.assertEqual(summary['n_timeout'],1); self.assertEqual(summary['n_collision_failure'],0)
            self.assertIsNone(summary['safe_success_rate']); self.assertIsNone(summary['mean_team_discounted_return'])
            self.assertEqual(failure['evaluation_episode_index'],1); self.assertEqual(failure['exception_type'],kind)
            self.assertIn('injected interface failure',failure['traceback'])
            self.assertEqual(failure['scenario']['scenario_id'],'entry_fixture_1')
            with self.assertRaises(FileExistsError):
                ev.evaluate(self.checkpoint,output,episodes=2,manifest=self.manifest)

    def test_fresh_end_checks_detect_policy_source_checkpoint_and_manifest_changes(self):
        payload=load_checkpoint(self.checkpoint)
        for kind in ('policy','source','checkpoint','manifest','saved_identity'):
            output=self.output('mutation_'+kind)
            cp=self.root/('copy_'+kind+'.pt'); shutil.copyfile(self.checkpoint,cp)
            manifest=self.root/('manifest_'+kind+'.json'); shutil.copyfile(self.manifest,manifest)
            source=copy.deepcopy(payload['source_identity'])
            def collector(config,scenario,policy,**kwargs):
                if kind=='policy':
                    with torch.no_grad(): next(policy.parameters()).add_(1.)
                elif kind=='source': source['sha256']='0'*64
                elif kind=='checkpoint':
                    with cp.open('ab') as stream: stream.write(b'changed')
                elif kind=='manifest': manifest.write_text('{}',encoding='utf-8')
                else: (output/'evaluation_identity.json').write_text('{}',encoding='utf-8')
                return trajectory(config,scenario)
            with patch.object(ev,'collect_trajectory',collector), patch.object(ev,'fresh_source_identity',side_effect=lambda:copy.deepcopy(source)):
                with self.subTest(kind=kind),self.assertRaises((ValueError,RuntimeError)):
                    ev.evaluate(cp,output,episodes=1,manifest=manifest)
            summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertFalse(summary['evaluation_complete'])
            self.assertIsNone(summary['mean_team_undiscounted_return'])
            self.assertTrue((output/'evaluation_failure.json').exists())

    def test_real_production_evaluation_matches_fixed_reference_actions_rewards_events(self):
        reference=self.root/'reference'; reference.mkdir()
        archive=subprocess.run(['git','-c','safe.directory='+ROOT.as_posix(),'archive',REFERENCE],cwd=ROOT,capture_output=True,check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            for member in source.getmembers():
                self.assertTrue((reference/member.name).resolve().is_relative_to(reference.resolve()))
                self.assertFalse(member.issym() or member.islnk())
            source.extractall(reference)
        current=self.root/'current'; current.mkdir()
        for name in ('core','chapter3_bser','configs'):
            shutil.copytree(ROOT/name,current/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for directory in (current,reference):
            (directory/'_hgr_evaluation_probe.py').write_text(PROBE,encoding='utf-8')
        before=ev.file_sha256(self.checkpoint)
        evidence=[]
        for mode in ('stochastic','deterministic_mean'):
            outputs={}
            for label,directory in (('reference',reference),('current',current)):
                output=self.output('behavior_'+label+'_'+mode); trace=self.root/(label+'_'+mode+'.json')
                result=subprocess.run([sys.executable,'-E','-s','-B','-m','_hgr_evaluation_probe',str(self.checkpoint),str(self.manifest),str(output),mode,str(trace)],
                    cwd=directory,capture_output=True,text=True,timeout=900)
                self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
                outputs[label]=(json.loads(trace.read_text(encoding='utf-8')),
                    json.loads((output/'episodes.json').read_text(encoding='utf-8'))[0],
                    json.loads((output/'summary.json').read_text(encoding='utf-8')))
            old,new=outputs['reference'],outputs['current']
            self.assertEqual(old[0],new[0])  # bytes for every array; hex for every float
            self.assertEqual(len(new[0]),8)
            self.assertEqual(set(new[1])-set(old[1]),NEW_ROW_FIELDS)
            self.assertEqual({k:new[1][k] for k in old[1]},old[1])
            for key in old[2]: self.assertEqual(old[2][key],new[2][key],key)
            self.assertEqual(new[1]['found_step'],1); self.assertEqual(new[1]['handoff_decision_step'],2)
            self.assertTrue(new[2]['evaluation_complete']); self.assertEqual(new[2]['actual_environment_steps'],8)
            self.assertEqual(new[2]['actual_training_updates'],0)
            output=self.output('behavior_current_'+mode)
            identity=json.loads((output/'evaluation_identity.json').read_text(encoding='utf-8'))
            self.assertEqual(identity['canonical']['random_streams']['episodes'][0]['policy_sampling_seed'],839)
            self.assertGreater(sum(identity['checkpoint_training']['costs'].values()),0)
            evidence.append(dict(policy_mode=mode, reference_commit=REFERENCE, exact_steps=8,
                                 trace_sha256=digest(new[0]), policy_hash=new[1]['policy_hash'],
                                 new_row_fields=sorted(NEW_ROW_FIELDS), summary=new[2]))
            if os.environ.get('HGR_ENTRY_EVIDENCE_DIR'):
                target=Path(os.environ['HGR_ENTRY_EVIDENCE_DIR'])/mode
                self.assertFalse(target.exists()); shutil.copytree(output,target)
        self.assertEqual(before,ev.file_sha256(self.checkpoint))
        self.assertNotEqual(evidence[0]['trace_sha256'],evidence[1]['trace_sha256'])
        if os.environ.get('HGR_ENTRY_EVIDENCE_DIR'):
            write_json(Path(os.environ['HGR_ENTRY_EVIDENCE_DIR'])/'behavior_comparison.json',
                       dict(reference_commit=REFERENCE,checkpoint_file_sha256=before,comparisons=evidence,
                            test_fixture_only=True,formal_experiments_completed=False,performance_claims_supported=False))


if __name__ == '__main__':
    unittest.main()
