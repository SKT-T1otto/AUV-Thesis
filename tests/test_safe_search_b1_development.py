"""B1 method isolation and strict 60-arm accounting; fake processes, no simulator."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chapter3_bser.experiments.safe_search_v1 import run_b1_development as b
from chapter3_bser.experiments.safe_search_v1.run_paired import write_json, read_json, file_hash, summarize
from tests.test_safe_search_development import fixtures


def make_plan(refs):
    return dict(schema="ch3.safe_search.b1_plan.v1", baseline=b.BASELINE, variants=list(b.VARIANTS),
        options=copy.deepcopy(b.OPTIONS), development_indices=list(b.INDICES), seed=b.SEED, max_steps=400,
        planned_episode_runs=60, training=False, checkpoint_loaded=False, heldout_scenarios_executed=0,
        primary_comparison=["V5", "V3"], secondary_comparisons=[["V5", "V0"], ["V3", "V0"]],
        bootstrap_replicates=10000, bootstrap_seed=20260928,
        development_gate=dict(comparator="V3", found_count_not_lower=True, pre_found_collision_count_not_higher=True,
            stall_plus_hold_fraction_strictly_lower=True, effective_observation_fraction_not_lower=True),
        reference_hashes={k:{f:r[f] for f in ("identity_sha256", "episodes_sha256", "source_inventory_sha256")}
                          for k,r in refs.items()})


def write_child(directory, scene, variant, parent):
    directory.mkdir(parents=True)
    identity = dict(schema="ch3.safe_search.paired_identity.v1", baseline=b.BASELINE, variants=[variant],
        seed=b.SEED, steps=400, task_horizon=400, full_episodes=True, training=False, checkpoint_loaded=False,
        formal_thesis_evaluation=False, residual_source="zeros_4x3", selected=[scene], search_coverage_enabled=True,
        input_sha256=parent["child_input_sha256"], sources_before=parent["sources_before"],
        sources_after=parent["sources_before"], source_and_input_verification_passed=True)
    row=dict(variant=variant, baseline=b.BASELINE, terminal=True, full_episode_completed=True,
        full_episode_requested=True, physical_steps=2, found_step=1, found_within_budget=True,
        stop_reason="success", pre_found_collision=False, pre_found_exposure_steps=1,
        searcher_motion_stall_proxy_agent_steps=0, searcher_hold_agent_steps=0, wall_seconds=.1,
        search_coverage=dict(available=True, pre_found_exposure_steps=1),
        episode_result=dict(termination_reason="success", terminal_step=2, success=True,
            first_collision_step=None, task_protocol="collision_terminal_v1"),
        **{k:scene[k] for k in ("original_episode_index", "scenario_id", "scenario_seed", "environment_innovation_seed")})
    row["controller"]={k:0 for k in b.ZERO_FIELDS}
    if variant != "V0":row["controller"]["safe_search"]={"options":copy.deepcopy(b.OPTIONS[variant])}
    row["complete_episode_row"]=dict(method="ch3_baseline_bser_prior", reference_runtime_method="ch3_baseline_bser_prior",
        found=True, found_step=1, success=True, actual_length=2, episode_length=2, terminal_step=2,
        termination_reason="success", scenario_id=scene["scenario_id"], scenario_seed=scene["scenario_seed"],
        environment_innovation_seed=scene["environment_innovation_seed"], evaluation_episode_index=scene["original_episode_index"],
        optimizer_update_count=0, training_update=False, first_collision_step=None)
    arm=directory/f"episode_{scene['original_episode_index']:04d}"/variant
    arm.mkdir(parents=True)
    write_json(arm/'summary.json',row)
    trace=[dict(step_before=i-1,step_after=i,before=dict(step=i-1,found=i>1),after=dict(step=i,found=True),
        search_transition=i==1,search_coverage=dict(step=i,pre_found_exposure_steps=int(i==1))) for i in (1,2)]
    (arm/'step_trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace),encoding='utf-8')
    summary=summarize([row],[variant],1,True)
    summary['source_and_input_verification_passed']=True
    for n,v in [('identity',identity),('summary',summary),('episodes',[row])]:write_json(directory/f'{n}.json',v)
    return row


class B1PlanTests(unittest.TestCase):
    def test_only_three_b1_variants_and_fixed_scene_seed(self):
        cmd=b.child_command(Path('m'),Path('c'),Path('out'),97,'V5')
        self.assertEqual(cmd[cmd.index('--baseline')+1],'B1_bser_prior')
        self.assertEqual(cmd[cmd.index('--seed')+1],'12729')
        self.assertEqual(cmd[cmd.index('--episode-indices')+1],'97')
        self.assertIn('--full-episodes',cmd)
        for i,v in [(1,'V5'),(97,'V4')]:
            with self.assertRaises(ValueError):b.child_command('m','c','o',i,v)

    def test_no_training_or_population_switch(self):
        plan=make_plan({})
        b.validate_plan(plan)
        for key,val in [('baseline','B0_search_prior'),('variants',['V5']),('seed',12826),('max_steps',30),('training',True)]:
            mutated=copy.deepcopy(plan);mutated[key]=val
            with self.subTest(key=key),self.assertRaises(ValueError):b.validate_plan(mutated)
        mutated=copy.deepcopy(plan);mutated['options']['V5']['failure_policy']=True
        with self.assertRaises(ValueError):b.validate_plan(mutated)

    def test_explicit_execution_required(self):
        args=['--manifest','m','--output-dir','o','--b0-reference-dir','r','--b0-ac-dir','a']
        with self.assertRaises(SystemExit):b.parser().parse_args(args)
        self.assertTrue(b.parser().parse_args(args+['--execute']).execute)


class B1AccountingTests(unittest.TestCase):
    def setup_run(self,root,fail=False):
        manifest,config,plan=fixtures()
        selected=b.validate_development_inputs(manifest,config,plan)
        for n,v in [('manifest',manifest),('config',config),('frozen',plan)]:write_json(root/f'{n}.json',v)
        child_inputs={str(root/f'{n}.json'):file_hash(root/f'{n}.json') for n in ['manifest','config']}
        refs={}
        for label in ['B0_D2','B0_AC']:
            d=root/label;d.mkdir()
            write_json(d/'identity.json',{})
            write_json(d/'episodes.json',[])
            refs[label]=dict(directory=str(d),identity_sha256=file_hash(d/'identity.json'),
                episodes_sha256=file_hash(d/'episodes.json'),source_inventory_sha256=label,
                selected=selected,child_input_sha256=child_inputs)
        write_json(root/'b1.json',make_plan(refs))
        source=dict(inventory=dict(sha256='new',files={}))
        calls=[]
        def child(cmd,log):
            i=int(cmd[cmd.index('--episode-indices')+1]);v=cmd[cmd.index('--variants')+1]
            calls.append((i,v));Path(log).write_text('fake B1 process',encoding='utf-8')
            if fail:return dict(returncode=7)
            directory=Path(cmd[cmd.index('--output-dir')+1]);parent=read_json(directory.parent/'identity.json')
            write_child(directory,next(s for s in selected if s['original_episode_index']==i),v,parent)
            return dict(returncode=0)
        with (patch.object(b,'PLAN',root/'frozen.json'),patch.object(b,'B1_PLAN',root/'b1.json'),
              patch.object(b,'reference_metadata',return_value=refs),patch.object(b,'_run_child',child),
              patch('chapter3_bser.experiments.safe_search_v1.provenance.framework_sources',return_value=source),
              patch('chapter3_bser.experiments.safe_search_v1.provenance.verify_sources')):
            kw=dict(b0_reference_dir=root/'B0_D2',b0_ac_dir=root/'B0_AC',config=root/'config.json',workers=1)
            if fail:
                with self.assertRaisesRegex(RuntimeError,'no complete evaluation'):b.run_b1_development(root/'manifest.json',root/'out',**kw)
            else:
                b.run_b1_development(root/'manifest.json',root/'out',**kw)
                parent,rows=b.collect_completed_b1(root/'out')
                self.assertEqual(len(rows),60)
                self.assertEqual(len({(r['original_episode_index'],r['variant']) for r in rows}),60)
                self.assertEqual(parent['planned_episode_runs'],60)
        return calls,selected

    def test_all60_actual_terminal_records_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);calls,_=self.setup_run(root)
            self.assertEqual(calls,[(i,v) for i in b.INDICES for v in b.VARIANTS])
            with patch.object(b,'PLAN',root/'frozen.json'),patch.object(b,'B1_PLAN',root/'b1.json'):
                p=b.arm_directory(root/'out',97,'V5')/'episodes.json'
                p.write_text('[]',encoding='utf-8')
                with self.assertRaises(ValueError):b.collect_completed_b1(root/'out')

    def test_failure_is_not_timeout_or_dropped_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);calls,_=self.setup_run(root,True)
            self.assertEqual(calls,[(0,'V0')])
            f=read_json(root/'out/failure.json')
            self.assertEqual(f['planned_episode_runs'],60)
            self.assertEqual(len(f['unstarted_pairs']),59)
            self.assertFalse((root/'out/summary.json').exists())

    def test_method_residual_prefix_and_trace_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);m,c,p=fixtures();scene=b.validate_development_inputs(m,c,p)[0]
            parent=dict(sources_before={},child_input_sha256={})
            directory=root/'child';row=write_child(directory,scene,'V5',parent)
            b.validate_child(directory,scene,'V5',parent)
            for kind in ['baseline','method','residual','prefix','overlap','options']:
                r=copy.deepcopy(row)
                if kind=='baseline':r['baseline']='B0_search_prior'
                elif kind=='method':r['complete_episode_row']['method']='ch3_hgr'
                elif kind=='residual':r['controller']['physical_residual_acceleration_max_abs']=.1
                elif kind=='prefix':r['terminal']=False
                elif kind=='overlap':r['searcher_hold_agent_steps']=4
                else:r['controller']['safe_search']['options']['failure_policy']=True
                with self.subTest(kind=kind),self.assertRaises(ValueError):b.validate_terminal(r,scene,'V5')
            trace=directory/'episode_0000/V5/step_trace.jsonl'
            trace.write_text(trace.read_text(encoding='utf-8').splitlines()[0]+'\n',encoding='utf-8')
            with self.assertRaises(ValueError):b.validate_child(directory,scene,'V5',parent)


if __name__=='__main__':unittest.main()
