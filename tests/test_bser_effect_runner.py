"""Manual launch, paired estimands, strict resume and additive source guards."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chapter3_bser.experiments.bser_effect_v1 import run_windows as run, provenance
from chapter3_bser.experiments.bser_effect_v1.options import ARMS, SETTINGS, parse_arms
from tests.test_found_search_v2_runner import scene, row as parent_row
from tests.test_safe_search_provenance import checked_copy


def row(index=0, arm="D0", found=None, collision=None):
    value=parent_row(index,found=found,collision=collision)
    value.update(arm=arm,baseline="bser_effect_v1",parent_variant="V4")
    value["controller"]["bser_effect_v1"]=dict(arm=arm,options=ARMS[arm],settings=SETTINGS,
        executor_pre_found_distance=3.,response_weight_search_change_proposals=0,proposal_count=2)
    value["complete_episode_row"]=dict(method="ch3_bser_effect_v1_"+arm,found=found is not None,found_step=found,
        actual_length=value["physical_steps"],episode_length=value["physical_steps"],termination_reason=value["stop_reason"],
        scenario_id=scene(index)["scenario_id"],scenario_seed=10+index,evaluation_episode_index=index,
        environment_innovation_seed=12729+index,training_update=False,optimizer_update_count=0)
    return value


def plan():
    return dict(baselines=["D"],arms=list(ARMS),planned_episode_runs=80,selected=[scene(i) for i in range(20)],
                sources={},input_sha256={})


class RunnerTests(unittest.TestCase):
    def test_default_is_plan_only(self):
        with patch.object(run,"request",return_value=plan()),patch.object(run,"run_jobs") as jobs, \
             patch.object(run,"run_episode") as episode, patch("builtins.print"):
            run.main(["--output-dir","runs/unused"])
        jobs.assert_not_called()
        episode.assert_not_called()

    def test_group_names_and_terminal_contract(self):
        self.assertEqual(parse_arms("D0,D1,D2,D3"),list(ARMS))
        for bad in ("D3","D0,D0","R0,R4","D0,",""):
            with self.assertRaises(ValueError): parse_arms(bad)
        for arm in ARMS:
            for found,collision in ((None,None),(None,20),(10,20),(90,None)):
                run.validate_terminal(row(arm=arm,found=found,collision=collision),scene(),"D",arm)
        for key,bad in (("physical_steps",5),("terminal",False),("found_step",401),("pre_found_collision",True)):
            value=row()
            value[key]=bad
            with self.assertRaises(ValueError): run.validate_terminal(value,scene(),"D","D0")
        value=row()
        value["complete_episode_row"]["method"]="ch3_baseline_bser_prior"
        with self.assertRaises(ValueError): run.validate_terminal(value,scene(),"D","D0")

    def test_pairing_retains_failure_penalty_and_no_results_are_called_success(self):
        rows=[row(i,a,found=100 if i<10 else None) for a in ARMS for i in range(20)]
        summary=run.summarize(rows,plan())
        self.assertTrue(summary["experiment_complete"])
        self.assertIsNone(summary["performance_passed"])
        self.assertEqual(summary["groups"]["D3"]["penalized_found_steps_mean_400"],250)
        self.assertEqual(summary["groups"]["D3"]["found_steps_mean_conditional"],100)
        paired=summary["paired"]["D3_vs_D0"]
        self.assertEqual(paired["found_mcnemar_exact_two_sided"],1.)
        self.assertEqual(paired["statistics"]["found_delta"]["paired_bootstrap_95_interval"],[0.,0.])
        incomplete=run.summarize(rows[:-1],plan())
        self.assertIsNone(incomplete["groups"]["D3"]["found_rate"])
        self.assertFalse(incomplete["experiment_complete"])
        with self.assertRaises(ValueError): run.summarize(rows+[rows[0]],plan())

    def test_scheduler_and_resume_use_same_verified_scene_jobs_without_real_processes(self):
        completed=lambda d,p,b,a,s:row(s["original_episode_index"],a)
        with tempfile.TemporaryDirectory() as temp,patch.object(run,"ROOT",Path(temp)), \
             patch.object(run,"verify_sources"),patch.object(run,"read_completed",side_effect=completed), \
             patch.object(run.subprocess,"Popen") as spawn,patch("builtins.print"):
            spawn.return_value.poll.return_value=0
            args=run.parser().parse_args(["--output-dir",str(Path(temp)/"runs/D"),"--execute"])
            self.assertTrue(run.run_jobs(args,plan())["experiment_complete"])
            self.assertEqual(spawn.call_count,80)
            self.assertIn("chapter3_bser.experiments.bser_effect_v1.run_windows",spawn.call_args.args[0])
            original=(args.output_dir/"episodes.json").read_bytes()
            spawn.reset_mock()
            args.resume=True
            run.run_jobs(args,plan())
            spawn.assert_not_called()
            self.assertEqual((args.output_dir/"episodes.json").read_bytes(),original)
            wrong=plan(); wrong["settings"]={"changed":True}
            with self.assertRaises(ValueError): run.run_jobs(args,wrong)
            values=run.read_json(args.output_dir/"episodes.json")
            values[0]["artifact_dir"]="../../outside"
            run.write_json(args.output_dir/"episodes.json",values)
            with self.assertRaisesRegex(ValueError,"escaped"): run.run_jobs(args,plan())

    def test_program_failure_is_incomplete_and_protected_outputs_rejected(self):
        args=run.parser().parse_args(["--output-dir",str(run.ROOT/"outputs/retained"),"--resume"])
        with self.assertRaises(ValueError): run.run_jobs(args,plan())
        with tempfile.TemporaryDirectory() as temp,patch.object(run,"ROOT",Path(temp)), \
             patch.object(run,"verify_sources"),patch.object(run.subprocess,"Popen") as spawn,patch("builtins.print"):
            spawn.return_value.poll.return_value=1
            args=run.parser().parse_args(["--output-dir",str(Path(temp)/"runs/D"),"--execute","--workers","1"])
            with self.assertRaises(RuntimeError): run.run_jobs(args,plan())
            self.assertEqual(run.read_json(args.output_dir/"episodes.json"),[])
            self.assertFalse(run.read_json(args.output_dir/"summary.json")["experiment_complete"])


class SourceTests(unittest.TestCase):
    def test_exact_evolution_preserves_all_old_runtime_bytes(self):
        result=provenance.framework_sources()
        parent=provenance.old.read_json(provenance.old.ROOT/provenance.PARENT)
        before=parent["profiles"][result["checkout_profile"]]["files"]
        # Check this historical transition against its own frozen end state.
        # framework_sources above separately validates the full D2 successor.
        sealed=provenance.old.read_json(provenance.old.ROOT/provenance.MANIFEST)
        after=sealed["profiles"][result["checkout_profile"]]["files"]
        for path,sha in before.items():
            if path != provenance.HOOK:
                self.assertEqual(after[path],sha,path)
        self.assertEqual(result["historical_record_count"],27)

    def test_all_profiles_parent_before_scope_and_sources_checked(self):
        with checked_copy() as root:
            original=provenance.old.read_json(root/provenance.MANIFEST)
            raw=(root/provenance.MANIFEST).read_bytes()
            for mode in ("scope","before","parent","unused_profile"):
                changed=copy.deepcopy(original)
                if mode=="scope": changed["added_paths"].append("core/unreviewed.py")
                elif mode=="before": changed["profiles"]["windows_existing"]["changes"][provenance.HOOK]["before"]="0"*64
                elif mode=="parent": changed["parent_manifest_sha256"]="0"*64
                else: changed["profiles"]["git_lf"]["sha256"]="0"*64
                changed.pop("sha256")
                changed["sha256"]=provenance.old.digest(changed)
                (root/provenance.MANIFEST).write_text(json.dumps(changed),encoding="utf-8")
                with self.subTest(mode=mode),self.assertRaises(ValueError): provenance.old.framework_sources(root)
            (root/provenance.MANIFEST).write_bytes(raw)
            for name in (provenance.REVIEW,"chapter3_bser/experiments/bser_effect_v1/solver.py"):
                source=root/name
                raw_source=source.read_bytes()
                source.write_bytes(raw_source+b"\n")
                with self.assertRaises(ValueError): provenance.old.framework_sources(root)
                source.write_bytes(raw_source)


if __name__ == "__main__":
    unittest.main()
