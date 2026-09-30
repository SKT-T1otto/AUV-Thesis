"""Seven-arm scheduling, complete paired units, resume and exact source gates."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chapter3_bser.experiments.bser_final_v1 import run_windows as run, provenance
from chapter3_bser.experiments.bser_final_v1.options import ARMS, SETTINGS
from tests.test_bser_effect_runner import row as d_row, scene
from tests.test_safe_search_provenance import checked_copy


def row(index=0,arm="F0",found=None,collision=None):
    result=d_row(index,"D2",found,collision)
    result.update(arm=arm,baseline="bser_final_v1",episode_wall_seconds=2.)
    result["complete_episode_row"]["method"]="ch3_bser_final_v1_"+arm
    result["controller"]["bser_final_v1"]=dict(arm=arm,options=ARMS[arm],settings=SETTINGS,
        original_D2_delegate=arm=="F0",proposal_count=2,nonreference_proposals=0,
        fallback_counts={},planning_seconds=[.1,.2],total_response_queries=0,prediction_error_records=0)
    return result


def plan(n=3):
    return dict(baselines=["F"],arms=list(ARMS),planned_episode_runs=n*7,
        selected=[scene(i) for i in range(n)],sources={},input_sha256={},stage="development")


class RunnerTests(unittest.TestCase):
    def test_default_is_plan_only_and_terminal_labels_match_actual_arm(self):
        with patch.object(run,"request",return_value=plan()),patch.object(run,"run_jobs") as jobs, \
             patch.object(run,"run_episode") as episode,patch("builtins.print"):
            run.main(["--output-dir","runs/unused"])
        jobs.assert_not_called();episode.assert_not_called()
        for arm in ARMS:
            for found,collision in ((None,None),(None,20),(10,20),(90,None)):
                run.validate_terminal(row(arm=arm,found=found,collision=collision),scene(),"F",arm)
        wrong=row();wrong["controller"]["bser_final_v1"]["arm"]="F1"
        with self.assertRaises(ValueError):run.validate_terminal(wrong,scene(),"F","F0")

    def test_incomplete_and_extra_rows_cannot_look_like_full_result(self):
        rows=[row(i,a,found=100 if i<2 else None) for a in ARMS for i in range(3)]
        result=run.summarize(rows,plan())
        self.assertTrue(result["experiment_complete"])
        self.assertIsNone(result["performance_passed"])
        self.assertAlmostEqual(result["groups"]["F0"]["penalized_found_steps_mean_400"],200.)
        self.assertEqual(result["paired"]["F1_vs_F2"]["n_pairs"],3)
        self.assertEqual(result["groups"]["F0"]["computation"]["planning_event_count"],6)
        self.assertIsNone(run.summarize(rows[:-1],plan())["groups"]["F6"]["found_rate"])
        for extra in (rows[0],row(99),row(0,"F0")):
            with self.assertRaises(ValueError):run.summarize(rows+[extra],plan())

    def test_seven_arms_resume_and_stage_are_propagated_without_real_processes(self):
        completed=lambda d,p,b,a,s:row(s["original_episode_index"],a)
        with tempfile.TemporaryDirectory() as temp,patch.object(run,"ROOT",Path(temp)), \
             patch.object(run,"verify_sources"),patch.object(run,"read_completed",side_effect=completed), \
             patch.object(run.subprocess,"Popen") as spawn,patch("builtins.print"):
            spawn.return_value.poll.return_value=0
            args=run.parser().parse_args(["--output-dir",str(Path(temp)/"runs/F"),"--execute","--stage","confirmation"])
            self.assertTrue(run.run_jobs(args,plan())["experiment_complete"])
            self.assertEqual(spawn.call_count,21)
            self.assertIn("chapter3_bser.experiments.bser_final_v1.run_windows",spawn.call_args.args[0])
            self.assertIn("confirmation",spawn.call_args.args[0])
            spawn.reset_mock();args.resume=True
            run.run_jobs(args,plan());spawn.assert_not_called()
            wrong=plan();wrong["settings"]={"changed":True}
            with self.assertRaises(ValueError):run.run_jobs(args,wrong)
            values=run.read_json(args.output_dir/"episodes.json")
            values[0]["artifact_dir"]="../../outside"
            run.write_json(args.output_dir/"episodes.json",values)
            with self.assertRaisesRegex(ValueError,"escaped"):run.run_jobs(args,plan())

    def test_failure_is_not_a_physical_timeout_or_silent_retry(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(run,"ROOT",Path(temp)), \
             patch.object(run,"verify_sources"),patch.object(run.subprocess,"Popen") as spawn,patch("builtins.print"):
            spawn.return_value.poll.return_value=1
            args=run.parser().parse_args(["--output-dir",str(Path(temp)/"runs/F"),"--execute","--workers","1"])
            with self.assertRaises(RuntimeError):run.run_jobs(args,plan())
            self.assertEqual(run.read_json(args.output_dir/"episodes.json"),[])
            self.assertFalse(run.read_json(args.output_dir/"summary.json")["experiment_complete"])


class SourceTests(unittest.TestCase):
    def test_only_new_namespace_and_exact_dispatch_hook_changed(self):
        current=provenance.framework_sources()
        parent=provenance.old.read_json(provenance.old.ROOT/provenance.PARENT)
        # F's transition stays frozen; the current D2 transition is separately
        # validated by framework_sources, including every actual source byte.
        sealed=provenance.old.read_json(provenance.old.ROOT/provenance.MANIFEST)
        after=sealed["profiles"][current["checkout_profile"]]["files"]
        for name,sha in parent["profiles"][current["checkout_profile"]]["files"].items():
            if name!=provenance.HOOK:
                self.assertEqual(after[name],sha,name)
        self.assertEqual(current["historical_record_count"],27)

    def test_all_profiles_parent_and_source_corruption_rejected(self):
        with checked_copy() as root:
            original=provenance.old.read_json(root/provenance.MANIFEST)
            for mode in ("scope","before","parent","unused_profile"):
                changed=copy.deepcopy(original)
                if mode=="scope":changed["added_paths"].append("core/unreviewed.py")
                elif mode=="before":changed["profiles"]["windows_existing"]["changes"][provenance.HOOK]["before"]="0"*64
                elif mode=="parent":changed["parent_manifest_sha256"]="0"*64
                else:changed["profiles"]["git_lf"]["sha256"]="0"*64
                changed.pop("sha256");changed["sha256"]=provenance.old.digest(changed)
                run.write_json(root/provenance.MANIFEST,changed)
                with self.assertRaises(ValueError):provenance.old.framework_sources(root)
            # Restore exact historical bytes, not reserialized JSON.
            (root/provenance.MANIFEST).write_bytes((provenance.old.ROOT/provenance.MANIFEST).read_bytes())
            for name in (provenance.REVIEW,"chapter3_bser/experiments/bser_final_v1/solver.py"):
                path=root/name;raw=path.read_bytes();path.write_bytes(raw+b"\n")
                with self.assertRaises(ValueError):provenance.old.framework_sources(root)
                path.write_bytes(raw)


class ScenarioTests(unittest.TestCase):
    def manifest(self):
        from chapter3_bser.experiments.bser_final_v1 import scenarios as sc
        scenes=[dict(scenario_id="fresh_"+str(i),scenario_seed=1000+i,scenario_split="validation",
            scenario_profile="fixture",max_steps=400,fixture_geometry=i) for i in range(2)]
        value=dict(schema=sc.SCHEMA,generator_seed=sc.GENERATOR_SEED,innovation_seed=sc.INNOVATION_SEED,
            count=2,scenarios=scenes,sources={"fixture_source":1},excluded_reference=dict(
                scenario_ids=["old_"+str(i) for i in range(100)],scenario_seeds=list(range(100)),
                content_keys=["old_content_"+str(i) for i in range(100)]))
        value["sha256"]=run.digest(value)
        return value,dict(profile="fixture",max_steps=400)

    def test_fresh_population_is_frozen_and_source_bound_without_generation(self):
        from chapter3_bser.experiments.bser_final_v1.scenarios import validate_confirmation, INNOVATION_SEED
        manifest,config=self.manifest()
        with patch.object(provenance,"framework_sources",return_value={"fixture_source":1}):
            selected,seed=validate_confirmation(manifest,config)
            self.assertEqual(len(selected),2)
            self.assertEqual(selected[1]["environment_innovation_seed"],INNOVATION_SEED+1)
            manifest["scenarios"][0]["fixture_geometry"]=99
            with self.assertRaises(ValueError):validate_confirmation(manifest,config)
        manifest,config=self.manifest()
        with patch.object(provenance,"framework_sources",return_value={"fixture_source":2}):
            with self.assertRaisesRegex(ValueError,"code changed"):validate_confirmation(manifest,config)

    def test_resealed_overlap_or_duplicate_is_still_rejected(self):
        from chapter3_bser.experiments.bser_final_v1.scenarios import validate_confirmation
        for mode in ("old_seed","old_id","duplicate"):
            manifest,config=self.manifest()
            if mode=="old_seed":manifest["scenarios"][0]["scenario_seed"]=0
            elif mode=="old_id":manifest["scenarios"][0]["scenario_id"]="old_0"
            else:manifest["scenarios"][1]=copy.deepcopy(manifest["scenarios"][0])
            manifest.pop("sha256");manifest["sha256"]=run.digest(manifest)
            with patch.object(provenance,"framework_sources",return_value={"fixture_source":1}),self.assertRaises(ValueError):
                validate_confirmation(manifest,config)


if __name__=="__main__":
    unittest.main()
