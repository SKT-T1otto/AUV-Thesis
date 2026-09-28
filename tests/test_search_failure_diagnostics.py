"""Diagnostic data contracts and observer noninterference; no simulator imports."""
import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace as NS
import tempfile
import unittest

from scripts import analyze_search_failures as a
from scripts.search_diagnostic_observer import QueryTap, TransitionObserver, segment_distance
from scripts.run_search_diagnostic_probe import run_probe
from scripts import summarize_search_trace as trace_summary


def episode(index=0,*,found=False,reason="timeout",method="ch3_baseline_search_prior"):
    end=400 if reason=="timeout" else 40
    return dict(scenario_id=f"scene_{index}",scenario_seed=index,method=method,
                environment_innovation_seed=index+10,found=found,success=reason=="success",
                found_step=10 if found else None,terminal_step=end,actual_length=end,
                episode_terminated=True,episode_truncated=False,termination_reason=reason,
                collision_episode=reason=="obstacle_collision",first_collision_step=end if reason=="obstacle_collision" else None,
                first_collision_agent_ids=[3] if reason=="obstacle_collision" else [],
                first_collision_phase=("Intercept" if found else "Search") if reason=="obstacle_collision" else None,
                task_protocol="collision_terminal_v1")


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding="utf-8")


def fixture(directory):
    rows=[episode(0),episode(1,reason="obstacle_collision"),episode(2,found=True,reason="success")]
    summary=dict(method="ch3_baseline_search_prior",n_valid_episodes=3,n_expected_episodes=3,evaluation_complete=True,
                 n_success=1,n_collision_failure=1,n_timeout=1,found_rate=1/3)
    save(directory/"episodes.json",rows);save(directory/"summary.json",summary)
    save(directory/"resolved_config.json",dict(common_task_conditions=dict(max_steps=400)))
    return rows,summary


class PopulationTests(unittest.TestCase):
    def test_hgr_only_main_not_auxiliary_or_prefix(self):
        main=dict(episode(method="ch3_hgr"),purpose="main",trajectory_complete=True,return_scope="full_mission")
        pilot=dict(main,purpose="pilot",trajectory_complete=False,return_scope="prefix_only")
        selected,scope,purposes,excluded=a.select_population([main,pilot,dict(main,purpose="suffix_training")],
            dict(method="ch3_hgr",completed_main_trajectories=1))
        self.assertEqual(selected,[main]);self.assertEqual(excluded,2)

    def test_incomplete_eval_and_partial_main_rejected(self):
        with self.assertRaises(ValueError):a.select_population([episode()],dict(method="ch3_baseline_search_prior",evaluation_complete=False))
        with self.assertRaises(ValueError):a.select_population([dict(episode(method="ch3_hgr"),purpose="main",trajectory_complete=False)],
            dict(method="ch3_hgr",completed_main_trajectories=1))

    def test_missing_main_is_not_denominator_zero(self):
        with self.assertRaises(ValueError):a.select_population([dict(episode(method="ch3_hgr"),purpose="pilot")],
            dict(method="ch3_hgr",completed_main_trajectories=1))

    def test_terminal_and_discovery_ordering(self):
        r=episode(found=True,reason="obstacle_collision")
        self.assertEqual(a.normalize_episode(r,400)["bucket"],"found_collision")
        for change in [dict(found_step=40),dict(first_collision_phase="Search"),dict(first_collision_step=39),dict(terminal_step=401)]:
            with self.subTest(change=change),self.assertRaises(ValueError):a.normalize_episode(dict(r,**change),400)

    def test_unknown_is_not_measured_zero(self):
        row=a.normalize_episode(episode(),400)
        self.assertIsNone(row["stagnant_steps"]);self.assertIsNone(row["effective_search_steps"])
        group=a.summarize_group([row],{})
        self.assertIsNone(group["allocation_counts_whole_episode"]["unreachable_search_queries"]["total"])

    def test_collision_exposure_not_productive_time(self):
        r=a.normalize_episode(episode(reason="obstacle_collision"),400)
        self.assertEqual(r["pre_found_exposure_steps"],40)
        self.assertEqual(r["completed_collision_free_search_steps"],39)
        self.assertIsNone(r["effective_search_steps"])

    def test_hgr_identity_uses_dataset_not_repeated_scene_name(self):
        row=episode(method="ch3_hgr")
        self.assertNotEqual(a.normalize_episode(dict(row,dataset_id="a"),400)["episode_key"],
                            a.normalize_episode(dict(row,dataset_id="b"),400)["episode_key"])

    def test_invalid_boolean_and_nonfinite_rejected(self):
        for mutation in (dict(found=1),dict(terminal_step=float("nan")),dict(terminal_step=3.5)):
            with self.assertRaises(ValueError):a.normalize_episode(dict(episode(),**mutation),400)


class FileTests(unittest.TestCase):
    def test_exact_duplicate_export_not_double_counted_and_sources_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/"source";fixture(source);fixture(source/"named")
            before={str(p):p.read_bytes() for p in source.rglob("*.json")}
            result=a.analyze(source)
            self.assertEqual(len(result["runs"]),1);self.assertEqual(len(result["duplicates"]),1)
            self.assertEqual(result["duplicates"][0]["canonical"],"named")
            a.write_report(result,root/"report")
            self.assertEqual(before,{str(p):p.read_bytes() for p in source.rglob("*.json")})
            with self.assertRaises(ValueError):a.write_report(result,source/"bad")
            with self.assertRaises(ValueError):a.write_report(result,root/"report")

    def test_duplicate_rows_and_summary_disagreement_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);rows,summary=fixture(p)
            save(p/"episodes.json",[rows[0],rows[0],rows[2]])
            with self.assertRaises(ValueError):a.analyze_run(p,p)
            save(p/"episodes.json",rows);save(p/"summary.json",dict(summary,found_rate=0.9))
            with self.assertRaises(ValueError):a.analyze_run(p,p)

    def test_diagnostic_join_missing_or_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);fixture(p)
            for ds in [[dict(scenario_id="scene_0")],[dict(scenario_id="scene_0")]*2]:
                save(p/"controller_diagnostics.json",dict(episodes=ds))
                with self.assertRaises(ValueError):a.analyze_run(p,p)

    def test_training_not_paired_with_evaluation(self):
        pair=a.paired_evaluations([dict(population="training_main"),dict(population="training_smoke")])
        self.assertEqual(pair,[])

    def test_step_bound_checked_before_runtime_import(self):
        for steps in (0,101):
            with self.assertRaises(ValueError):run_probe(Path("unused"),Path("unused"),steps=steps)
        with self.assertRaises(ValueError):run_probe(Path("unused"),Path("unused"),live_public_map=True)


class QueryObserverTests(unittest.TestCase):
    def test_exact_call_result_rng_and_restore_on_exception(self):
        result=NS(reachable=False,failure_reason="no_start_connector");calls=[]
        class Service:
            def __init__(self):self.state=NS(step=3,map_revision=1,planning_graph=NS(endpoint_connectors=[]))
            def query(self,*args):calls.append(args);return result
        original=Service.query;state=random.getstate()
        with self.assertRaisesRegex(RuntimeError,"fixture"):
            with QueryTap(Service) as tap:
                self.assertIs(Service().query([0,0,0],[1,1,1],NS(agent_id=0,role="searcher")),result)
                d=tap.drain();self.assertEqual(d["counts"][0]["reason"],"no_start_connector")
                self.assertFalse(d["samples"][0]["start_endpoint_present"])
                raise RuntimeError("fixture")
        self.assertIs(Service.query,original);self.assertEqual(len(calls),1);self.assertEqual(random.getstate(),state)

    def test_nested_taps_rejected(self):
        class Service:
            def query(self,*args):pass
        with QueryTap(Service):
            with self.assertRaises(RuntimeError):
                with QueryTap(Service):pass

    def test_consumed_coordinate_iterators_do_not_change_query_result(self):
        result=NS(reachable=False,failure_reason="no_start_connector")
        class Service:
            state=NS(step=0,map_revision=0,planning_graph=NS(endpoint_connectors=[]))
            def query(self,start,goal,agent):
                tuple(start);tuple(goal);return result
        with QueryTap(Service) as tap:
            self.assertIs(Service().query(iter([0,0,0]),iter([1,1,1]),NS(agent_id=0,role="searcher")),result)
            self.assertIsNone(tap.drain()["samples"][0]["start"])

    def test_stall_proxy_requires_stable_assignment_window(self):
        def snap(step,assignment="a"):
            return dict(step=step,found=False,agents=[dict(agent_id=i,assignment_id=assignment,position=[0,0,0],
                remaining_path_length=5,speed=0,reachable=True,hold=False) for i in range(4)])
        observer=TransitionObserver(window=2);empty=dict(counts=[],samples=[])
        before=snap(0);saved=copy.deepcopy(before)
        self.assertFalse(observer.observe(before,snap(1),empty)["agent_metrics"][0]["motion_stall_proxy"])
        self.assertTrue(observer.observe(snap(1),snap(2),empty)["agent_metrics"][0]["motion_stall_proxy"])
        self.assertFalse(observer.observe(snap(2),snap(3,"b"),empty)["agent_metrics"][0]["motion_stall_proxy"])
        self.assertEqual(before,saved);self.assertIsNone(observer.result()["effective_search_steps"])

    def test_segment_error_includes_active_segment(self):
        self.assertEqual(segment_distance([1,0,0],[0,0,0],[2,0,0]),0)
        self.assertEqual(segment_distance([1,1,0],[0,0,0],[2,0,0]),1)


class TraceSummaryTests(unittest.TestCase):
    def test_anchor_failures_do_not_become_search_candidate_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            save(root/"identity.json",dict(scenario_id="fixture",source_before=dict(sha256="a"),
                 recorded_source_sha256="a",different_source_files=[]))
            save(root/"summary.json",dict(physical_steps=1,terminal=False,observer_noninterference={},telemetry={}))
            row=dict(step_before=0,step_after=1,collision_records=[],
                     before=dict(allocation_hash="x",cached_map_revision=1,live_map_revision=2),
                     after=dict(allocation_hash="x",decision_reason="ATOMIC_REJECT_MISSING_SEARCH_ROUTE"),
                     queries=dict(counts=[dict(caller="generate_search_candidates",reason="no_start_connector",count=800),
                                         dict(caller="_assignment",reason="invalid_goal",count=1)]))
            (root/"step_trace.jsonl").write_text(json.dumps(row)+"\n",encoding="utf-8")
            result=trace_summary.summarize(root)
            self.assertEqual(result["search_candidate_query_counts"],dict(no_start_connector=800))
            self.assertEqual(result["all_query_counts"]["invalid_goal"],1)
            self.assertEqual(result["controller_map_lag_steps"],1)
            self.assertTrue(result["atomic_reject_retained_allocation"][0]["same_allocation"])

    def test_public_grid_ties_and_bounds(self):
        mapping=dict(grid_origin=[0,0,0],grid_spacing=[2,2,1],grid_shape=[10,10,8])
        self.assertEqual(trace_summary.public_cell([1,1,0.5],mapping),0)
        self.assertEqual(trace_summary.public_cell([2,1,0.5],mapping),0)
        self.assertEqual(trace_summary.public_cell([2.001,1,0.5],mapping),80)
        self.assertIsNone(trace_summary.public_cell([-0.01,1,0.5],mapping))

    def test_posthoc_closed_segment_box_geometry(self):
        box=dict(center=[0,0,0],size=[2,2,2])
        self.assertTrue(trace_summary.segment_hits_box([-2,0,0],[2,0,0],box))
        self.assertTrue(trace_summary.segment_hits_box([-2,1,0],[2,1,0],box))
        self.assertFalse(trace_summary.segment_hits_box([-2,1.01,0],[2,1.01,0],box))

    def test_public_voxel_corner_contact_is_not_lost_by_point_sampling(self):
        mapping=dict(grid_origin=[0,0,0],grid_spacing=[2,2,1],grid_shape=[10,10,8],occupied_cell_indices=[113])
        self.assertEqual(trace_summary.occupied_voxel_intersections([3,9,2.5],[1,7,1.5],mapping),[113])
        self.assertEqual(trace_summary.occupied_voxel_intersections([3,9,2.6],[1,7,1.6],mapping),[])


if __name__=="__main__":unittest.main()
