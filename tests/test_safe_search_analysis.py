"""Offline development accounting must not hide incomplete or failed arms."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from chapter3_bser.experiments.safe_search_v1.analyze_development import (
    analyze_rows, choose_variant, collision_role, paired_bootstrap, trace_summary,
    validate_rows,
)


def fixtures():
    scenes = [dict(source_episode_index=i, scenario_id=f"scene_{i}", scenario_seed=12729+i,
                   environment_innovation_seed=12729+i) for i in range(20)]
    plan = dict(splits=dict(development=scenes), statistics=dict(bootstrap_replicates=100,
                bootstrap_seed=20260928))
    rows = []
    for scene in scenes:
        for variant in ("V0", "V1", "V2", "V3", "V4"):
            rows.append(dict(original_episode_index=scene["source_episode_index"], variant=variant,
                scenario_id=scene["scenario_id"], scenario_seed=scene["scenario_seed"],
                environment_innovation_seed=scene["environment_innovation_seed"],
                baseline="B0_search_prior", terminal=True, full_episode_completed=True,
                full_episode_requested=True, physical_steps=400, found_within_budget=False,
                found_step=None, pre_found_collision=False, pre_found_exposure_steps=400,
                searcher_motion_stall_proxy_agent_steps=30, searcher_hold_agent_steps=0,
                stop_reason="timeout", controller={}, wall_seconds=1.0,
                episode_result=dict(termination_reason="timeout", terminal_step=400,
                    success=False, first_collision_step=None, task_protocol="collision_terminal_v1")))
    return rows, plan


def set_found(row, step=100, success=True):
    row.update(found_step=step, found_within_budget=True, pre_found_exposure_steps=step,
               searcher_motion_stall_proxy_agent_steps=0)
    if success:
        row.update(stop_reason="success", physical_steps=max(1, step + 20))
        row["episode_result"].update(termination_reason="success", success=True,
                                      terminal_step=row["physical_steps"])


class SafeSearchAnalysisTests(unittest.TestCase):
    def test_complete_schedule_and_no_found_timeout(self):
        rows, plan = fixtures()
        result = analyze_rows(rows, plan)
        self.assertEqual(len(validate_rows(rows, plan)), 100)
        base = result["variants"]["V0"]
        self.assertEqual(base["outcome_partition"], {"no_found_timeout": 20})
        self.assertEqual(base["found_time_restricted_mean_400"], 400)
        self.assertIsNone(base["mean_found_step_conditional"])
        self.assertEqual(base["searcher_motion_stall_proxy_fraction"], .025)

    def test_missing_duplicate_outside_and_reseeded_fail_closed(self):
        for mutation in (lambda r: r.pop(), lambda r: r.append(copy.deepcopy(r[0])),
                         lambda r: r[0].update(original_episode_index=100),
                         lambda r: r[0].update(environment_innovation_seed=1)):
            rows, plan = fixtures()
            mutation(rows)
            with self.assertRaises(ValueError):
                validate_rows(rows, plan)

    def test_cutoff_cannot_become_timeout(self):
        for mutation in (lambda r: r.update(terminal=False),
                         lambda r: r.update(stop_reason="budget_cutoff"),
                         lambda r: r.update(physical_steps=100)):
            rows, plan = fixtures()
            mutation(rows[0])
            with self.assertRaises(ValueError):
                validate_rows(rows, plan)

    def test_found_zero_and_success_are_distinct(self):
        rows, plan = fixtures()
        row = rows[1]
        set_found(row, step=0, success=False)
        result = analyze_rows(rows, plan)
        arm = result["variants"]["V1"]
        self.assertEqual(arm["found_count"], 1)
        self.assertEqual(arm["success_count"], 0)
        self.assertEqual(arm["outcome_partition"]["post_found_timeout"], 1)
        self.assertEqual(arm["found_time_restricted_mean_400"], 380)
        self.assertEqual(arm["mean_found_step_conditional"], 0)

    def test_early_collision_is_not_fast_found(self):
        rows, plan = fixtures()
        rows[0].update(physical_steps=18, pre_found_exposure_steps=18,
                       stop_reason="obstacle_collision", pre_found_collision=True)
        rows[0]["episode_result"].update(termination_reason="obstacle_collision",
            terminal_step=18, first_collision_step=18, first_collision_agent_ids=[3])
        result = analyze_rows(rows, plan)
        arm = result["variants"]["V0"]
        self.assertEqual(arm["found_time_restricted_mean_400"], 400)
        self.assertEqual(arm["outcome_partition"]["pre_found_collision_executor"], 1)
        self.assertEqual(arm["pre_found_exposure_steps"], 19*400+18)

    def test_collision_roles_partition_and_unknown_is_not_zero(self):
        for ids, expected in (([0], "searcher"), ([0, 2], "searcher"), ([3], "executor"), ([0, 3], "mixed")):
            self.assertEqual(collision_role(dict(episode_result=dict(first_collision_agent_ids=ids))), expected)
        with self.assertRaises(ValueError):
            collision_role(dict(episode_result=dict(first_collision_agent_ids=[])))

    def test_paired_discordants_and_deterministic_interval(self):
        rows, plan = fixtures()
        set_found(rows[1])
        set_found(rows[5])
        result = analyze_rows(rows, plan)
        paired = result["paired_vs_v0"]["V1"]["found"]
        self.assertEqual(paired["n_pairs"], 20)
        self.assertEqual(paired["discordant_0_to_1"], 1)
        self.assertEqual(paired["discordant_1_to_0"], 1)
        self.assertEqual(paired["mean_delta"], 0)
        self.assertEqual(paired_bootstrap([1, 0, -1], 100), paired_bootstrap([1, 0, -1], 100))
        self.assertEqual(paired_bootstrap([1]*20)["bootstrap_95_percentile_ci"], [1, 1])

    def test_selection_needs_strict_gain_and_d1(self):
        rows, plan = fixtures()
        set_found(rows[1])
        no_d1 = analyze_rows(rows, plan)
        self.assertIsNone(no_d1["selection"]["selected_variant"])
        result = analyze_rows(rows, plan, d1_passed=True)
        self.assertEqual(result["selection"]["selected_variant"], "V1")

    def test_parallel_tie_does_not_silently_skip_registered_wall_criterion(self):
        rows, plan = fixtures()
        set_found(rows[1])
        set_found(rows[2])
        result = analyze_rows(rows, plan, d1_passed=True)
        self.assertIsNone(result["selection"]["selected_variant"])
        self.assertEqual(result["selection"]["tied_variants"], ["V1", "V2"])
        serial = choose_variant(result["variants"], d1_passed=True, wall_time_comparable=True)
        self.assertEqual(serial["selected_variant"], "V1")

    def test_trace_query_scope_is_pre_found_and_agent_switches_recorded(self):
        before_agents = [dict(agent_id=i, assignment_id="a", position=[0, 0, 0]) for i in range(4)]
        after_agents = [dict(agent_id=i, assignment_id="b", position=[1, 0, 0]) for i in range(4)]
        lines = []
        for step in range(2):
            lines.append(dict(step_before=step, step_after=step+1, search_transition=step==0,
                before=dict(agents=before_agents), after=dict(agents=after_agents),
                queries=dict(counts=[dict(reason="no_start_connector", caller="candidate", count=7)])))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in lines), encoding="utf-8")
            result = trace_summary(path)
        self.assertEqual(result["physical_steps"], 2)
        self.assertEqual(result["pre_found_query_reasons"], {"no_start_connector": 7})
        self.assertEqual(result["assignment_switches_by_agent"]["0"], 1)
        self.assertEqual(result["pre_found_path_length_by_agent"]["0"], 1)


if __name__ == "__main__":
    unittest.main()
