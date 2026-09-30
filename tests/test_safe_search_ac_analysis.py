"""Offline V5 accounting, zero-exposure and separate-gate checks; no simulator."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from chapter3_bser.experiments.safe_search_v1.run_development import INDICES
from docs.chapter3.search_diagnostics.safe_search_v1 import analyze_ac_development as ac


def fixture():
    old, new, traces = [], [], {}
    for index in INDICES:
        base = dict(original_episode_index=index, scenario_id=f"scene_{index}", scenario_seed=12729 + index,
            environment_innovation_seed=12729 + index, baseline="B0_search_prior", terminal=True,
            full_episode_completed=True, full_episode_requested=True, physical_steps=400,
            found_step=None, found_within_budget=False, pre_found_collision=False, pre_found_exposure_steps=400,
            stop_reason="timeout", searcher_motion_stall_proxy_agent_steps=30, searcher_hold_agent_steps=0,
            episode_result=dict(success=False, termination_reason="timeout", terminal_step=400,
                first_collision_step=None, first_collision_agent_ids=[], task_protocol="collision_terminal_v1"),
            search_coverage=dict(available=True, effective_observation_steps=10, observation_update_steps=200,
                unique_observed_grid_fraction=.1, new_cell_observations=80, aged_revisit_cell_observations=2,
                repeated_cell_observations=100), controller={}, wall_seconds=1)
        for variant in (*ac.VARIANTS, ac.NEW_VARIANT):
            row = dict(copy.deepcopy(base), variant=variant)
            (new if variant == ac.NEW_VARIANT else old).append(row)
            traces[index, variant] = dict(physical_steps=400, pre_found_query_reasons={"reachable": 1, "no_start_connector": 9})
    return old, new, traces


def mark_found(row, step=100):
    row.update(found_step=step, found_within_budget=True, pre_found_exposure_steps=step,
               searcher_motion_stall_proxy_agent_steps=0, searcher_hold_agent_steps=0)


def summary(found=7, collision=5, stall=100, hold=10, exposure=1000, effective=.3):
    return dict(found_count=found, pre_found_collision_count=collision,
        searcher_motion_stall_proxy_agent_steps=stall, searcher_hold_agent_steps=hold,
        pre_found_exposure_steps=exposure, actual_belief_footprint=dict(effective_observation_fraction=effective))


class ACAnalysisTests(unittest.TestCase):
    def test_exact_twenty_and_identity(self):
        old, new, _ = fixture()
        self.assertEqual(set(ac.validate_new_rows(new, old)), set(INDICES))
        for mutate in (lambda rows: rows.pop(), lambda rows: rows.append(copy.deepcopy(rows[0])),
                       lambda rows: rows[0].update(variant="V4"),
                       lambda rows: rows[0].update(environment_innovation_seed=999)):
            changed = copy.deepcopy(new)
            mutate(changed)
            with self.assertRaises(ValueError):
                ac.validate_new_rows(changed, old)

    def test_terminal_and_coverage_fail_closed(self):
        old, new, _ = fixture()
        for mutate in (lambda r: r.update(physical_steps=399), lambda r: r.update(terminal=False),
                       lambda r: r.update(found_within_budget=True),
                       lambda r: r["search_coverage"].update(available=False),
                       lambda r: r.update(searcher_hold_agent_steps=1200)):
            changed = copy.deepcopy(new)
            mutate(changed[0])
            with self.assertRaises(ValueError):
                ac.validate_new_rows(changed, old)

    def test_found_zero_is_valid_and_collision_same_step_has_precedence(self):
        old, new, _ = fixture()
        mark_found(new[0], 0)
        mark_found(new[1], 10)
        new[1].update(stop_reason="obstacle_collision", physical_steps=10, pre_found_collision=True)
        new[1]["episode_result"].update(termination_reason="obstacle_collision", terminal_step=10,
            first_collision_step=10, first_collision_agent_ids=[1])
        ac.validate_new_rows(new, old)
        self.assertEqual(ac.original.outcome(new[1]), "pre_found_collision_searcher")
        self.assertEqual(new[0]["pre_found_exposure_steps"], 0)

    def test_zero_exposure_rates_are_not_silently_zero(self):
        old, new, traces = fixture()
        mark_found(new[0], 0)
        refs = [r for r in old if r["variant"] == "V3"]
        result = ac.paired_metrics(list(zip(refs, new)), traces, replicates=30)
        self.assertEqual(result["metrics"]["found"]["n_pairs"], 20)
        self.assertEqual(result["metrics"]["found"]["discordant_0_to_1"], 1)
        self.assertEqual(result["metrics"]["stall_fraction"]["n_pairs"], 19)
        self.assertEqual(result["metrics"]["stall_fraction"]["excluded_zero_exposure_original_indices"], [INDICES[0]])
        self.assertEqual(result["metrics"]["restricted_found_time_400"]["mean_delta"], -20)

    def test_paired_discordants_not_only_net_gain(self):
        old, new, traces = fixture()
        refs = [r for r in old if r["variant"] == "V3"]
        mark_found(refs[0]); mark_found(new[1]); mark_found(new[2])
        pairs = list(zip(refs, new))
        first = ac.paired_metrics(pairs, traces, replicates=30)
        self.assertEqual(first, ac.paired_metrics(pairs, traces, replicates=30))
        self.assertEqual(first["metrics"]["found"]["discordant_0_to_1"], 2)
        self.assertEqual(first["metrics"]["found"]["discordant_1_to_0"], 1)
        self.assertEqual(first["metrics"]["found"]["mean_delta"], .05)

    def test_gate_uses_combined_stall_hold_and_strict_improvement(self):
        summaries = {"V3": summary(), "V5": summary(stall=90)}
        self.assertTrue(ac.gate_result(summaries, source_comparability_passed=True)["passed"])
        summaries["V5"] = summary(stall=0, hold=110)
        self.assertFalse(ac.gate_result(summaries, source_comparability_passed=True)["passed"])
        summaries["V5"] = summary(stall=0, hold=120)
        self.assertFalse(ac.gate_result(summaries, source_comparability_passed=True)["passed"])
        summaries["V5"] = summary(stall=0, collision=6)
        self.assertFalse(ac.gate_result(summaries, source_comparability_passed=True)["passed"])

    def test_gate_requires_actual_comparability_and_defined_rates(self):
        summaries = {"V3": summary(), "V5": summary(stall=0)}
        self.assertFalse(ac.gate_result(summaries, source_comparability_passed=False)["passed"])
        summaries["V5"] = summary(stall=0, exposure=0)
        self.assertFalse(ac.gate_result(summaries, source_comparability_passed=True)["passed"])

    def test_frozen_plan_rejects_changed_gate_and_controls(self):
        plan = ac.read_json(ac.AC_PLAN)
        ac.validate_plan(plan)
        for section, key, value in (("development_gate", "comparator", "V0"),
                ("development_gate", "stall_plus_hold_fraction_strictly_lower", False),
                ("statistics", "bootstrap_seed", 1),
                ("source_comparability", "old_arm_replay_controls", [])):
            changed = copy.deepcopy(plan)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                ac.validate_plan(changed)

    def test_output_cannot_overwrite_original_runs(self):
        with self.assertRaises(ValueError):
            ac.output_directory(ac.ROOT / "runs/new_source/not_created_report", [ac.ROOT / "runs/new_source"])
        with self.assertRaises(ValueError):
            ac.output_directory(ac.ROOT / "core/not_created_report", [])

    def test_source_comparability_reads_real_control_records_not_only_passed(self):
        old, _, _ = fixture()
        source = dict(inventory=dict(sha256="new-source"))
        identity = dict(sources_before=source, child_input_sha256={"manifest": "manifest-sha", "config": "config-sha"},
            selected=[dict(original_episode_index=i, scenario_id=f"scene_{i}", scenario_seed=12729+i,
                environment_innovation_seed=12729+i, scenario_sha256=f"scene-sha-{i}") for i in INDICES])
        plan = ac.read_json(ac.AC_PLAN)
        controls, data = [], {}
        for index, variant in [(3, "V3"), (28, "V4")]:
            row = next(r for r in old if r["original_episode_index"] == index and r["variant"] == variant)
            row["signature_sha256"] = f"{index:064x}"
            row["complete_episode_row"] = dict(episode_index=index, scenario_id=row["scenario_id"],
                scenario_seed=row["scenario_seed"], environment_innovation_seed=row["environment_innovation_seed"],
                found=False, found_step=None, success=False, actual_length=400, episode_length=400,
                termination_reason="timeout", first_collision_step=None, first_collision_agent_ids=[])
            directory = (ac.ROOT / "runs/not-created-ac-analysis-test-control" / str(index)).resolve()
            controls.append(dict(original_episode_index=index, variant=variant, replay_directory=str(directory)))
            data[str(directory / "identity.json")] = dict(schema="ch3.safe_search.paired_identity.v1", variants=[variant],
                baseline="B0_search_prior", selected=[s for s in identity["selected"] if s["original_episode_index"] == index],
                full_episodes=True, task_horizon=400, steps=400, seed=12729, sources_before=source, sources_after=source,
                source_and_input_verification_passed=True, training=False, checkpoint_loaded=False,
                input_sha256=identity["child_input_sha256"])
            data[str(directory / "summary.json")] = dict(all_requested_runs_recorded=True, source_and_input_verification_passed=True)
            data[str(directory / "episodes.json")] = [copy.deepcopy(row)]
            data[str(directory / f"episode_{index:04d}" / variant / "summary.json")] = copy.deepcopy(row)
        evidence_path = ac.ROOT / "runs/not-created-ac-analysis-test-control/evidence.json"
        data[str(evidence_path)] = dict(schema="ch3.safe_search.ac_source_comparability.v1", passed=True,
            reviewed_exact_source_diff_passed=True, new_source_inventory_sha256="new-source",
            reference_source_inventory_sha256="old-source", controls=controls)

        class Reader:
            def read(self, path):
                return data[str(Path(path))]

        reference_identity = dict(sources_before=dict(inventory=dict(sha256="old-source")))
        with patch.object(ac, "file_hash", return_value="file-sha"):
            result = ac.verify_comparability(Reader(), evidence_path, plan, identity, reference_identity, old)
            self.assertTrue(result["passed"])
            self.assertEqual(len(result["controls"]), 2)
            path = str(Path(controls[0]["replay_directory"]) / "episode_0003/V3/summary.json")
            data[path]["signature_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                ac.verify_comparability(Reader(), evidence_path, plan, identity, reference_identity, old)


if __name__ == "__main__":
    unittest.main()
