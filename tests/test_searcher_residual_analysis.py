"""Pure offline file fixtures: no simulator, training, model or checkpoint load."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts import analyze_searcher_residual_effect as analysis


def write_csv(path, rows, fields=None):
    fields = fields or sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class SearcherResidualAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.full, self.off, self.output = (self.root/name for name in ("full", "off", "analysis"))
        self.rows = {}
        self.manifests = {}
        for path, mode, flags in (
            (self.full, "full_prrac", [(1, 1), (1, 0), (1, 1), (0, 0), (0, 0)]),
            (self.off, "searcher_residual_off", [(1, 0), (1, 1), (1, 1), (0, 0), (1, 1)]),
        ):
            path.mkdir()
            rows = [dict(scenario_id=sid, scenario_seed=100+i, checkpoint="episode_0100.pt", max_steps=400,
                         manifest_sha256="a"*64, evaluation_mode=mode, found=bool(found), success=bool(success))
                    for i, (sid, (found, success)) in enumerate(zip("abcde", flags))]
            manifest = dict(scenario_seed=51729, manifest_sha256="a"*64,
                            scenarios=[dict(scenario_id=r["scenario_id"], scenario_seed=r["scenario_seed"]) for r in rows])
            self.rows[path], self.manifests[path] = rows, manifest
            self.save(path)
            write_json(path/"evaluation_summary.json", {})
            for name in analysis.OPTIONAL:
                (path/name).write_text("", encoding="utf-8")

    def save(self, path):
        write_csv(path/"episode_evaluation.csv", self.rows[path])
        write_json(path/"evaluation_manifest.json", self.manifests[path])

    def run_analysis(self):
        return analysis.analyze(self.full, self.off, self.output)

    def table(self, name):
        return analysis.read_csv(self.output/name)

    def test_scenario_mismatch_rejected_before_output(self):
        self.rows[self.off].pop()
        self.manifests[self.off]["scenarios"].pop()
        self.save(self.off)
        with self.assertRaisesRegex(ValueError, "scenario_id"):
            self.run_analysis()
        self.assertFalse(self.output.exists())

    def test_all_manifest_identity_mismatches_rejected(self):
        for key, value in (("scenario_seed", 12), ("manifest_sha256", "b"*64),
                           ("checkpoint", "another.pt"), ("max_steps", 399)):
            with self.subTest(key=key):
                original = dict(self.manifests[self.off])
                self.manifests[self.off][key] = value
                self.save(self.off)
                with self.assertRaises(ValueError):
                    self.run_analysis()
                self.manifests[self.off] = original
        self.save(self.off)
        self.manifests[self.off]["scenarios"][0]["scenario_seed"] = 987
        self.save(self.off)
        with self.assertRaisesRegex(ValueError, "scenario_seed"):
            self.run_analysis()

    def test_transition_classification_and_effect_denominators(self):
        summary = self.run_analysis()
        transitions = {r["scenario_id"]: r["transition_type"] for r in self.table("outcome_groups.csv")}
        self.assertEqual(transitions, dict(a="full_success_searcher_off_fail", b="full_fail_searcher_off_success",
                                          c="both_success", d="both_fail", e="full_fail_searcher_off_success"))
        self.assertEqual([r["scenario_id"] for r in self.table("success_transition_cases.csv")], ["a", "b", "e"])
        self.assertEqual(summary["success_gain"], 1)
        self.assertEqual(summary["found_gain"], 1)
        self.assertEqual(summary["searcher_residual_effect"]["percentage_point_change"], 20)
        self.assertEqual(summary["searcher_residual_effect"]["relative_change"], .5)
        self.assertEqual(summary["paired_transitions"], dict(R0_success_R1_fail=1, R0_fail_R1_success=2))
        self.assertEqual(summary["paired_metrics_by_transition"]["full_fail_searcher_off_success"]["found_step_difference"]["missing_pairs"], 2)
        self.assertEqual(len(list(self.output.iterdir())), 11)

    def test_missing_optional_fields_are_NA_not_zero(self):
        summary = self.run_analysis()
        for row in self.table("found_state_comparison.csv"):
            self.assertEqual(row["executor_target_distance"], "NA")
            self.assertEqual(row["found_step"], "NA")
        for row in self.table("searcher_trajectory_comparison.csv"):
            self.assertEqual(row["agent_id"], "NA")
            self.assertEqual(row["distance_travelled"], "NA")
        self.assertIsNone(summary["mean_metrics"]["found_step_difference"])
        self.assertEqual(summary["paired_metric_coverage"]["found_step_difference"]["missing_pairs"], 5)

    def test_empty_collision_recovery_files_and_aggregate_not_broadcast(self):
        (self.full/"search_collision_recovery_episode.csv").write_text("scenario_id,search_recovery_entry_count\n", encoding="utf-8")
        write_csv(self.full/"search_collision_recovery_summary.csv", [dict(search_recovery_entry_count=999)])
        self.run_analysis()
        self.assertEqual(len(self.table("collision_comparison.csv")), 10)
        for row in self.table("recovery_comparison.csv"):
            self.assertEqual(row["search_recovery_entry_count"], "NA")

    def test_native_wide_fields_timing_and_complete_pair_means(self):
        for path, steps, collision, recovery, distance in (
            (self.full, [10, 20, 15, None, None], 4, 2, 5),
            (self.off, [20, 30, 15, None, 40], 1, 1, 3),
        ):
            for row, step in zip(self.rows[path], steps):
                row.update(found_step=step, executor_final_distance_to_target=distance,
                           searcher_collision_count_pre_found=collision,
                           search_recovery_entry_count=recovery, recovery_duration_mean=2,
                           route_refresh_attempt_count=2, route_refresh_success_count=1,
                           egress_attempt_count=3, egress_success_count=2,
                           searcher_distance_travelled_pre_found=60, map_known_fraction_gain_pre_found=.2,
                           searcher_distance_travelled_pre_found_agent_0=10,
                           searcher_distance_travelled_pre_found_agent_1=20,
                           searcher_distance_travelled_pre_found_agent_2=30)
            self.save(path)
            write_csv(path/"search_continuity_episode.csv", self.rows[path])
        summary = self.run_analysis()
        self.assertAlmostEqual(summary["mean_metrics"]["found_step_difference"], 20/3)
        self.assertEqual(summary["mean_metrics"]["executor_distance_difference"], -2)
        self.assertEqual(summary["mean_metrics"]["collision_difference"], -3)
        self.assertEqual(summary["mean_metrics"]["recovery_difference"], -1)
        states = {(r["scenario_id"], r["variant"]): r for r in self.table("found_state_comparison.csv")}
        self.assertEqual(float(states["a", "full_prrac"]["remaining_steps"]), 390)
        self.assertEqual(states["a", "full_prrac"]["executor_target_distance"], "NA")
        self.assertEqual(states["d", "full_prrac"]["known_map_fraction_gain"], "NA")
        trajectory = self.table("searcher_trajectory_comparison.csv")
        self.assertEqual(len(trajectory), 30)
        self.assertEqual([float(r["distance_travelled"]) for r in trajectory[:3]], [10, 20, 30])
        self.assertTrue(all(r["waypoint_switch_count"] == "NA" for r in trajectory))

    def test_explicit_at_found_fields_and_long_agent_grain(self):
        self.rows[self.full][0].update(found_step=5, executor_target_distance_at_found=12,
                                      executor_wait_distance_at_found=8, searcher_target_distance_at_found=1)
        self.save(self.full)
        write_csv(self.full/"search_continuity_episode.csv", [dict(scenario_id="a", agent_id=1, distance_travelled=7,
                                                                 waypoint_switch_count=3, guidance_change_count=2)])
        self.run_analysis()
        state = self.table("found_state_comparison.csv")[0]
        self.assertEqual(float(state["executor_target_distance"]), 12)
        trajectory = self.table("searcher_trajectory_comparison.csv")[0]
        self.assertEqual(trajectory["agent_id"], "1")
        self.assertEqual(float(trajectory["distance_travelled"]), 7)

    def test_duplicate_scenario_and_mixed_variant_rejected(self):
        self.rows[self.full].append(dict(self.rows[self.full][0]))
        self.save(self.full)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.run_analysis()
        self.rows[self.full].pop()
        self.rows[self.full][0]["evaluation_mode"] = "searcher_residual_off"
        self.save(self.full)
        with self.assertRaisesRegex(ValueError, "evaluation_mode"):
            self.run_analysis()

    def test_sidecar_conflicts_and_unknown_keys_rejected(self):
        for sidecar in ([dict(scenario_id="a", scenario_seed=999)], [dict(scenario_id="unknown")],
                        [dict(scenario_id="a", found=False)]):
            with self.subTest(sidecar=sidecar):
                write_csv(self.full/"search_continuity_episode.csv", sidecar)
                with self.assertRaises(ValueError):
                    self.run_analysis()

    def test_missing_identity_cannot_be_inferred(self):
        for row in self.rows[self.full]:
            row.pop("max_steps")
        self.save(self.full)
        with self.assertRaisesRegex(ValueError, "missing max_steps"):
            self.run_analysis()

    def test_negative_or_fractional_counts_rejected(self):
        for value in (-1, .5, "inf"):
            with self.subTest(value=value):
                self.rows[self.full][0]["search_recovery_entry_count"] = value
                self.save(self.full)
                with self.assertRaises(ValueError):
                    self.run_analysis()
                self.assertFalse(self.output.exists())

    def test_mixed_collision_scopes_are_not_paired(self):
        for row in self.rows[self.full]:
            row["searcher_collision_count_pre_found"] = 2
        for row in self.rows[self.off]:
            row["collision_count"] = 4
        self.save(self.full)
        self.save(self.off)
        result = self.run_analysis()
        self.assertIsNone(result["mean_metrics"]["collision_difference"])
        self.assertEqual(result["paired_metric_coverage"]["collision_difference"]["valid_pairs"], 0)

    def test_manifest_explicit_fields_and_read_only_inputs(self):
        for path in (self.full, self.off):
            self.manifests[path].update(checkpoint="episode_0100.pt", max_steps=400)
            self.save(path)
        before = {p: analysis.file_hash(p) for directory in (self.full, self.off) for p in directory.iterdir()}
        self.run_analysis()
        self.assertEqual(before, {p: analysis.file_hash(p) for p in before})
        manifest = json.loads((self.output/"analysis_manifest.json").read_text(encoding="utf-8"))
        self.assertIn("git_commit", manifest)
        self.assertIn("timestamp", manifest)
        self.assertIn("evaluation_manifest.json", manifest["provenance"]["full_prrac"]["identity_sources"]["checkpoint"])
        with self.assertRaises(FileExistsError):
            self.run_analysis()
        with self.assertRaises(ValueError):
            analysis.analyze(self.full, self.off, self.full/"new")

    def test_zero_baseline_relative_change_is_null(self):
        for row in self.rows[self.full]:
            row["success"] = False
        self.save(self.full)
        result = self.run_analysis()["searcher_residual_effect"]
        self.assertIsNone(result["relative_change"])
        self.assertEqual(result["relative_change_reason"], "baseline_count_zero")

    def test_cli_success_default_output_and_failure_exit(self):
        command = [sys.executable, str(Path(analysis.__file__).resolve()), "--full-output", str(self.full),
                   "--searcher-off-output", str(self.off)]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root/"searcher_residual_analysis"/"outcome_groups.csv").is_file())
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("analysis failed", result.stderr)

    def json_output(self, name):
        return json.loads((self.output/name).read_text(encoding="utf-8"))

    def test_found_distance_never_falls_back_to_final_distance(self):
        for path in (self.full, self.off):
            self.rows[path][0]["executor_final_distance_to_target"] = 5
            self.save(path)
        result = self.run_analysis()
        for row in self.table("found_execution_transition_comparison.csv")[:2]:
            self.assertEqual(row["executor_distance_at_found"], "NA")
            self.assertEqual(row["executor_distance_at_handoff"], "NA")
            self.assertEqual(float(row["final_executor_target_distance"]), 5)
        self.assertIsNone(result["mean_metrics"]["executor_distance_at_found_difference"])
        self.assertEqual(result["mean_metrics"]["executor_distance_difference"], 0)
        self.assertIn("deprecated / final-state only", result["definitions"]["executor_distance_difference"])

    def test_explicit_found_distance_positive_R1_minus_R0(self):
        self.rows[self.full][0]["executor_distance_to_target_at_found"] = 10
        self.rows[self.off][0]["executor_target_distance_at_found"] = 14
        self.save(self.full)
        self.save(self.off)
        result = self.run_analysis()
        self.assertEqual(result["mean_metrics"]["executor_distance_at_found_difference"], 4)
        self.assertEqual(result["paired_metric_coverage"]["executor_distance_at_found_difference"]["valid_pairs"], 1)
        sources = self.json_output("analysis_manifest.json")["field_sources"]
        self.assertEqual(sources["full_prrac"]["a"]["executor_distance_at_found"], "executor_distance_to_target_at_found")
        self.assertEqual(sources["searcher_residual_off"]["a"]["executor_distance_at_found"], "executor_target_distance_at_found")

    def test_handoff_contact_and_success_timing(self):
        for path, delta in ((self.full, 0), (self.off, 4)):
            # c is both-success, so all timing endpoints are eligible.
            self.rows[path][2].update(contact_episode=True, found_step=20, executor_distance_at_handoff=7+delta,
                                      handoff_delay=3+delta, found_to_first_contact_steps=10+delta,
                                      found_to_success_steps=30+delta)
            self.save(path)
        result = self.run_analysis()
        for field in ("executor_distance_at_handoff", "handoff_delay", "found_to_first_contact_steps", "found_to_success_steps"):
            self.assertEqual(result["mean_metrics"][field+"_difference"], 4)
        self.assertEqual(result["paired_metric_coverage"]["handoff_delay_difference"]["valid_pairs"], 1)
        rows = {r["variant"]: r for r in self.table("found_execution_transition_comparison.csv") if r["scenario_id"] == "c"}
        self.assertEqual(rows["full_prrac"]["contact"], "True")
        self.assertEqual(float(rows["searcher_residual_off"]["remaining_steps"]), 380)

    def test_exact_mcnemar_five_vs_ten(self):
        self.assertAlmostEqual(analysis.exact_mcnemar(5, 10), .3017578125)
        self.assertEqual(analysis.exact_mcnemar(5, 10), analysis.exact_mcnemar(10, 5))
        pairs = [(True, False)]*5 + [(False, True)]*10 + [(True, True)]*27 + [(False, False)]*58
        stats = analysis.paired_binary_stats(pairs)
        self.assertEqual((stats["r0_positive"], stats["r1_positive"]), (32, 37))
        self.assertEqual(stats["discordant_total"], 15)
        self.assertEqual(stats["percentage_point_change"], 5)
        self.assertAlmostEqual(stats["exact_mcnemar_p_two_sided"], .3017578125)

    def test_exact_mcnemar_zero_vs_six(self):
        self.assertEqual(analysis.exact_mcnemar(0, 6), .03125)

    def test_exact_mcnemar_no_discordance(self):
        self.assertEqual(analysis.exact_mcnemar(0, 0), 1.0)
        result = analysis.paired_binary_stats([(False, False), (True, True)])
        self.assertTrue(result["available"])
        self.assertEqual(result["exact_mcnemar_p_two_sided"], 1.0)

    def test_missing_contact_is_unavailable_not_false(self):
        self.run_analysis()
        result = self.json_output("paired_binary_significance.json")["contact"]
        self.assertFalse(result["available"])
        self.assertEqual(result["valid_pairs"], 0)
        self.assertEqual(result["missing_pairs"], 5)
        self.assertIsNone(result["both_negative"])
        self.assertIsNone(result["exact_mcnemar_p_two_sided"])
        self.assertTrue(all(r["contact"] == "NA" for r in self.table("found_execution_transition_comparison.csv")))

    def test_missing_residual_is_not_zero(self):
        result = self.run_analysis()
        for row in self.table("searcher_residual_diagnostics.csv"):
            self.assertTrue(all(row[name] == "NA" for name in analysis.RESIDUAL_METRICS))
        group = result["full_prrac_residual_by_transition"]["full_fail_searcher_off_success"]
        self.assertEqual(group["count"], 2)
        self.assertEqual(group["raw_residual_norm_mean_pre_found"], dict(mean=None, valid=0, missing=2))

    def test_residual_off_retains_raw_value_and_explicit_zero(self):
        self.rows[self.off][0].update(searcher_raw_residual_norm_mean_pre_found=.8,
                                      searcher_applied_residual_norm_mean_pre_found=0)
        self.save(self.off)
        self.run_analysis()
        row = next(r for r in self.table("searcher_residual_diagnostics.csv") if r["scenario_id"] == "a" and r["variant"] == "searcher_residual_off")
        self.assertEqual(float(row["raw_residual_norm_mean_pre_found"]), .8)
        self.assertEqual(float(row["applied_residual_norm_mean_pre_found"]), 0)

    def test_transition_groups_use_only_full_residual(self):
        for row, value in zip(self.rows[self.full], (.2, .6, .1, .3, 1.0)):
            row.update(searcher_raw_residual_norm_mean_pre_found=value,
                       searcher_applied_residual_norm_mean_pre_found=value,
                       searcher_residual_contribution_ratio_mean_pre_found=value,
                       searcher_residual_negative_alignment_rate_pre_found=value,
                       searcher_waypoint_switch_count_pre_found=2,
                       searcher_route_active_rate_pre_found=.5)
        for row in self.rows[self.off]:
            row.update(searcher_raw_residual_norm_mean_pre_found=99, searcher_applied_residual_norm_mean_pre_found=0)
        self.save(self.full)
        self.save(self.off)
        groups = self.run_analysis()["full_prrac_residual_by_transition"]
        self.assertEqual(groups["full_success_searcher_off_fail"]["raw_residual_norm_mean_pre_found"], dict(mean=.2, valid=1, missing=0))
        self.assertEqual(groups["full_fail_searcher_off_success"]["raw_residual_norm_mean_pre_found"], dict(mean=.8, valid=2, missing=0))
        self.assertEqual(groups["full_fail_searcher_off_success"]["applied_residual_norm_mean_pre_found"]["mean"], .8)

    def test_core_field_coverage_counts_eligible_scenarios(self):
        for path in (self.full, self.off):
            self.rows[path][0].update(found_step=10, executor_distance_to_target_at_found=10,
                                      searcher_raw_residual_norm_mean_pre_found=.4)
            # d is not found: its stale at_found field must not enter coverage.
            self.rows[path][3]["executor_distance_to_target_at_found"] = 99
        self.rows[self.full][1]["executor_distance_to_target_at_found"] = 8
        self.save(self.full)
        self.save(self.off)
        self.run_analysis()
        coverage = self.json_output("analysis_manifest.json")["field_coverage"]
        self.assertEqual(coverage["executor_distance_at_found"], dict(full_prrac=2, searcher_residual_off=1, paired_valid=1))
        self.assertEqual(coverage["raw_residual_norm_mean_pre_found"], dict(full_prrac=1, searcher_residual_off=1, paired_valid=1))
        self.assertEqual(coverage["executor_distance_at_handoff"]["paired_valid"], 0)

    def test_existing_output_headers_are_preserved(self):
        self.run_analysis()
        expected = {
            "outcome_groups.csv": "scenario_id full_found searcher_off_found full_success searcher_off_success transition_type",
            "success_transition_cases.csv": "scenario_id transition_type full_found_step searcher_off_found_step found_step_difference full_success searcher_off_success full_final_distance searcher_off_final_distance",
            "found_state_comparison.csv": "scenario_id variant found_step remaining_steps searcher_target_distance executor_target_distance executor_wait_distance searcher_distance_travelled known_map_fraction_gain",
            "collision_comparison.csv": "scenario_id variant collision_episode collision_count max_collision_streak recovery_entry_count route_refresh_attempt_count egress_attempt_count",
            "recovery_comparison.csv": "scenario_id variant search_recovery_entry_count route_refresh_success_count egress_success_count mean_recovery_duration",
            "searcher_trajectory_comparison.csv": "scenario_id variant agent_id distance_travelled waypoint_switch_count guidance_change_count",
        }
        for name, columns in expected.items():
            with self.subTest(name=name), (self.output/name).open(encoding="utf-8", newline="") as handle:
                self.assertEqual(next(csv.reader(handle)), columns.split())
        self.assertTrue((self.output/"searcher_residual_effect_summary.json").is_file())
        self.assertTrue((self.output/"analysis_manifest.json").is_file())

    def test_execution_groups_separate_variants_and_unreached_stages(self):
        for path in (self.full, self.off):
            for row in self.rows[path]:
                row.update(found_step=10, executor_distance_at_handoff=9, handoff_delay=2,
                           found_to_first_contact_steps=3, found_to_success_steps=4,
                           contact_episode=False, searcher_collision_count_pre_found=1,
                           search_recovery_entry_count=2)
            self.save(path)
        result = self.run_analysis()
        group = result["execution_chain_by_transition"]["full_fail_searcher_off_success"]
        # B contains b (both found) and e (R0 not found, R1 found).
        self.assertEqual(group["full_prrac"]["found_step"], dict(mean=10, valid=1, missing=1))
        self.assertEqual(group["searcher_residual_off"]["found_step"], dict(mean=10, valid=2, missing=0))
        self.assertEqual(group["full_prrac"]["found_to_success_steps"]["valid"], 0)
        self.assertEqual(group["searcher_residual_off"]["found_to_first_contact_steps"]["valid"], 0)
        self.assertEqual(group["full_prrac"]["collision_count"]["valid"], 2)

    def test_partial_contact_pairs_use_complete_pair_denominator(self):
        result = analysis.paired_binary_stats([(True, False), (False, True), (False, True), (None, True), (True, None)])
        self.assertEqual(result["valid_pairs"], 3)
        self.assertEqual(result["missing_pairs"], 2)
        self.assertEqual(result["absolute_count_change"], 1)
        self.assertAlmostEqual(result["percentage_point_change"], 100/3)
        self.assertEqual(result["both_negative"], 0)

    def test_all_explicit_contact_aliases_and_conflicts(self):
        for key in ("contact", "reached_contact", "contact_reached", "has_contact", "contact_episode"):
            with self.subTest(key=key):
                self.assertTrue(analysis.metric({key: "True"}, "contact"))
                self.assertFalse(analysis.metric({key: "False"}, "contact"))
        with self.assertRaisesRegex(ValueError, "conflicting.*contact"):
            analysis.metric(dict(contact=True, contact_episode=False), "contact")

    def test_timing_does_not_infer_contact_and_aggregate_not_broadcast(self):
        self.rows[self.full][0]["found_to_first_contact_steps"] = 5
        self.save(self.full)
        write_csv(self.full/"search_collision_recovery_summary.csv", [dict(searcher_raw_residual_norm_mean_pre_found=.9,
                                                                         executor_distance_to_target_at_found=10)])
        self.run_analysis()
        contact = self.json_output("paired_binary_significance.json")["contact"]
        self.assertFalse(contact["available"])
        row = self.table("found_execution_transition_comparison.csv")[0]
        self.assertEqual(float(row["found_to_first_contact_steps"]), 5)
        self.assertEqual(row["contact"], "NA")
        self.assertEqual(row["executor_distance_at_found"], "NA")
        self.assertEqual(self.table("searcher_residual_diagnostics.csv")[0]["raw_residual_norm_mean_pre_found"], "NA")


if __name__ == "__main__":
    unittest.main()
