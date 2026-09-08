"""Synthetic episode fixtures; no model, simulator or historical result mutation."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from scripts import analyze_found_timing_failure_attribution as a
from tests import test_found_executor_distance_analysis as fixtures


def row(index=0, *, found=True, success=False, step=50):
    result = fixtures.base_row(index, found, success)
    result.update(found_step=step if found else "", episode_length=400,
                  executor_target_received_step=step+1 if found else "",
                  first_contact_step=min(400, step+5) if success else "")
    return result


class EpisodeTests(unittest.TestCase):
    def test_found_filtering_and_three_population_percentages(self):
        episodes = a.build_episodes([row(0, found=False), row(1), row(2, success=True), row(3, found=False)])
        counts = a.population_counts(episodes)
        self.assertEqual([counts[k]["count"] for k in counts], [2, 1, 1])
        self.assertEqual([counts[k]["percentage"] for k in counts], [50, 25, 25])
        self.assertEqual(sum(r["found"] for r in episodes), 2)

    def test_not_found_is_na_never_zero_or_final_step(self):
        raw = dict(row(found=False), found_step=200, remaining_steps_after_found=200)
        result = a.build_episodes([raw])[0]
        self.assertIsNone(result["found_step"])
        self.assertIsNone(result["remaining_steps"])
        self.assertIsNone(result["executor_distance_at_found"])

    def test_remaining_steps_exact_and_contradictions_rejected(self):
        result = a.build_episodes([row(step=399)])[0]
        self.assertEqual(result["remaining_steps"], 1)
        result = a.build_episodes([row(step=400)])[0]
        self.assertEqual(result["remaining_steps"], 0)
        for mutation in (dict(remaining_steps_after_found=11), dict(episode_length=20), dict(max_steps=500)):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                a.build_episodes([dict(row(), **mutation)])

    def test_fixed_bin_boundaries_and_empty_bin_na(self):
        for step, expected in ((0, "0-100"), (1, "0-100"), (100, "0-100"), (101, "101-200"),
                               (200, "101-200"), (201, "201-300"), (300, "201-300"), (301, "301-400"), (400, "301-400")):
            self.assertEqual(a.timing_bin(step), expected)
        for step in (-1, 401, 100.1):
            with self.assertRaises(ValueError):
                a.timing_bin(step)
        episodes = a.build_episodes([row(0, step=100, success=True), row(1, step=101), row(2, found=False)])
        bins = a.timing_bins(episodes)
        self.assertEqual([b["found_count"] for b in bins], [1, 1, 0, 0])
        self.assertEqual([b["success_rate"] for b in bins], [1, 0, None, None])

    def test_episode_unit_duplicate_steps_do_not_create_samples(self):
        episodes = a.build_episodes([row(0, step=50), row(1, step=50)])
        self.assertEqual(a.timing_bins(episodes)[0]["found_count"], 2)
        with self.assertRaises(ValueError):
            a.build_episodes([row(0), row(0)])
        with self.assertRaises(ValueError):
            a.build_episodes([row(0), dict(row(1), episode_index=0)])

    def test_logistic_input_predictor_order_and_no_not_found(self):
        episodes = a.build_episodes([row(0, found=False), row(1, step=20), row(2, success=True, step=30)])
        found = [r for r in episodes if r["found"]]
        inputs = a.model_inputs(found, True, True)
        np.testing.assert_equal(inputs["found_step_model"][0], [[20], [30]])
        np.testing.assert_equal(inputs["distance_model"][0], [[4], [5]])
        np.testing.assert_equal(inputs["combined_model"][0], [[20, 4], [30, 5]])
        self.assertEqual(inputs["combined_model"][1], ["found_step", "distance"])

    def test_missing_geometry_stops_whole_geometry_population(self):
        raw = dict(row(), executor_distance_to_target_at_found="", executor_final_distance_to_target=1)
        found = a.build_episodes([raw, row(1, success=True)])
        self.assertIsNone(found[0]["executor_distance_at_found"])
        models = a.model_inputs(found, True, False)
        self.assertEqual(list(models), ["found_step_model"])

    def test_contact_budget_censoring_is_not_contact_duration(self):
        found = a.build_episodes([row(0, step=10, success=True), row(1, step=350)])
        summary = a.contact_budget(found)
        self.assertEqual(summary["observed_found_to_contact_steps"]["n"], 1)
        self.assertEqual(summary["observed_found_to_contact_steps"]["mean"], 5)
        self.assertEqual(summary["no_contact_at_horizon_count"], 1)
        self.assertEqual(summary["no_contact_observed_exposure_steps"]["mean"], 50)
        self.assertEqual(summary["found_without_contact_count"], 1)

    def test_found_step_stats_mean_median_and_bootstrap_reproduce(self):
        found = a.build_episodes([row(0, step=10, success=True), row(1, step=100),
                                 row(2, step=30, success=True), row(3, step=300)])
        first = a.estimate(found, seed=42, repetitions=30, timing_complete=True, geometry_complete=True)
        second = a.estimate(found, seed=42, repetitions=30, timing_complete=True, geometry_complete=True)
        self.assertEqual(first, second)
        timing = first[0]["found_step"]
        self.assertEqual(timing["success_mean"], 20)
        self.assertEqual(timing["failure_mean"], 200)
        self.assertEqual(timing["difference"], 180)
        self.assertEqual(timing["median_difference"]["estimate"], 180)
        self.assertEqual(timing["success"]["n"], 2)
        for key in ("mean_difference", "median_difference"):
            self.assertEqual(timing[key]["valid_repetitions"]+timing[key]["invalid_repetitions"], 30)


class DecisionTests(unittest.TestCase):
    @staticmethod
    def models(timing, distance):
        def term(supported):
            ci = [-.2, -.1] if supported else [-.2, .1]
            return dict(beta_bootstrap=dict(valid_repetitions=5000, ci95=ci), beta_wald_ci95=ci)
        return dict(found_step_model=dict(status="ok", bootstrap_fit_status_counts=dict(ok=5000), terms=dict(found_step=term(timing))),
                    distance_model=dict(status="ok", bootstrap_fit_status_counts=dict(ok=5000), terms=dict(distance=term(distance))),
                    combined_model=dict(status="ok", bootstrap_fit_status_counts=dict(ok=5000), terms=dict(found_step=term(timing), distance=term(distance))))

    def test_all_four_predeclared_decisions(self):
        for timing, distance, expected in (
            (True, False, "Evidence favors early discovery limitation"),
            (False, True, "Evidence favors found-state geometry limitation"),
            (True, True, "Both discovery timing and handoff geometry contribute"),
            (False, False, "No dominant factor identified"),
        ):
            result = a.dominant_factor(self.models(timing, distance), 5000)
            self.assertEqual(result["conclusion"], expected)
            self.assertEqual(result["interpretation_type"], "descriptive")

    def test_missing_data_instability_or_adjusted_disagreement_prevents_selection(self):
        models = self.models(True, False)
        models["combined_model"]["terms"]["found_step"]["beta_bootstrap"]["ci95"] = [-.1, .1]
        self.assertIsNone(a.dominant_factor(models, 5000)["associated_factor"])
        models = self.models(True, False)
        models["distance_model"]["status"] = "not_fitted_missing_required_predictor"
        self.assertFalse(a.dominant_factor(models, 5000)["comparison_assessable"])
        models = self.models(True, False)
        models["found_step_model"]["terms"]["found_step"]["beta_bootstrap"]["valid_repetitions"] = 2000
        self.assertIsNone(a.dominant_factor(models, 5000)["associated_factor"])
        models = self.models(True, False)
        models["distance_model"]["bootstrap_fit_status_counts"] = dict(ok=2000, separation=3000)
        self.assertFalse(a.dominant_factor(models, 5000)["comparison_assessable"])
        self.assertFalse(a.dominant_factor(self.models(True, False), 50)["comparison_assessable"])


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.output = self.root/"source", self.root/"analysis"
        self.source.mkdir()
        self.rows = []
        steps = [10, 80, 120, 180, 200, 250, 301, 350, 390, 400]
        for i in range(12):
            sample = row(i, found=i<10, success=i in (0, 1, 3, 5, 8), step=steps[i] if i<10 else 50)
            sample.update(report_schema=a.geometry.REPORT_SCHEMA, evaluation_runtime_revision=a.geometry.REVISION,
                          execution_variant="B1_ATOMIC_LAST_VALID", search_recovery_variant="S2A1_C2_LOCAL_CONNECTOR",
                          runtime_integration_mode="native", manifest_sha256="fixture-identity", checkpoint="fixture.pt",
                          checkpoint_episode=100, checkpoint_config_hash="fixture-hash", checkpoint_runtime_revision=a.geometry.REVISION,
                          diagnostic_only=False, privileged_oracle=False, explore=False, training_update=False,
                          searcher_residual_off_enabled=False, search_value_guidance=json.dumps(dict(enabled=False)))
            self.rows.append(sample)
        self.manifest = dict(evaluation_episodes=12, scenario_seed=1729, manifest_sha256="fixture-identity",
                             scenarios=[dict(scenario_id=r["scenario_id"], scenario_seed=r["scenario_seed"], max_steps=400) for r in self.rows])
        self.config = dict(evaluation_episodes=12, scenario_seed=1729, manifest_sha256="fixture-identity", modes=["full_prrac"],
                           resolved_scenario_ids=[r["scenario_id"] for r in self.rows], observation_dim=28, action_dim=3, critic_dim=124,
                           report_schema=a.geometry.REPORT_SCHEMA, evaluation_runtime_revision=a.geometry.REVISION,
                           checkpoint_runtime_revision=a.geometry.REVISION, execution_variant="B1_ATOMIC_LAST_VALID",
                           runtime_integration_mode="native", search_value_guidance=dict(enabled=False),
                           resolved_search_recovery_variants=["S2A1_C2_LOCAL_CONNECTOR"], max_steps=400,
                           resolved_checkpoint_paths=["fixture.pt"])
        self.save()

    def save(self):
        fixtures.save_csv(self.source/"episode_evaluation.csv", self.rows)
        for name, document in (("evaluation_manifest.json", self.manifest), ("resolved_evaluation_config.json", self.config)):
            (self.source/name).write_text(json.dumps(document), encoding="utf-8")

    def analyze(self):
        return a.analyze(self.source, self.output, expected_episodes=12, bootstrap_repetitions=30)

    def test_all_outputs_hashes_input_immutability_and_denominators(self):
        before = {p: p.read_bytes() for p in self.source.iterdir()}
        summary = self.analyze()
        self.assertEqual(before, {p: p.read_bytes() for p in self.source.iterdir()})
        self.assertEqual(summary["total_episodes"], 12)
        self.assertAlmostEqual(summary["not_found_ratio"], 2/12)
        self.assertAlmostEqual(summary["found_failure_ratio"], 5/12)
        self.assertAlmostEqual(summary["failure_given_found_ratio"], .5)
        self.assertAlmostEqual(summary["not_found_share_of_all_failures"], 2/7)
        timing = a.geometry.read_csv(self.output/"found_timing_episode.csv")
        self.assertEqual(len(timing), 12)
        self.assertEqual(timing[-1]["found_step"], "NA")
        self.assertEqual(timing[-1]["remaining_steps"], "NA")
        self.assertEqual(len(a.geometry.read_csv(self.output/"found_geometry_episode.csv")), 10)
        manifest = json.loads((self.output/"analysis_manifest.json").read_text())
        self.assertEqual(len(manifest["output_sha256"]), 5)
        for name, digest in manifest["output_sha256"].items():
            self.assertEqual(a.geometry.sha256(self.output/name), digest)
        for item in manifest["inputs"]:
            self.assertEqual(a.geometry.sha256(item["path"]), item["sha256"])
        probabilities = a.geometry.read_csv(self.output/"success_probability_summary.csv")
        for dimension in ("found_step_bin", "found_geometry_quartile"):
            self.assertEqual(sum(int(r["found_count"]) for r in probabilities if r["dimension"] == dimension), 10)

    def test_missing_distance_keeps_all_rows_and_stops_geometry_only(self):
        self.rows[0][a.geometry.DISTANCE_FIELD] = "NA"
        self.rows[1][a.geometry.DISTANCE_FIELD] = ""
        self.save()
        summary = self.analyze()
        self.assertEqual(summary["status"], "partial_missing_required_data")
        self.assertEqual(summary["missingness"]["executor_distance_at_found"]["count"], 2)
        self.assertEqual(summary["missingness"]["executor_distance_at_found"]["scenario_ids"], ["scenario_0", "scenario_1"])
        self.assertEqual(summary["found_step"]["status"], "complete")
        self.assertEqual(summary["distance"]["status"], "stopped_missing_distance")
        self.assertNotEqual(summary["logistic"]["combined_model"]["status"], "ok")
        self.assertEqual(len(a.geometry.read_csv(self.output/"found_geometry_episode.csv")), 10)
        self.assertEqual(summary["geometry_quartiles"], [])

    def test_missing_found_step_does_not_drop_episode_from_geometry(self):
        self.rows[0]["found_step"] = ""
        self.save()
        summary = self.analyze()
        self.assertEqual(summary["found_step"]["status"], "stopped_missing_found_step")
        self.assertEqual(summary["distance"]["success"]["n"], 5)
        self.assertEqual(summary["timing_bins"], [])

    def test_seed_checkpoint_mode_and_manifest_identity_rejected(self):
        original = copy.deepcopy(self.rows)
        for key, value in (("scenario_seed", 999), ("checkpoint", "wrong.pt"), ("checkpoint_config_hash", "wrong"),
                           ("evaluation_mode", "searcher_residual_off"), ("execution_variant", "wrong"),
                           ("search_recovery_variant", "wrong")):
            self.rows = copy.deepcopy(original)
            self.rows[0][key] = value
            self.save()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.analyze()
            self.assertFalse(self.output.exists())
        self.rows = original
        self.config["scenario_seed"] = 99
        self.save()
        with self.assertRaises(ValueError):
            self.analyze()

    def test_checkpoint_metadata_identity_validation(self):
        metadata = [dict(checkpoint="fixture.pt", completed_episode=100,
                         metadata=dict(config_hash="fixture-hash", execution_runtime_revision=a.geometry.REVISION,
                                       observation_dim=28, action_dim=3, critic_dim=124))]
        path = self.source/"checkpoint_metadata.json"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        self.assertTrue(a.audit(self.source, 12)[2]["checkpoint_metadata_verified"])
        metadata[0]["metadata"]["config_hash"] = "wrong"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.analyze()

    def test_no_found_or_single_outcome_does_not_force_factor(self):
        for raw in self.rows:
            raw.update(found=False, success=False, contact_episode=False)
        self.save()
        summary = self.analyze()
        self.assertEqual(summary["found_count"], 0)
        self.assertEqual(summary["not_found_ratio"], 1)
        self.assertIsNone(summary["dominant_failure_factor"]["associated_factor"])

    def test_refuse_output_overwrite_or_historical_write(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.analyze()
        with self.assertRaises(ValueError):
            a.analyze(self.source, self.source/"results")

    def test_direct_cli_and_expected_episode_gate(self):
        command = [sys.executable, str(Path(a.__file__)), "--source-dir", str(self.source), "--output-dir", str(self.output),
                   "--expected-episodes", "12", "--bootstrap-repetitions", "10"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.output/"success_probability_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
