"""Offline fixtures and source-contract checks; no simulator/checkpoint imports."""
import ast
import copy
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from scripts import analyze_found_executor_distance as a


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)


def base_row(index=0, found=True, success=False):
    return dict(scenario_id=f"scenario_{index}", scenario_seed=1729+index, episode_index=index,
                evaluation_mode="full_prrac", found=found, success=success, contact_episode=success,
                found_step=10 if found else "", max_steps=400,
                executor_distance_to_target_at_found=3+index if found else "",
                executor_target_received_step=11 if found else "", handoff_delay=1 if found else "",
                first_contact_step=15 if success else "")


class PopulationTests(unittest.TestCase):
    def test_found_and_full_only_filtering_and_grouping(self):
        rows = [base_row(0, success=True), base_row(1), base_row(2, found=False),
                dict(base_row(3), evaluation_mode="searcher_residual_off")]
        population = a.prepare_population(rows)
        self.assertEqual([r["scenario_id"] for r in population], ["scenario_0", "scenario_1"])
        self.assertEqual([r["success"] for r in population], [True, False])

    def test_3d_xy_and_dz(self):
        self.assertEqual(a.geometry([4, 6, 14], [1, 2, 2]), (13, 5, 12))

    def test_invalid_geometry(self):
        for vector in ([1, 2], [1, 2, float("nan")]):
            with self.assertRaises(ValueError):
                a.geometry(vector, [0, 0, 0])

    def test_missing_primary_preserved_no_final_or_alias_fallback(self):
        row = dict(base_row(), executor_distance_to_target_at_found="", executor_final_distance_to_target=2,
                   executor_distance_at_found=3)
        result = a.prepare_population([row])
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0][a.X_FIELD])

    def test_scalar_does_not_fabricate_coordinates_xy_dz_speed(self):
        row = a.prepare_population([base_row()])[0]
        for key in ("executor_x_at_found", "target_x_at_found", "executor_target_distance_xy_at_found",
                    "executor_target_abs_dz_at_found", "target_speed_at_found"):
            self.assertIsNone(row[key])

    def test_timing_semantics_and_receive_are_distinct(self):
        row = a.prepare_population([base_row()])[0]
        self.assertEqual((row["handoff_step"], row["executor_received_target_step"]), (10, 11))
        self.assertEqual((row["handoff_delay_steps"], row["executor_receive_delay"], row["source_handoff_delay"]), (0, 1, 1))
        self.assertTrue(all(v is None for v in a.timing(10, 11, semantics={}).values()))

    def test_conflicting_timing_and_remaining_rejected(self):
        for mutation in (dict(handoff_delay=2), dict(remaining_steps_after_found=391),
                         dict(executor_target_received_step=10), dict(found_step=10.5)):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                a.prepare_population([dict(base_row(), **mutation)])

    def test_contact_equality_and_missing(self):
        population = a.prepare_population([base_row(0), base_row(1, success=True)])
        self.assertTrue(a.contact_equality(population)["equal_on_all_found"])
        population[0]["contact"] = True
        self.assertFalse(a.contact_equality(population)["equal_on_all_found"])
        population[0]["contact"] = None
        self.assertIsNone(a.contact_equality(population)["equal_on_all_found"])

    def test_first_event_latch_uses_actual_diagnostics_source(self):
        # Compile just the read-only event/distance prefix of the EXISTING method.
        # This executes its real first-event logic without loading torch or a runtime.
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root/a.SOURCE_FILES[0]).read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ExecutionEpisodeDiagnostics")
        method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "observe_step"))
        stop = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign) and
                    isinstance(n.targets[0], ast.Name) and n.targets[0].id == "received")
        method.body = method.body[:stop]
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
        scope = {"_runtime": lambda env: env}
        exec(compile(ast.fix_missing_locations(module), "<existing diagnostics event prefix>", "exec"), scope)
        diagnostic = SimpleNamespace(found_step=None, success_step=None, executor_distance_to_target_at_found=None,
                                     _distances=lambda env: (env.distance, None))
        task = lambda step, found: SimpleNamespace(step=step, target_found=found, mission_complete=False)
        observe = scope["observe_step"]
        observe(diagnostic, SimpleNamespace(distance=99), None, task_before=task(8, False), task_after=task(9, False))
        self.assertIsNone(diagnostic.found_step)
        observe(diagnostic, SimpleNamespace(distance=13), None, task_before=task(9, False), task_after=task(10, True))
        observe(diagnostic, SimpleNamespace(distance=2), None, task_before=task(10, True), task_after=task(11, True))
        self.assertEqual((diagnostic.found_step, diagnostic.executor_distance_to_target_at_found), (10, 13))
        distance_method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_distances"))
        distance_method.decorator_list = []
        module.body[-1] = distance_method
        scope.update(np=np, _vector=lambda v: None if v is None else np.asarray(v, dtype=float))
        exec(compile(ast.fix_missing_locations(module), "<existing diagnostics distance>", "exec"), scope)
        env = SimpleNamespace(_agent_pos=[[99]*3]*3+[[4, 6, 14]], target_state=SimpleNamespace(position=[1, 2, 2]))
        self.assertEqual(scope["_distances"](env)[0], 13)

    def test_off_by_one_wrapper_calls_diagnostics_after_physical_step(self):
        root = Path(__file__).resolve().parents[1]
        source = (root/a.SOURCE_FILES[1]).read_text(encoding="utf-8")
        self.assertLess(source.index("observations, base_rewards, dones = self.env.step(actions)"), source.index("task_after = self.get_task_state()"))
        self.assertLess(source.index("task_after = self.get_task_state()"), source.index("self.diagnostics.observe_step("))
        runtime = (root/"core/env/uav_env.py").read_text(encoding="utf-8")
        tree = ast.parse(runtime)
        publisher = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_publish_detection")
        text = ast.unparse(publisher)
        self.assertIn("self.handoff_step = int(self.step_count)", text)
        self.assertIn("self.found_step = int(self.step_count)", text)
        steps = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "step"]
        physical = next(ast.unparse(n) for n in steps if "advance_target_state" in ast.unparse(n))
        self.assertLess(physical.index("self.target_state = advance_target_state"), physical.index("self._maybe_detect_swept"))


class StatisticsTests(unittest.TestCase):
    def test_point_biserial_known_value(self):
        self.assertAlmostEqual(a.correlation([1, 2, 3, 4], [1, 1, 0, 0]), -2/math.sqrt(5))
        self.assertIsNone(a.correlation([2, 2], [0, 1]))

    def test_summary_sample_std_and_grouping(self):
        result = a.describe([1, 2, 3, 4])
        self.assertEqual(result["n"], 4)
        self.assertEqual((result["mean"], result["median"], result["Q1"], result["Q3"]), (2.5, 2.5, 1.75, 3.25))
        self.assertAlmostEqual(result["std"], math.sqrt(5/3))

    def test_logistic_direction_matches_analytic_solution(self):
        x = [0]*4+[1]*4
        fit = a.logistic(x, [1, 1, 1, 0, 1, 0, 0, 0])
        self.assertEqual(fit["status"], "ok")
        self.assertAlmostEqual(fit["beta"][0], -2*math.log(3), places=7)
        self.assertAlmostEqual(fit["intercept"], math.log(3), places=7)

    def test_adjusted_model_matches_known_two_predictor_odds(self):
        matrix, y = [], []
        for distance, step, successes in ((1, 10, 8), (1, 20, 5), (2, 10, 5), (2, 20, 2)):
            matrix.extend([[distance, step]]*10)
            y.extend([1]*successes+[0]*(10-successes))
        fit = a.logistic(matrix, y)
        self.assertEqual(fit["status"], "ok")
        np.testing.assert_allclose(fit["beta"], [-math.log(4), -math.log(4)/10], atol=1e-8)

    def test_separation_and_rank_detection(self):
        self.assertEqual(a.logistic([1, 2, 3, 4], [1, 1, 0, 0])["status"], "complete_or_quasi_separation")
        self.assertEqual(a.logistic([[1, 2], [2, 4], [3, 6], [4, 8]], [1, 0, 1, 0])["status"], "rank_deficient")
        self.assertEqual(a.logistic([1, 1, 1, 1], [0, 1, 0, 1])["status"], "constant_predictor")
        self.assertEqual(a.logistic([1, 2], [1, 1])["status"], "single_outcome")
        separated = a.logistic([[1, 4], [4, 1], [1, 1], [2, 2]], [1, 1, 0, 0])
        self.assertNotEqual(separated["status"], "ok")

    def test_auc_negative_distance_direction_and_ties(self):
        self.assertEqual(a.auc([1, 2, 3, 4], [1, 1, 0, 0]), 1)
        self.assertEqual(a.auc([4, 3, 2, 1], [1, 1, 0, 0]), 0)
        self.assertEqual(a.auc([2, 2, 2, 2], [1, 1, 0, 0]), .5)

    def test_quartile_partition_and_duplicate_boundaries(self):
        bins, _ = a.quartiles(np.arange(16), [1]*4+[0]*12)
        self.assertEqual([b["n"] for b in bins], [4]*4)
        self.assertEqual([b["success_rate"] for b in bins], [1, 0, 0, 0])
        bins, method = a.quartiles([1]*8, [0, 1]*4)
        self.assertEqual(len(bins), 1)
        self.assertEqual(method["edges"], [1])

    def test_bootstrap_reproducible_and_failed_draws_accounted(self):
        args = ([1, 2, 3, 4, 5, 6], [10, 40, 20, 30, 80, 60], [1, 0, 1, 0, 1, 0])
        first = a.outcome_statistics(*args, seed=33, repetitions=50)
        second = a.outcome_statistics(*args, seed=33, repetitions=50)
        self.assertEqual(first, second)
        self.assertEqual(first["distance_statistics"]["success"]["mean"], 3)
        self.assertEqual(first["distance_statistics"]["found_but_failed"]["mean"], 4)
        self.assertEqual(sum(first["adjusted_logistic"]["bootstrap_fit_status_counts"].values()), 50)
        ci = first["point_biserial"]
        self.assertEqual(ci["valid_repetitions"]+ci["invalid_repetitions"], 50)
        ci = a.interval([None, 1], 2)
        self.assertEqual(ci["invalid_repetitions"], 1)
        self.assertIsNone(ci["ci95"])


class FileAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.output = self.root/"full_prrac", self.root/"result"
        self.source.mkdir()
        self.rows = []
        for i in range(12):
            row = base_row(i, found=i<10, success=i in (0, 1, 4, 6, 7))
            row.update(report_schema=a.REPORT_SCHEMA, evaluation_runtime_revision=a.REVISION,
                       execution_variant="B1_ATOMIC_LAST_VALID", search_recovery_variant="S2A1_C2_LOCAL_CONNECTOR",
                       runtime_integration_mode="native", manifest_sha256="fixture-identity", checkpoint="fixture.pt",
                       diagnostic_only=False, privileged_oracle=False, explore=False, training_update=False,
                       searcher_residual_off_enabled=False, search_value_guidance=json.dumps(dict(enabled=False)))
            if i<10:
                row.update(found_step=10+2*i, executor_target_received_step=11+2*i)
                if row["success"]:
                    row["first_contact_step"] = 15+2*i
            self.rows.append(row)
        self.manifest = dict(evaluation_episodes=12, manifest_sha256="fixture-identity",
                             scenarios=[dict(scenario_id=r["scenario_id"], scenario_seed=r["scenario_seed"], max_steps=400) for r in self.rows])
        self.config = dict(evaluation_episodes=12, manifest_sha256="fixture-identity", modes=["full_prrac"],
                           resolved_scenario_ids=[r["scenario_id"] for r in self.rows], observation_dim=28, action_dim=3, critic_dim=124,
                           report_schema=a.REPORT_SCHEMA, evaluation_runtime_revision=a.REVISION, execution_variant="B1_ATOMIC_LAST_VALID",
                           runtime_integration_mode="native", search_value_guidance=dict(enabled=False),
                           resolved_search_recovery_variants=["S2A1_C2_LOCAL_CONNECTOR"], max_steps=400,
                           resolved_checkpoint_paths=["fixture.pt"])
        self.save()

    def save(self):
        save_csv(self.source/"episode_evaluation.csv", self.rows)
        for name, doc in (("evaluation_manifest.json", self.manifest), ("resolved_evaluation_config.json", self.config)):
            (self.source/name).write_text(json.dumps(doc), encoding="utf-8")

    def run_analysis(self):
        return a.analyze(self.source, self.output, bootstrap_repetitions=20, expected_episodes=12)

    def test_manifest_hashes_and_no_historical_input_modifications(self):
        before = {p: p.read_bytes() for p in self.source.iterdir()}
        summary = self.run_analysis()
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["episode_counts"]["found_count"], 10)
        manifest = json.loads((self.output/"found_executor_distance_manifest.json").read_text())
        for filename, digest in manifest["output_sha256"].items():
            self.assertEqual(a.sha256(self.output/filename), digest)
        for item in manifest["input_files"]:
            self.assertEqual(a.sha256(item["path"]), item["sha256"])
        self.assertEqual(manifest["source_evaluation_manifest_sha256"], a.sha256(self.source/"evaluation_manifest.json"))
        self.assertEqual(before, {p: p.read_bytes() for p in self.source.iterdir()})
        self.assertEqual(summary["contact_auxiliary"]["status"], "not_refitted")
        rows = a.read_csv(self.output/"found_executor_distance_episode.csv")
        self.assertEqual(rows[0]["target_speed_at_found"], "NA")
        self.assertEqual(rows[0]["handoff_delay_steps"], "0")  # observed/derived zero is not missing filler

    def test_missing_primary_blocks_all_models_and_keeps_ids(self):
        self.rows[0][a.DISTANCE_FIELD] = ""
        self.rows[1][a.DISTANCE_FIELD] = "NA"
        self.save()
        summary = self.run_analysis()
        self.assertEqual(summary["status"], "blocked_incomplete_primary_population")
        self.assertEqual(summary["missingness"][a.X_FIELD], dict(count=2, scenario_ids=["scenario_0", "scenario_1"]))
        self.assertNotIn("adjusted_logistic", summary)
        self.assertEqual(len(a.read_csv(self.output/"found_executor_distance_episode.csv")), 10)

    def test_missing_found_step_blocks_adjustment_without_deleting(self):
        self.rows[0].update(found_step="", executor_target_received_step="", handoff_delay="", first_contact_step="")
        self.save()
        self.assertEqual(self.run_analysis()["status"], "blocked_incomplete_primary_population")

    def test_reject_mixed_mode_or_subset_or_duplicate_before_writes(self):
        original = copy.deepcopy(self.rows)
        for change in (lambda: self.rows[0].update(evaluation_mode="searcher_residual_off"),
                       lambda: self.rows.pop(), lambda: self.rows[0].update(scenario_id="scenario_1")):
            self.rows = copy.deepcopy(original)
            change()
            self.save()
            with self.assertRaises(ValueError):
                self.run_analysis()
            self.assertFalse(self.output.exists())

    def test_reject_unknown_semantics_and_protocol(self):
        for key, value in (("evaluation_runtime_revision", "unknown"), ("action_dim", 4), ("modes", ["searcher_residual_off"])):
            previous = self.config[key]
            self.config[key] = value
            self.save()
            with self.assertRaises(ValueError):
                self.run_analysis()
            self.config[key] = previous

    def test_refuse_overwrite_and_nested_source(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_analysis()
        with self.assertRaises(ValueError):
            a.analyze(self.source, self.source/"analysis")

    def test_contact_discordance_runs_auxiliary(self):
        self.rows[2].update(contact_episode=True, first_contact_step=20)
        self.save()
        summary = self.run_analysis()
        self.assertIn("distance_to_contact", summary["contact_auxiliary"])
        self.assertEqual(summary["contact_success_equality"]["discordant_scenario_ids"], ["scenario_2"])


if __name__ == "__main__":
    unittest.main()
