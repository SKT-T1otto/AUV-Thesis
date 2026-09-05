import copy
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from chapter3_bser.experiments.phase1c_prrac.search_value_audit.metrics import window_label, binary_metrics, prediction_summary, paired_outcomes
from chapter3_bser.experiments.phase1c_prrac.search_value_audit.provenance import training_label_baseline, overlap_report, Progress, read_json, atomic_json, load_checkpoint, file_hash
from chapter3_bser.experiments.phase1c_prrac.search_value_audit.features import compare_feature, pool_audit, feature_catalog
from chapter3_bser.experiments.phase1c_prrac.search_value_audit.runner import run_work, record_exception
from chapter3_bser.experiments.phase1c_prrac.search_value_audit.state_fingerprint import component_fingerprint, rng_state
from chapter3_bser.experiments.phase1c_prrac.search_value_audit.analysis_bundle import bundle, D1
from chapter3_bser.greedy_solver import solve_joint_greedy
from tests.test_search_value_guided_ranking import fixture, scorer
from tests.search_value_audit_support import scene, synthetic_checkpoint, arithmetic_worker


class LabelMetricTests(unittest.TestCase):
    def test_pre_action_window_boundaries(self):
        for delta, expected in ((1, 1), (50, 1), (51, 0)):
            self.assertEqual(window_label(100, 100+delta, 200)["label"], expected)
        self.assertEqual(window_label(100, 100, 200)["window"], "already_found")
        self.assertFalse(window_label(351, 370, 400)["main_eligible"])
        self.assertTrue(window_label(350, None, 400)["main_eligible"])
        self.assertEqual(window_label(350, None, 400)["label"], 0)
        self.assertTrue(window_label(100, None, 120)["censored"])
        self.assertIsNone(window_label(351, None, 400)["label"])
        self.assertEqual(window_label(100, 110, 120)["label"], 1)

    def test_empty_and_single_class_are_not_fake_success(self):
        for y in ([], [0, 0], [1, 1]):
            result = binary_metrics(y, [.2]*len(y))
            self.assertIsNone(result["average_precision"])
            self.assertIsNone(result["balanced_accuracy"])
        self.assertIsNone(binary_metrics([], [])["brier"])
        self.assertIsNone(binary_metrics([0, 1], [0, 0])["precision"])
        self.assertAlmostEqual(binary_metrics([1, 0], [.8, .8])["average_precision"], .5)

    def test_scenario_units_not_repeated_team_labels(self):
        rows = [dict(scenario_id=str(scene_id), scenario_seed=scene_id, step=t, agent_id=a,
                     prediction=.25, label=scene_id, censored=False, main_eligible=True)
                for scene_id, count in ((0, 3), (1, 1)) for t in range(count) for a in range(3)]
        result, bins = prediction_summary(rows, dict(positive_fraction=.2), replicates=50)
        group = result["groups"]["all"]
        self.assertEqual((group["scenario_count"], group["unique_state_count"], group["agent_state_count"]), (2, 4, 12))
        self.assertEqual(group["predictors"]["head"]["micro"]["positive_fraction"], .25)
        self.assertEqual(group["predictors"]["head"]["scenario_equal"]["positive_fraction"], .5)
        self.assertEqual(result["brier_minus_training_constant_cluster_bootstrap"]["seed"], 61729)
        self.assertTrue(bins)

    def test_training_baseline_uses_unique_stored_masked_labels_only(self):
        payload = {"prrac_replay_state": dict(base_replay=dict(filled_i=5, max_steps=5, episode_ids=[1, 1, 1, 2, 2], steps=[1, 1, 2, 1, 2]),
                    search_value_valid=[True, True, True, False, True], stage_before=[0, 0, 0, 0, 1],
                    future_found=np.array([[[v]]*3 for v in (1, 1, 0, 1, 1)]))}
        result = training_label_baseline(payload)
        self.assertEqual(result["unique_transitions"], 2)
        self.assertEqual(result["positive_fraction"], .5)
        self.assertFalse(result["sample_weights_used"])
        # Neither the API nor the source payload has validation labels.
        self.assertIsNone(training_label_baseline({})["positive_fraction"])

    def test_same_id_different_seed_and_geometry_is_not_overlap(self):
        a, b = scene(), scene(200)
        b["initial_agent_positions"][0][0] += 1
        self.assertFalse(overlap_report([a], [b])["overlapping"])
        b["initial_agent_positions"] = copy.deepcopy(a["initial_agent_positions"])
        self.assertTrue(overlap_report([a], [b])["overlapping"])
        b["target_initial_velocity"][0] += .1
        self.assertFalse(overlap_report([a], [b])["overlapping"])
        b["scenario_seed"] = a["scenario_seed"]
        self.assertTrue(overlap_report([a], [b])["overlapping"])


class FeatureTests(unittest.TestCase):
    def test_empty_actual_generation_is_observed_before_early_return(self):
        from chapter3_bser.online.allocator import BSEROnlineAllocator
        from tests.bser_test_utils import synthetic_state
        state = synthetic_state()
        allocator = BSEROnlineAllocator()
        observed = []
        allocator._audit_candidate_generation_observer = lambda *args, **kwargs: observed.append((args, kwargs))
        with patch("chapter3_bser.online.allocator.generate_candidates", return_value=SimpleNamespace(search_candidates=(), standby_candidates=())) as generated:
            result = allocator.allocate(state)
        self.assertEqual(generated.call_count, 1)
        self.assertEqual(result.status, "NO_FEASIBLE_CANDIDATES")
        self.assertEqual(observed[0][0][1], ())
        self.assertEqual(observed[0][1]["scope"], "full")

    def test_same_time_required_and_overlay_stratified(self):
        kwargs = dict(observation_step=2, public_step=2, observation_position=[0]*3, public_position=[0]*3,
                      semantic=[2, 0, 0], public_tracking=[1, 0, 0], effective_tracking=[1, 0, 0])
        self.assertEqual(compare_feature(np.zeros(34), np.ones(34), **kwargs)["stratum"], "no_C2_overlay")
        kwargs["effective_tracking"] = [0, 1, 0]
        self.assertEqual(compare_feature(np.zeros(34), np.ones(34), **kwargs)["stratum"], "C2_overlay")
        kwargs["public_step"] = 3
        self.assertIsNone(compare_feature(np.zeros(34), np.ones(34), **kwargs)["delta"])
        self.assertEqual(len(feature_catalog()["fields"]), 34)

    def test_different_endpoints_can_have_identical_features(self):
        state, candidates, standby, context = fixture()
        route = np.array([state.agents[0].position, [2, 1, 1]])
        candidates = tuple(replace(c, path_points=route) for c in candidates)
        context = replace(context, candidates=candidates)
        value = scorer()
        value.observe_state(np.zeros((4, 28), np.float32), state)
        baseline = solve_joint_greedy(candidates, (standby,), context)
        representation, scores, _ = pool_audit(candidates, context, baseline, value, identity={}, decision_index=0)
        self.assertEqual(representation[0]["endpoint_count"], 2)
        self.assertEqual(representation[0]["exact_feature_count"], 1)
        self.assertEqual(representation[0]["same_feature_different_endpoint_groups"], 1)
        self.assertEqual(representation[0]["prediction_std"], 0)
        self.assertTrue(all("selected_prefix" in r and "context_hash" in r for r in scores))


class InfrastructureTests(unittest.TestCase):
    def test_checkpoint_is_read_only_and_head_and_schema_required(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"synthetic.pt"
            payload = synthetic_checkpoint()
            torch.save(payload, path)
            before = file_hash(path)
            load_checkpoint(path)
            self.assertEqual(before, file_hash(path))
            del payload["prrac_training_state"]["search_value_head"]
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "trained.*head"):
                load_checkpoint(path)
            payload = synthetic_checkpoint()
            payload["metadata"]["observation_dim"] = 29
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "observation_dim"):
                load_checkpoint(path)

    def test_complete_inventory_detects_missing_tracker_recovery_rng(self):
        state = dict(physics=np.ones(3), tracker=SimpleNamespace(cursor=4), recovery=SimpleNamespace(failed_routes=["a"], cooldown=20), rng=rng_state())
        expected = component_fingerprint(state)["sha256"]
        for name in ("tracker", "recovery", "rng"):
            self.assertNotEqual(expected, component_fingerprint({k: v for k, v in state.items() if k != name})["sha256"])
        state["recovery"].failed_routes.append("b")
        self.assertNotEqual(expected, component_fingerprint(state)["sha256"])
        with self.assertRaises(TypeError):
            component_fingerprint(dict(unknown=object()))

    def test_spawn_order_and_incremental_progress(self):
        work = [dict(unit=str(i), value=i) for i in (3, 1, 2)]
        with tempfile.TemporaryDirectory() as folder:
            progress = Progress(folder, 3)
            self.assertIsNone(read_json(Path(folder)/"progress.json")["estimated_remaining_seconds"])
            serial = run_work(work, 1, progress, worker=arithmetic_worker)
            self.assertEqual(read_json(Path(folder)/"progress.json")["completed"], 3)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                progress.write(stage="test", unit="3")
            parallel = run_work(work, 2, Progress(folder, 3), worker=arithmetic_worker)
            self.assertEqual(serial, parallel)
            try:
                raise RuntimeError("synthetic failure")
            except RuntimeError:
                self.assertEqual(record_exception(folder, Progress(folder, 1)), 1)
            self.assertEqual(read_json(Path(folder)/"progress.json")["status"], "failed")

    def test_bundle_allowlist_missing_files_and_large_model_exclusion(self):
        import zipfile
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/"D1"
            source.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "missing"):
                bundle([source], root/"missing.zip")
            for name in D1:
                atomic_json(source/name, {"status": "completed"})
            torch.save(synthetic_checkpoint(), source/"model.pt")
            atomic_json(source/"full_environment_state.json", {"not": "small evidence"})
            bundle([source], root/"ok.zip")
            with zipfile.ZipFile(root/"ok.zip") as archive:
                self.assertFalse(any("model.pt" in n or "full_environment" in n for n in archive.namelist()))


if __name__ == "__main__":
    unittest.main()
