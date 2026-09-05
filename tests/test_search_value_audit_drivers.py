import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from chapter3_bser.experiments.phase1c_prrac.search_value_audit.runner import historical_inputs, arguments
from chapter3_bser.experiments.phase1c_prrac.search_value_audit.prediction_audit import training_scenarios, freeze_scenarios
from chapter3_bser.experiments.phase1c_prrac.search_value_audit.provenance import atomic_json, atomic_csv, read_json, fresh_output
from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from tests.search_value_audit_support import scene, small_job, synthetic_checkpoint


class DriverContractTests(unittest.TestCase):
    def history(self, root):
        payload = synthetic_checkpoint()
        identity = dict(checkpoint_path=str(root/"synthetic.pt"), checkpoint_sha256="synthetic-sha")
        scenarios = [scene(100, "reused_1"), scene(101, "reused_2")]
        body = dict(scenarios=scenarios, profile="M20_MOVING_UNKNOWN_MULTI")
        body["manifest_sha256"] = evaluator._hash(body)
        for mode in ("OFF", "ON"):
            output = root/mode
            output.mkdir()
            config = small_job(enabled=mode == "ON")["config"]
            config.update(max_steps=400, search_recovery_variants=["S2A1_C2_LOCAL_CONNECTOR"])
            atomic_json(output/"resolved_evaluation_config.json", config)
            atomic_json(output/"evaluation_manifest.json", body)
            rows = [dict(scenario_id=s["scenario_id"], scenario_seed=s["scenario_seed"], checkpoint=str(root/"synthetic.pt"),
                         checkpoint_config_hash=payload["metadata"]["config_hash"], checkpoint_episode=payload["completed_episode"],
                         execution_variant="B1_ATOMIC_LAST_VALID", search_recovery_variant="S2A1_C2_LOCAL_CONNECTOR",
                         evaluation_mode="full_prrac", runtime_integration_mode="native", manifest_sha256=body["manifest_sha256"],
                         search_value_guidance=dict(accepted_search_change_count=int(mode == "ON" and i == 1))) for i, s in enumerate(scenarios)]
            atomic_csv(output/"episode_evaluation.csv", rows)
            atomic_json(output/"search_value_guidance_metrics.json", dict(episodes=[{k: r[k] for k in
                ("scenario_id", "checkpoint", "evaluation_mode", "execution_variant", "search_recovery_variant", "search_value_guidance")} for r in rows]))
        return identity, payload

    def test_history_selects_all_logged_changes_and_checks_seed_join(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            identity, payload = self.history(root)
            history = historical_inputs(root/"OFF", root/"ON", identity, payload)
            self.assertEqual([r["scenario_id"] for r in history["selected"]], ["reused_2"])
            self.assertFalse(history["checkpoint_verification"]["historical_sha256_verified"])
            rows = evaluator._read_csv(root/"ON/episode_evaluation.csv")
            rows[0]["scenario_seed"] += 1
            atomic_csv(root/"ON/episode_evaluation.csv", rows)
            with self.assertRaisesRegex(ValueError, "join manifest"):
                historical_inputs(root/"OFF", root/"ON", identity, payload)

    def test_history_malformed_manifest_and_metrics_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            identity, payload = self.history(root)
            metrics = read_json(root/"ON/search_value_guidance_metrics.json")
            metrics["episodes"][1]["search_value_guidance"]["accepted_search_change_count"] = 500
            atomic_json(root/"ON/search_value_guidance_metrics.json", metrics)
            with self.assertRaisesRegex(ValueError, "metrics.*match"):
                historical_inputs(root/"OFF", root/"ON", identity, payload)
            body = read_json(root/"ON/evaluation_manifest.json")
            body["scenarios"][0]["scenario_seed"] += 1
            atomic_json(root/"ON/evaluation_manifest.json", body)
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                historical_inputs(root/"OFF", root/"ON", identity, payload)

    def test_training_manifest_must_crosscheck_checkpoint_episodes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"train.json"
            s = scene()
            atomic_json(path, dict(scenarios=[s]))
            args = SimpleNamespace(training_manifest=path, training_config=None)
            _, report = training_scenarios(args, dict(completed_episode=1, episode_metrics=[dict(scenario_id=s["scenario_id"], scenario_seed=s["scenario_seed"])]))
            self.assertTrue(report["verified"])
            _, report = training_scenarios(args, dict(completed_episode=1, episode_metrics=[]))
            self.assertFalse(report["verified"])

    def test_fixed_30_manifest_and_unverified_independence_not_relabeled(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenarios = [scene(51729+i, f"reused_{i}") for i in range(30)]
            for i, s in enumerate(scenarios):
                s["initial_agent_positions"][0][0] += .01*(i+1)
            manifest = dict(generator_seed=51729, scenarios=scenarios)
            atomic_json(root/"validation.json", manifest)
            atomic_json(root/"training.json", dict(scenarios=[scene()]))
            args = SimpleNamespace(manifest=root/"validation.json", training_manifest=root/"training.json", training_config=None)
            _, frozen, report = freeze_scenarios(args, {"ON": dict(scenarios=[scene(101)])}, dict(completed_episode=1, episode_metrics=[]))
            self.assertEqual(len(frozen), 30)
            self.assertEqual(report["status"], "independence_unverified")
            atomic_json(root/"validation.json", dict(generator_seed=51729, scenarios=scenarios[:29]))
            with self.assertRaisesRegex(ValueError, "30-scene"):
                freeze_scenarios(args, {}, {})

    def test_cli_help_modes_and_output_refuses_overwrite(self):
        for prediction in (False, True):
            with self.assertRaises(SystemExit) as stopped:
                arguments("diagnostic", ["--help"], prediction=prediction)
            self.assertEqual(stopped.exception.code, 0)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            atomic_json(root/"existing.json", {})
            with self.assertRaises(FileExistsError):
                fresh_output(root)

    def test_launchers_are_lf_and_root_relative(self):
        root = evaluator.ROOT
        for name in ("_search_value_audit_common.sh", "run_search_value_d1_audit.sh", "run_search_value_d2_audit.sh", "bundle_search_value_audits.sh"):
            data = (root/"scripts/linux"/name).read_bytes()
            self.assertNotIn(b"\r", data)
            self.assertIn(b"BASH_SOURCE[0]", data)
        self.assertIn(b"%~dp0..", (root/"scripts/test_search_value_audit.bat").read_bytes())


if __name__ == "__main__":
    unittest.main()
