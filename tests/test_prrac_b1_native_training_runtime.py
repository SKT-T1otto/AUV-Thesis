import json
import unittest

from chapter3_bser.experiments.phase1c_prrac import train_phase1c_prrac as trainer
from chapter3_bser.experiments.phase1c_prrac.runtime_factory import NATIVE_B1_RUNTIME_REVISION
from tests.test_prrac_cross_platform_runtime import _worker_jobs


class B1NativeTrainingRuntimeTests(unittest.TestCase):
    def test_s1_config_changes_only_allowed_runtime_fields(self):
        old = json.loads((trainer.ROOT / "configs/chapter3/bser_phase1c_prrac_train.json").read_text(encoding="utf-8"))
        new = trainer._load_config(trainer.ROOT / "configs/chapter3/bser_phase1c_prrac_s1_train.json")
        self.assertEqual(new["execution_runtime_revision"], NATIVE_B1_RUNTIME_REVISION)
        self.assertEqual(new["execution_variant"], "B1_ATOMIC_LAST_VALID")
        self.assertEqual(new["runtime_integration_mode"], "native")
        for key in ("architecture", "loss", "rl", "reward", "replay", "max_steps", "checkpoint_interval", "rolling_window"):
            self.assertEqual(new[key], old[key], key)
        self.assertTrue(new["search_continuity_diagnostics"]["enabled"])

    def test_native_worker_uses_b1_and_returns_search_metrics_without_tensors(self):
        native = trainer._load_config(
            trainer.ROOT / "configs/chapter3/bser_phase1c_prrac_s1_train.json"
        )
        job = _worker_jobs(1)[0]
        job.update(
            execution_runtime=native["execution_runtime"],
            execution_runtime_revision=NATIVE_B1_RUNTIME_REVISION,
            execution_variant="B1_ATOMIC_LAST_VALID",
            runtime_integration_mode="native",
            controller_factory_version="prrac.controller_factory.v1",
            search_continuity_diagnostics=native["search_continuity_diagnostics"],
        )
        result = trainer._collect_episode(job)
        self.assertFalse(trainer._contains_tensor(result))
        metrics = result[0]
        self.assertEqual(metrics["pre_found_step_count"], 1)
        self.assertIn("searcher_route_active_rate_pre_found", metrics)
        self.assertFalse(metrics["searcher_residual_off_enabled"])
        self.assertEqual(
            metrics["searcher_raw_action_norm_pre_found"],
            metrics["searcher_raw_residual_norm_mean_pre_found"],
        )
        self.assertIn("searcher_hold_rate_pre_found_agent_2", metrics)


if __name__ == "__main__":
    unittest.main()
