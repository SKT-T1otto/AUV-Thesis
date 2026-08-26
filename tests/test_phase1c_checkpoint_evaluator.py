from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chapter3_bser.experiments.phase1c_bser_rmaddpg import (
    evaluate_phase1c_checkpoints as evaluator,
)


def _payload(schema: str = "bser.phase1c.training_state.v1"):
    return {
        "schema": schema,
        "metadata": {
            "method": "ch3_bser_rmaddpg_phase1c",
            "observation_dim": 28,
            "action_dim": 3,
            "critic_dim": 124,
        },
        "completed_episode": 100,
        "maddpg_training_state": {
            "init_dict": {
                "agent_init_params": [
                    {"num_in_pol": 28, "num_out_pol": 3, "num_in_critic": 124}
                    for _ in range(4)
                ]
            }
        },
    }


class Phase1CCheckpointEvaluatorTests(unittest.TestCase):
    def test_checkpoint_validation_accepts_v1_and_v2_but_rejects_dimension_drift(self) -> None:
        self.assertTrue(evaluator._validate_checkpoint_payload(_payload())["schema"].endswith("v1"))
        self.assertTrue(
            evaluator._validate_checkpoint_payload(
                _payload("bser.phase1c.training_state.v2")
            )["schema"].endswith("v2")
        )
        bad = _payload()
        bad["metadata"]["critic_dim"] = 125
        with self.assertRaisesRegex(ValueError, "critic_dim"):
            evaluator._validate_checkpoint_payload(bad)

    def test_checkpoint_aggregation_reports_conditional_success_rate(self) -> None:
        rows = [
            {"found": True, "success": True, "episode_length": 4},
            {"found": True, "success": False, "episode_length": 8},
            {"found": False, "success": False, "episode_length": 10},
        ]
        summary = evaluator._aggregate_checkpoint(
            rows,
            {
                "checkpoint": "/tmp/a.pt",
                "schema": "bser.phase1c.training_state.v1",
                "completed_episode": 100,
                "implementation_version": "v1",
            },
        )
        self.assertAlmostEqual(summary["found_rate"], 2 / 3)
        self.assertAlmostEqual(summary["success_rate"], 1 / 3)
        self.assertAlmostEqual(summary["success_if_found_rate"], 1 / 2)
        self.assertAlmostEqual(summary["mean_episode_length"], 22 / 3)

    def test_run_evaluation_writes_diagnostic_artifacts_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            checkpoint = tmp_path / "checkpoint.pt"
            checkpoint.write_bytes(b"placeholder")
            config = {
                "schema": "bser.phase1c.diagnostic_eval.v1",
                "method": "ch3_bser_rmaddpg_phase1c",
                "implementation_version": "diagnostics.v1",
                "profile": "M20_MOVING_UNKNOWN_MULTI",
                "base_candidate": "unused",
                "device": "cpu",
                "split": "validation",
                "evaluation_seed": 1,
                "scenario_count": 2,
                "max_steps": 10,
                "observation_dim": 28,
                "action_dim": 3,
                "critic_dim": 124,
                "checkpoints": [],
                "checkpoint_globs": [],
                "output_dir": str(tmp_path / "unused"),
            }
            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            scenarios = [
                {"scenario_id": "s0", "scenario_seed": 10},
                {"scenario_id": "s1", "scenario_seed": 11},
            ]
            fake_checkpoint_payload = {
                "schema": "bser.phase1c.training_state.v1",
                "metadata": {"implementation_version": "v1"},
                "completed_episode": 100,
            }

            def fake_episode(_actor, scenario, *, episode_index, config):
                return {
                    "episode_index": episode_index,
                    "scenario_id": scenario["scenario_id"],
                    "found": True,
                    "success": episode_index == 0,
                    "episode_length": 5,
                    "executor_path_unreachable_count": None,
                }

            with patch.object(
                evaluator,
                "_build_scenarios",
                return_value=(scenarios, {"ok": True}),
            ), patch.object(
                evaluator,
                "_load_actor",
                return_value=(object(), fake_checkpoint_payload),
            ), patch.object(
                evaluator,
                "_evaluate_episode",
                side_effect=fake_episode,
            ):
                output = tmp_path / "diagnostics"
                summary = evaluator.run_evaluation(
                    config_path=config_path,
                    checkpoints=[checkpoint],
                    output_dir=output,
                )

            self.assertIs(summary["passed"], True)
            self.assertIs(summary["training_update"], False)
            self.assertTrue((output / "episode_execution_diagnostics.csv").is_file())
            self.assertTrue((output / "checkpoint_execution_summary.csv").is_file())
            self.assertTrue((output / "diagnostic_eval_summary.json").is_file())
            source = Path(evaluator.__file__).read_text(encoding="utf-8")
            self.assertNotIn(".update(", source)
            self.assertNotIn("update_critic_only", source)


if __name__ == "__main__":
    unittest.main()
