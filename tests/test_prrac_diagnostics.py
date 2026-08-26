from __future__ import annotations

import unittest
from unittest import mock

import torch

from chapter3_bser.experiments.phase1c_prrac import train_phase1c_prrac as trainer


class PRRACDiagnosticsAggregationTests(unittest.TestCase):
    def test_update_diagnostics_do_not_overwrite_rollout_diagnostics(self) -> None:
        rollout = {
            "gate_mean": 0.2,
            "router_accuracy": 2.0 / 3.0,
            "router_confusion_matrix": [[2, 1, 0], [0, 0, 0], [0, 0, 0]],
            "stage_critic_losses": {"search": None, "intercept": None, "hold": None},
            "stage_td_errors": {"search": None, "intercept": None, "hold": None},
        }
        update = {
            "gate_mean": 0.9,
            "router_accuracy": 0.5,
            "router_confusion_matrix": [[1, 1, 0], [0, 0, 0], [0, 0, 0]],
            "stage_critic_losses": {"search": 1.25, "intercept": None, "hold": None},
            "stage_td_errors": {"search": 0.75, "intercept": None, "hold": None},
        }

        combined = trainer._combine_episode_diagnostics(rollout, update)

        self.assertEqual(combined["gate_mean"], 0.2)
        self.assertEqual(combined["router_accuracy"], 2.0 / 3.0)
        self.assertEqual(combined["router_rollout_accuracy"], 2.0 / 3.0)
        self.assertEqual(combined["router_update_accuracy"], 0.5)
        self.assertEqual(combined["rollout_diagnostics"]["gate_mean"], 0.2)
        self.assertEqual(combined["update_diagnostics"]["gate_mean"], 0.9)
        self.assertEqual(combined["stage_critic_losses"]["search"], 1.25)

    def test_summary_accuracy_matches_confusion_matrix_and_stage_coverage(self) -> None:
        rows = [
            {
                "router_rollout_confusion_matrix": [
                    [4, 1, 0],
                    [2, 3, 0],
                    [0, 0, 0],
                ],
                "router_update_confusion_matrix": [
                    [2, 0, 0],
                    [1, 1, 0],
                    [0, 0, 0],
                ],
            },
            {
                "router_rollout_confusion_matrix": [
                    [1, 0, 0],
                    [0, 2, 0],
                    [0, 0, 0],
                ],
                "router_update_confusion_matrix": [
                    [1, 1, 0],
                    [0, 2, 0],
                    [0, 0, 0],
                ],
            },
        ]

        summary = trainer._router_summary(rows)
        matrix = summary["router_confusion_matrix"]
        expected = sum(matrix[index][index] for index in range(3)) / sum(
            sum(row) for row in matrix
        )

        self.assertEqual(summary["router_accuracy"], expected)
        self.assertEqual(summary["router_rollout_accuracy"], expected)
        self.assertAlmostEqual(summary["router_update_accuracy"], 6.0 / 8.0)
        self.assertEqual(
            summary["stage_coverage"],
            {"search": True, "intercept": True, "hold": False},
        )
        self.assertFalse(summary["all_stages_observed"])

    def test_runtime_metadata_records_required_cpu_fields(self) -> None:
        with mock.patch.object(trainer.torch.cuda, "is_available", return_value=False):
            metadata = trainer._runtime_metadata("cpu", torch.device("cpu"))

        for field in (
            "python_version",
            "torch_version",
            "cuda_available",
            "requested_device",
            "resolved_device",
            "cuda_device_name",
            "hostname",
        ):
            self.assertIn(field, metadata)
        self.assertFalse(metadata["cuda_available"])
        self.assertEqual(metadata["requested_device"], "cpu")
        self.assertEqual(metadata["resolved_device"], "cpu")
        self.assertIsNone(metadata["cuda_device_name"])


if __name__ == "__main__":
    unittest.main()
