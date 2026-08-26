from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import unittest

import torch

from chapter3_bser.experiments.phase1c_common.execution_diagnostics import (
    ExecutionEpisodeDiagnostics,
)


@dataclass
class Task:
    step: int
    target_found: bool = False
    executor_knows_target: bool = False
    mission_complete: bool = False
    handoff_step: int | None = None
    completion_step: int | None = None


class Runtime:
    def __init__(self) -> None:
        self.max_steps = 10
        self._agent_pos = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]
        )
        self.target_state = SimpleNamespace(position=torch.tensor([0.0, 0.0, 0.0]))
        self.predicted_intercept_position = torch.tensor([1.0, 0.0, 0.0])
        self._collision_flags = torch.tensor([False, False, False, False])
        self.last_residual_contribution_ratio_search = 0.1
        self.last_residual_contribution_ratio_executor = 0.2
        self.capture_contact_step_count = 0
        self.capture_full_hold_step_count = 0
        self.capture_hold_counter_max = 0
        self.capture_swept_min_distance = 5.0
        self.target_prediction_error_at_delivery = None
        self.mean_target_prediction_error = None
        self.target_prediction_map_fallback_count = 0
        self.executor_received_target_step = None
        self.last_handoff_delay = None
        self.path_unreachable_count = 99


class Env:
    def __init__(self) -> None:
        self.unwrapped = Runtime()
        self.task = Task(step=0)
        self.initial_endpoint_fallback_count = 2

    def get_task_state(self):
        return self.task


class Phase1CExecutionDiagnosticsTests(unittest.TestCase):
    def test_execution_diagnostics_tracks_post_found_chain_without_mutation(self) -> None:
        env = Env()
        collector = ExecutionEpisodeDiagnostics()
        collector.reset(
            env,
            episode_index=0,
            scenario_id="scenario-1",
            scenario_seed=11,
            max_steps=10,
        )

        before = Task(step=1)
        after = Task(step=2, target_found=True, handoff_step=2)
        env.task = after
        collector.observe_step(
            env,
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            task_before=before,
            task_after=after,
        )

        env.unwrapped.executor_received_target_step = 3
        env.unwrapped.last_handoff_delay = 1
        env.unwrapped.target_prediction_error_at_delivery = 0.75
        env.unwrapped.mean_target_prediction_error = 0.50
        env.unwrapped.capture_contact_step_count = 1
        env.unwrapped.capture_swept_min_distance = 0.25
        env.unwrapped._agent_pos[3] = torch.tensor([1.0, 0.0, 0.0])
        before = after
        after = Task(
            step=3,
            target_found=True,
            executor_knows_target=True,
            handoff_step=2,
        )
        env.task = after
        collector.observe_step(
            env,
            torch.tensor([0.0, 0.0, 0.0, 0.5]),
            task_before=before,
            task_after=after,
        )

        env.unwrapped.capture_full_hold_step_count = 1
        env.unwrapped.capture_hold_counter_max = 3
        before = after
        after = Task(
            step=4,
            target_found=True,
            executor_knows_target=True,
            mission_complete=True,
            handoff_step=2,
            completion_step=4,
        )
        env.task = after
        collector.observe_step(
            env,
            torch.tensor([0.0, 0.0, 0.0, 2.0]),
            task_before=before,
            task_after=after,
        )

        row = collector.finalize(env)
        self.assertIs(row["found"], True)
        self.assertIs(row["success"], True)
        self.assertIs(row["success_if_found"], True)
        self.assertEqual(row["found_step"], 2)
        self.assertEqual(row["executor_target_received_step"], 3)
        self.assertEqual(row["found_to_target_received_steps"], 1)
        self.assertEqual(row["found_to_success_steps"], 2)
        self.assertEqual(row["remaining_steps_after_found"], 8)
        self.assertEqual(row["executor_distance_to_target_at_found"], 5.0)
        self.assertEqual(row["executor_distance_to_target_at_received"], 1.0)
        self.assertEqual(row["executor_distance_to_intercept_at_received"], 0.0)
        self.assertEqual(row["capture_contact_step_count"], 1)
        self.assertEqual(row["capture_full_hold_step_count"], 1)
        self.assertEqual(row["target_prediction_error_at_delivery"], 0.75)
        self.assertEqual(row["initial_planner_endpoint_fallback_count"], 2)
        self.assertIsNone(row["executor_path_unreachable_count"])
        self.assertEqual(env.unwrapped.path_unreachable_count, 99)

    def test_execution_diagnostics_uses_only_exact_executor_path_metric(self) -> None:
        env = Env()
        env.unwrapped.executor_path_unreachable_count = 7
        env.task = Task(step=1)
        collector = ExecutionEpisodeDiagnostics()
        collector.reset(env, episode_index=0, max_steps=10)
        row = collector.finalize(env)
        self.assertEqual(row["executor_path_unreachable_count"], 7)


if __name__ == "__main__":
    unittest.main()
