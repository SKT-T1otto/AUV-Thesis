from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
import unittest

import torch

from chapter3_bser.experiments.phase1c_common import TransitionPhase
from chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.reward_adapter import (
    Phase1CExecutionRewardAdapter,
)
from chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.training_env import (
    Phase1CV2TrainingEnv,
)


@dataclass
class Task:
    step: int = 0
    target_found: bool = False
    executor_knows_target: bool = False
    mission_complete: bool = False
    handoff_step: int | None = None
    completion_step: int | None = None


class Runtime:
    def __init__(self) -> None:
        self.max_steps = 10
        self.device = torch.device("cpu")
        self.reward_scale = 1.0
        self.last_reward_components = {}
        self._capture_hold_counter = 0
        self.capture_contact_step_count = 0
        self.capture_full_hold_step_count = 0
        self.capture_hold_counter_max = 0
        self.capture_swept_min_distance = 1.0
        self._agent_pos = torch.zeros(4, 3)
        self.target_state = SimpleNamespace(position=torch.zeros(3))
        self.predicted_intercept_position = torch.zeros(3)
        self._collision_flags = torch.zeros(4, dtype=torch.bool)
        self.last_residual_contribution_ratio_search = 0.0
        self.last_residual_contribution_ratio_executor = 0.0
        self.target_prediction_map_fallback_count = 0
        self.initial_endpoint_fallback_count = 0


class FakeGuidedEnv:
    def __init__(self) -> None:
        self.unwrapped = Runtime()
        self.task = Task()
        self.initial_endpoint_fallback_count = 0
        self.sequence = [
            {
                "task": Task(step=1, target_found=True, handoff_step=1),
                "reward": torch.tensor([1.0, 1.0, 1.0, 0.0]),
            },
            {
                "task": Task(step=2, target_found=True, executor_knows_target=True, handoff_step=1),
                "reward": torch.tensor([1.0, 1.0, 1.0, 0.0]),
                "contact": 1,
                "hold": 1,
            },
            {
                "task": Task(step=3, target_found=True, executor_knows_target=True, handoff_step=1),
                "reward": torch.tensor([1.0, 1.0, 1.0, 0.0]),
                "contact": 1,
                "full_hold": 1,
                "hold": 2,
            },
            {
                "task": Task(
                    step=4,
                    target_found=True,
                    executor_knows_target=True,
                    mission_complete=True,
                    handoff_step=1,
                    completion_step=4,
                ),
                "reward": torch.tensor([1.0, 1.0, 1.0, 0.0]),
                "contact": 1,
                "full_hold": 1,
                "hold": 3,
            },
        ]
        self.index = 0

    def reset(self, scenario=None):
        self.index = 0
        self.task = Task()
        return tuple(torch.zeros(28) for _ in range(4))

    def step(self, actions):
        item = self.sequence[self.index]
        self.index += 1
        self.task = item["task"]
        self.unwrapped.capture_contact_step_count += int(item.get("contact", 0))
        self.unwrapped.capture_full_hold_step_count += int(item.get("full_hold", 0))
        self.unwrapped._capture_hold_counter = int(item.get("hold", 0))
        self.unwrapped.capture_hold_counter_max = max(
            self.unwrapped.capture_hold_counter_max,
            self.unwrapped._capture_hold_counter,
        )
        obs = tuple(torch.full((28,), float(self.index)) for _ in range(4))
        done = tuple(self.task.mission_complete for _ in range(4))
        return obs, item["reward"], done

    def get_task_state(self):
        return replace(self.task)

    def close(self):
        return None


class Phase1CV2ExecutionRewardTests(unittest.TestCase):
    def test_execution_bonuses_are_incremental_and_terminal_bonus_is_one_shot(self) -> None:
        adapter = Phase1CExecutionRewardAdapter()
        runtime = SimpleNamespace(reward_scale=1.0, last_reward_components={})
        before = Task(target_found=True)
        after = Task(target_found=True)
        first = adapter.adjust(
            torch.zeros(4),
            task_before=before,
            task_after=after,
            runtime=runtime,
            contact=True,
            full_hold=True,
            hold_counter_before=0,
            hold_counter_after=1,
        )
        self.assertAlmostEqual(first.rewards[3].item(), 0.45, places=6)
        second = adapter.adjust(
            torch.zeros(4),
            task_before=after,
            task_after=Task(target_found=True, mission_complete=True),
            runtime=runtime,
            contact=True,
            full_hold=True,
            hold_counter_before=1,
            hold_counter_after=2,
        )
        self.assertAlmostEqual(second.rewards[3].item(), 2.20, places=6)
        repeated = adapter.adjust(
            torch.zeros(4),
            task_before=Task(target_found=True, mission_complete=True),
            task_after=Task(target_found=True, mission_complete=True),
            runtime=runtime,
            contact=True,
            full_hold=False,
            hold_counter_before=2,
            hold_counter_after=2,
        )
        self.assertEqual(repeated.rewards[3].item(), 0.0)
        self.assertEqual(adapter.terminal_bonus_count, 1)

    def test_training_wrapper_emits_reward_metadata_and_diagnostics(self) -> None:
        env = Phase1CV2TrainingEnv(FakeGuidedEnv())
        observations = env.reset(
            {"scenario_id": "s", "scenario_seed": 7},
            episode_id=4,
            episode_index=4,
        )
        self.assertEqual(len(observations), 4)
        phases = []
        rewards = []
        for _ in range(4):
            _, reward, _ = env.step(torch.zeros(4, 3))
            phases.append(env.last_transition_metadata.phase)
            rewards.append(torch.as_tensor(reward))
        self.assertEqual(
            phases,
            [
                TransitionPhase.POST_FOUND,
                TransitionPhase.CONTACT,
                TransitionPhase.HOLD,
                TransitionPhase.SUCCESS,
            ],
        )
        self.assertTrue(torch.equal(rewards[1][:3], torch.zeros(3)))
        self.assertAlmostEqual(rewards[1][3].item(), 0.25, places=6)
        self.assertAlmostEqual(rewards[2][3].item(), 0.20, places=6)
        self.assertAlmostEqual(rewards[3][3].item(), 2.20, places=6)
        row = env.finalize_episode()
        self.assertIs(row["found"], True)
        self.assertIs(row["success"], True)
        self.assertEqual(row["contact_bonus_count"], 1)
        self.assertEqual(row["hold_bonus_count"], 2)
        self.assertEqual(row["terminal_bonus_count"], 1)


if __name__ == "__main__":
    unittest.main()
