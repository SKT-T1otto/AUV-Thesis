from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.reward_adapter import (
    Phase1CExecutionRewardAdapter,
)


@dataclass
class Task:
    target_found: bool = False
    mission_complete: bool = False


def runtime(protocol: str):
    return SimpleNamespace(
        reward_protocol=protocol,
        reward_scale=1.0,
        last_reward_components={
            "reward_find_event": torch.tensor([1.0, 2.0, 3.0, 0.0]),
            "reward_early_find": torch.tensor([0.5, 0.5, 0.5, 0.0]),
        },
    )


class Phase1CV2RewardProtocolTests(unittest.TestCase):
    def test_searcher_freeze_is_explicit_and_independent_of_core_protocol(self) -> None:
        for protocol in ("legacy", "efficiency_protocol_v2", "unknown"):
            with self.subTest(protocol=protocol):
                adapter = Phase1CExecutionRewardAdapter()
                result = adapter.adjust(
                    torch.tensor([1.0, 2.0, 3.0, 0.5]),
                    task_before=Task(target_found=True),
                    task_after=Task(target_found=True),
                    runtime=runtime(protocol),
                    contact=False,
                    full_hold=False,
                    hold_counter_before=0,
                    hold_counter_after=0,
                )
                self.assertTrue(torch.equal(result.rewards[:3], torch.zeros(3)))
                self.assertAlmostEqual(result.rewards[3].item(), 0.5)

    def test_discovery_transition_restores_only_discovery_components(self) -> None:
        adapter = Phase1CExecutionRewardAdapter()
        result = adapter.adjust(
            torch.tensor([-0.5, -0.5, -0.5, 0.0]),
            task_before=Task(target_found=False),
            task_after=Task(target_found=True),
            runtime=runtime("anything"),
            contact=False,
            full_hold=False,
            hold_counter_before=0,
            hold_counter_after=0,
        )
        expected = torch.tanh(torch.tensor([1.5, 2.5, 3.5]))
        self.assertTrue(torch.allclose(result.rewards[:3], expected))
        self.assertAlmostEqual(
            result.breakdown["discovery_reward_restored"],
            float(expected.sum()),
        )

    def test_reward_adapter_source_has_no_core_protocol_dependency(self) -> None:
        source = Path(
            "chapter3_bser/experiments/phase1c_bser_rmaddpg_v2/reward_adapter.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("efficiency_protocol_v2", source)
        self.assertNotIn("reward_protocol ==", source)


if __name__ == "__main__":
    unittest.main()
