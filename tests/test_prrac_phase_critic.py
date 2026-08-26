import unittest

import torch

from chapter3_bser.models.prrac.phase_twin_critic import (
    PhaseTwinCritic,
    gather_stage_values,
)


class PRRACCriticTests(unittest.TestCase):
    def test_strict_input_heads_gather_twin_independence_and_target_update(self):
        twin = PhaseTwinCritic(hidden_dim=32)
        values1, values2 = twin(torch.randn(6, 124))
        self.assertEqual(tuple(values1.shape), (6, 3))
        self.assertEqual(tuple(values2.shape), (6, 3))
        first1 = next(twin.critic1.parameters())
        first2 = next(twin.critic2.parameters())
        self.assertIsNot(first1, first2)
        labels = torch.tensor([0, 1, 2, 2, 1, 0])
        expected = values1[torch.arange(6), labels].reshape(-1, 1)
        torch.testing.assert_close(gather_stage_values(values1, labels), expected)
        with self.assertRaisesRegex(ValueError, "124"):
            twin(torch.randn(6, 125))

    def test_empty_stage_statistics_remain_finite(self):
        from chapter3_bser.models.prrac.prrac_maddpg import PRRACMADDPG

        losses, errors = PRRACMADDPG._stage_statistics(
            torch.ones(4), torch.ones(4), torch.zeros(4, dtype=torch.long)
        )
        self.assertEqual(losses["search"], 1.0)
        self.assertIsNone(losses["intercept"])
        self.assertIsNone(errors["hold"])


if __name__ == "__main__":
    unittest.main()
