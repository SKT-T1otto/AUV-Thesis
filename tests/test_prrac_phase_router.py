import unittest

import torch

from chapter3_bser.models.prrac.phase_routed_actor import PhaseRoutedResidualActor


class PRRACRouterTests(unittest.TestCase):
    def test_router_probabilities_are_deterministic_and_finite(self):
        torch.manual_seed(11)
        actor = PhaseRoutedResidualActor(hidden_dim=16, expert_hidden_dim=12).eval()
        observation = torch.randn(7, 28)
        first = actor(observation)
        second = actor(observation)
        self.assertEqual(tuple(first.router_logits.shape), (7, 3))
        self.assertEqual(tuple(first.router_probabilities.shape), (7, 3))
        torch.testing.assert_close(first.router_probabilities.sum(dim=-1), torch.ones(7))
        torch.testing.assert_close(first.router_probabilities, second.router_probabilities)
        self.assertTrue(torch.isfinite(first.router_probabilities).all())
        with self.assertRaisesRegex(ValueError, "28"):
            actor(torch.randn(7, 29))


if __name__ == "__main__":
    unittest.main()
