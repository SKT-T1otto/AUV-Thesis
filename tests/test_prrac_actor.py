import unittest

import torch

from chapter3_bser.models.prrac.phase_routed_actor import PhaseRoutedResidualActor


class PRRACActorTests(unittest.TestCase):
    def test_shapes_ranges_independence_and_gradients(self):
        actor = PhaseRoutedResidualActor(hidden_dim=24, expert_hidden_dim=16)
        observation = torch.randn(9, 28)
        output = actor(observation)
        self.assertEqual(tuple(output.gated_residual_action.shape), (9, 3))
        self.assertEqual(tuple(output.expert_actions.shape), (9, 3, 3))
        self.assertEqual(tuple(output.trust_gate.shape), (9, 1))
        self.assertEqual(tuple(output.alignment_cosine.shape), (9, 1))
        self.assertTrue((output.expert_actions.abs() <= 1.0).all())
        self.assertTrue((output.gated_residual_action.abs() <= 1.0).all())
        experts = actor.residual_experts.experts
        self.assertIsNot(experts[0].net[0].weight, experts[1].net[0].weight)
        output.gated_residual_action.square().mean().backward()
        for expert in experts:
            gradients = [p.grad for p in expert.parameters()]
            self.assertTrue(all(value is not None for value in gradients))
            self.assertTrue(all(torch.isfinite(value).all() for value in gradients))


if __name__ == "__main__":
    unittest.main()
