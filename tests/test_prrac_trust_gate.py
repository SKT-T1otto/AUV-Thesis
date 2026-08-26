import unittes

import torch

from chapter3_bser.models.prrac.phase_routed_actor import PhaseRoutedResidualActor
from chapter3_bser.models.prrac.residual_trust_gate import ResidualTrustGate


class PRRACTrustGateTests(unittest.TestCase):
    def test_monotone_alignment_zero_direction_and_bounds(self):
        gate = ResidualTrustGate(embedding_dim=8, hidden_dim=8)
        base = torch.zeros(5, 1)
        alignment = torch.tensor([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
        values = gate.gate_from_base_logit(base, alignment).reshape(-1)
        self.assertTrue(torch.all(values[1:] >= values[:-1]))
        self.assertTrue(torch.all((values >= 0.0) & (values <= 1.0)))

        actor = PhaseRoutedResidualActor(hidden_dim=8, expert_hidden_dim=8)
        observation = torch.randn(4, 28)
        observation[:, 9:12] = 0.0
        output = actor(observation)
        torch.testing.assert_close(output.alignment_cosine, torch.zeros(4, 1))
        self.assertTrue(torch.isfinite(output.gated_residual_action).all())
        self.assertTrue(torch.isfinite(output.trust_gate).all())


if __name__ == "__main__":
    unittest.main()
