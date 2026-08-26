import inspec
import unittes

import torch

from chapter3_bser.models.prrac.phase_routed_actor import PhaseRoutedResidualActor
from chapter3_bser.models.prrac.phase_twin_critic import PhaseCritic


class PRRACInformationBoundaryTests(unittest.TestCase):
    def test_actor_has_single_28d_observation_input_and_no_prior_composition(self):
        signature = inspect.signature(PhaseRoutedResidualActor.forward)
        self.assertEqual(tuple(signature.parameters), ("self", "observation"))
        source = inspect.getsource(PhaseRoutedResidualActor).lower()
        for forbidden in ("_task_target", "target_state", "true_obstacle", "waypoint_prior", "privileged"):
            self.assertNotIn(forbidden, source)
        actor = PhaseRoutedResidualActor(hidden_dim=8, expert_hidden_dim=8)
        self.assertEqual(tuple(actor(torch.zeros(2, 28)).gated_residual_action.shape), (2, 3))

    def test_stage_is_not_part_of_critic_features(self):
        critic = PhaseCritic(hidden_dim=8)
        self.assertEqual(critic.input_dim, 124)
        self.assertEqual(tuple(critic(torch.zeros(2, 124)).shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
