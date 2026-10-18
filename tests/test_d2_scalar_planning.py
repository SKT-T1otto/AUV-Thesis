"""Independent Torch/scalar references for online planner arithmetic."""
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from core.env import MissionCoreEnv
from core.mapping.path_planner import ObstacleAwareTaskMapPlanner, OnlineUnknownMapTaskPlanner


def reference_physical(planner, left, right, role):
    return ObstacleAwareTaskMapPlanner._edge_time(planner, left, right, role)


def reference_cost(planner, left, right, role):
    physical = reference_physical(planner, left, right, role)
    probability = .5 * (float(planner.occupancy_probability[left].item())
                        + float(planner.occupancy_probability[right].item()))
    entropy = 0.0
    if 0.0 < probability < 1.0:
        entropy = -(probability * math.log(probability)
                    + (1.0 - probability) * math.log(1.0 - probability)) / math.log(2.0)
    return (physical + planner.occupancy_unknown_cost_weight * entropy
            + planner.occupancy_risk_cost_weight * probability * physical)


class ScalarPlanningTests(unittest.TestCase):
    def planner(self, dtype=torch.float32):
        return OnlineUnknownMapTaskPlanner(space_size=(7.3, 5.9, 4.1), grid_size=(3, 3, 3),
                                          z_range=(.27, 3.81), dtype=dtype)

    def test_every_cell_pair_and_role_matches_original_torch_bits(self):
        rng = np.random.default_rng(1249)
        for dtype in (torch.float32, torch.float64):
            planner = self.planner(dtype)
            planner.occupancy_probability.copy_(torch.as_tensor(rng.random((3, 3, 3)), dtype=dtype))
            planner.occupancy_probability[0, 0, 0] = 0
            planner.occupancy_probability[-1, -1, -1] = 1
            for role in ("searcher", "executor", "Exec_special"):
                for left in np.ndindex(planner.grid_size):
                    for right in np.ndindex(planner.grid_size):
                        self.assertEqual(planner._physical_edge_time(left, right, role).hex(),
                                         reference_physical(planner, left, right, role).hex())
                        self.assertEqual(planner._edge_time(left, right, role).hex(),
                                         reference_cost(planner, left, right, role).hex())

    def test_in_place_and_replaced_tensors_are_read_live_and_rng_is_unchanged(self):
        planner = self.planner()
        rng_before = torch.get_rng_state().clone()
        for replace_tensor in (False, True):
            if replace_tensor:
                planner.xyz_centers = planner.xyz_centers.clone()
                planner.occupancy_probability = planner.occupancy_probability.clone()
            planner.xyz_centers[1, 1, 1] += .012345
            planner.occupancy_probability[1, 1, 1] = .13
            before = planner.xyz_centers.clone(), planner.occupancy_probability.clone()
            for role in ("searcher", "executor"):
                self.assertEqual(planner._edge_time((0, 0, 0), (1, 1, 1), role),
                                 reference_cost(planner, (0, 0, 0), (1, 1, 1), role))
            for actual, expected in zip((planner.xyz_centers, planner.occupancy_probability), before):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)

    def test_astar_cell_sequence_cost_and_expansion_count_match_scalar_reference(self):
        planner = self.planner()
        planner.known_occupied_mask[1, 1, :] = True
        planner.valid_mask[1, 1, :] = False
        for role in ("searcher", "executor"):
            for start, goal in (((0, 0, 0), (2, 2, 2)), ((0, 2, 0), (2, 0, 2)), ((2, 1, 0), (2, 1, 0))):
                planner._geodesic_cache = {}
                with patch.object(OnlineUnknownMapTaskPlanner, "_physical_edge_time", reference_physical), \
                     patch.object(OnlineUnknownMapTaskPlanner, "_edge_time", reference_cost):
                    expected = planner._core_cell_path(start, goal, role)
                planner._geodesic_cache = {}
                self.assertEqual(planner._core_cell_path(start, goal, role), expected)


if __name__ == "__main__":
    unittest.main()
