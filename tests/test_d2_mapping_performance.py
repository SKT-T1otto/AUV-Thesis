"""Scalar-reference checks for exact sampling and geometry cache invalidation."""
import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from core.env import MissionCoreEnv
from core.mapping.path_planner import OnlineUnknownMapTaskPlanner


def scalar_segment(planner, start, end):
    start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    if (not np.isfinite(start).all() or not np.isfinite(end).all()
            or np.any(start < 0) or np.any(end < 0)
            or np.any(start > planner._space_size_np) or np.any(end > planner._space_size_np)):
        return False
    step = max(.15, .40 * min(planner.cell_dx, planner.cell_dy,
               planner.cell_dz if planner.cell_dz > planner.eps else min(planner.cell_dx, planner.cell_dy)))
    count = max(1, int(math.ceil(float(np.linalg.norm(end-start)) / step)))
    return all(not bool(planner.known_occupied_mask[planner._cell_from_point_np(start+r*(end-start))].item())
               for r in np.linspace(0., 1., count+1))


def scalar_scan_votes(planner, origins, directions, distances, hits):
    """Original scalar ray votes, independent of the batched implementation."""
    free, occupied = {}, {}
    step = max(.15, .45 * min(planner.cell_dx, planner.cell_dy,
               planner.cell_dz if planner.cell_dz > planner.eps else min(planner.cell_dx, planner.cell_dy)))
    for i, origin in enumerate(origins):
        for j, direction in enumerate(directions):
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                continue
            unit = direction / norm
            distance = max(0., float(distances[i, j]))
            hit = bool(hits[i, j])
            limit = max(0., distance-(.55*step if hit else 0.))
            if limit > 0.:
                visited = set()
                count = max(1, int(math.ceil(limit/step)))
                for value in np.linspace(0., limit, count+1):
                    point = origin + unit*float(value)
                    if np.any(point < 0) or np.any(point > planner._space_size_np):
                        continue
                    visited.add(planner._cell_from_point_np(point))
                for cell in visited:
                    free[cell] = free.get(cell, 0)+1
            if hit:
                endpoint = origin+unit*distance
                if np.all(endpoint >= 0.) and np.all(endpoint <= planner._space_size_np):
                    cell = planner._cell_from_point_np(endpoint)
                    occupied[cell] = occupied.get(cell, 0)+1
    return free, occupied


class MappingPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def planner(self):
        from chapter3_bser.experiments.baselines.common.runtime import _make_base_env
        import json
        from pathlib import Path
        config = json.loads((Path(__file__).resolve().parents[1] / "configs/chapter3/d2_v1/direct_mc_train.json").read_text())
        env = _make_base_env(config)
        self.addCleanup(env.close)
        return env.unwrapped.map_module

    def test_batched_cells_and_segments_match_scalar_on_boundaries_and_random_points(self):
        planner = self.planner()
        rng = np.random.default_rng(4491)
        points = rng.uniform(-1, 21, (500, 3))
        points = np.vstack((points, np.zeros(3), planner._space_size_np,
                            planner._flat_xyz_centers_np[:25]))
        expected = np.array([planner._cell_from_point_np(point) for point in points])
        np.testing.assert_array_equal(planner._cells_from_points_np(points), expected)
        planner.known_occupied_mask.reshape(-1)[::19] = True
        segments = list(zip(points[:150], points[150:300]))
        segments += [(p, p) for p in points[-27:]]
        segments += [(np.array([np.nan, 1., 1.]), np.ones(3))]
        for left, right in segments:
            self.assertEqual(planner._segment_is_free_np(left, right, .4), scalar_segment(planner, left, right))

    def test_ray_vote_multiplicity_and_logodds_match_original_scalar_scan(self):
        planner = self.planner()
        rng = np.random.default_rng(27)
        for step in range(3):
            origins = rng.uniform(.1, 19.9, (4, 3))
            origins[:, 2] = rng.uniform(.5, 6.5, 4)
            directions = np.vstack((rng.normal(size=(24, 3)), np.zeros(3), [1., 0., 0.]))
            distances = rng.uniform(0, 10, (4, len(directions)))
            hits = rng.random(distances.shape) < .35
            free, occupied = scalar_scan_votes(planner, origins, directions, distances, hits)
            expected = planner.occupancy_logodds.clone()
            counts = planner.occupancy_observation_count.clone()
            observed = planner.occupancy_last_observed_step.clone()
            for votes, weight in ((free, planner.occupancy_free_logodds), (occupied, planner.occupancy_occupied_logodds)):
                for cell, count in votes.items():
                    expected[cell] += weight*min(count, 3)
                    counts[cell] += count
                    observed[cell] = step
            expected.clamp_(-planner.occupancy_logodds_clip, planner.occupancy_logodds_clip)
            planner.integrate_obstacle_scan(origins, directions, distances, hits, current_step=step)
            torch.testing.assert_close(planner.occupancy_logodds, expected, rtol=0, atol=0)
            torch.testing.assert_close(planner.occupancy_observation_count, counts, rtol=0, atol=0)
            torch.testing.assert_close(planner.occupancy_last_observed_step, observed, rtol=0, atol=0)

    def test_component_cache_changes_with_geometry_not_probability_revision(self):
        from core.mapping.path_planner import ObstacleAwareTaskMapPlanner, _ONLINE_COMPONENTS_CACHE
        planner = self.planner()
        _ONLINE_COMPONENTS_CACHE.clear()
        expected = planner._component_labels()
        planner.map_revision += 1
        planner.grid_revision += 1
        planner._geodesic_cache = {}
        with patch.object(ObstacleAwareTaskMapPlanner, "_component_labels", side_effect=AssertionError("unnecessary rebuild")):
            actual = planner._component_labels()
        self.assertEqual(actual, expected)
        actual.clear()
        self.assertEqual(planner._component_labels(), expected)
        planner.known_occupied_mask[0] = True
        planner.valid_mask[0] = False
        changed = planner._component_labels()
        self.assertNotEqual(changed, expected)
        planner._geodesic_cache = {}
        reference = ObstacleAwareTaskMapPlanner._component_labels(planner)
        self.assertEqual(changed, reference)

    def test_cached_physical_time_preserves_torch_precision_and_roles(self):
        from core.mapping.path_planner import ObstacleAwareTaskMapPlanner, _online_physical_time
        planner = self.planner()
        rng = np.random.default_rng(137)
        _online_physical_time.cache_clear()
        for role in ('searcher', 'executor', 'Exec_special'):
            for _ in range(50):
                cells = [tuple(int(rng.integers(n)) for n in planner.grid_size) for _ in range(2)]
                expected = ObstacleAwareTaskMapPlanner._edge_time(planner, *cells, role)
                self.assertEqual(planner._physical_edge_time(*cells, role), expected)
                points = rng.uniform(0., 10., (2, 3))
                expected = ObstacleAwareTaskMapPlanner._continuous_edge_time(planner, *points, role)
                self.assertEqual(planner._continuous_edge_time(*points, role), expected)
                self.assertEqual(planner._continuous_edge_time(*points, role), expected)
        self.assertGreater(_online_physical_time.cache_info().hits, 100)

    def test_segment_cache_uses_current_content_endpoints_and_bounds(self):
        from core.mapping.path_planner import _sampled_online_segment_free
        planner = self.planner()
        _sampled_online_segment_free.cache_clear()
        planner.known_occupied_mask[:] = False
        left, right = np.array([2., 2., 2.]), np.array([6., 2., 2.])
        self.assertTrue(planner._segment_is_free_np(left, right, .4))
        first = _sampled_online_segment_free.cache_info()
        planner.map_revision += 1
        self.assertTrue(planner._segment_is_free_np(left, right, .4))
        self.assertEqual(_sampled_online_segment_free.cache_info().hits, first.hits+1)
        revision = planner.map_revision
        planner.known_occupied_mask[planner._cell_from_point_np(left)] = True
        self.assertFalse(planner._segment_is_free_np(left, right, .4))
        self.assertEqual(planner.map_revision, revision)
        for a, b in ((left+.01, right), (left, right+.03), (right, left)):
            self.assertEqual(planner._segment_is_free_np(a, b, .4), scalar_segment(planner, a, b))
        planner.known_occupied_mask[:] = False
        self.assertTrue(planner._segment_is_free_np(left, right, .4))
        planner._space_size_np = np.array([5., 20., 10.])
        self.assertFalse(planner._segment_is_free_np(left, right, .4))
        self.assertEqual(_sampled_online_segment_free.cache_info().maxsize, 32768)


if __name__ == "__main__":
    unittest.main()
