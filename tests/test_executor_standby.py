import unittest
import numpy as np
import torch

from chapter3_bser.online.executor_standby import compute_standby_target, normalized_weights, apply_standby_action, standby_controller


class ExecutorStandbyTests(unittest.TestCase):
    def test_equilateral_three_point_center(self):
        target = compute_standby_target([9, 9, 9], [[0, 0, 1], [2, 0, 1], [1, np.sqrt(3), 1]])
        np.testing.assert_allclose(target, [1, np.sqrt(3)/3, 1], atol=1e-8)

    def test_duplicate_and_identical_searchers(self):
        np.testing.assert_allclose(compute_standby_target([0, 0, 0], [[1, 2, 3]]*3), [1, 2, 3])
        np.testing.assert_allclose(compute_standby_target([0, 0, 0], [[1, 2, 3], [1, 2, 3], [10, 2, 3]]), [1, 2, 3], atol=1e-7)

    def test_collinear_median_not_centroid(self):
        np.testing.assert_allclose(compute_standby_target([0, 0, 0], [[0, 1, 1], [1, 1, 1], [10, 1, 1]]), [1, 1, 1], atol=1e-7)

    def test_weighted_median_and_unreliable_weights(self):
        points = [[0, 0, 1], [1, 0, 1], [9, 0, 1]]
        np.testing.assert_allclose(compute_standby_target([0, 0, 0], points, [0, 0, 1]), points[2])
        for weights in ([0, 0, 0], [-1, 2, 3], [1, np.nan, 3], None):
            np.testing.assert_allclose(normalized_weights(weights), [1/3]*3)

    def test_finite_and_convex_hull_and_invalid_input(self):
        self.assertTrue(np.isfinite(compute_standby_target([0, 0, 0], [[1e300, 0, 0], [0, 1e300, 0], [-1e300, 0, 0]])).all())
        rng = np.random.default_rng(19)
        for _ in range(20):
            points = rng.normal(size=(3, 3))
            target = compute_standby_target([0, 0, 0], points)
            self.assertTrue(np.isfinite(target).all())
            self.assertTrue(np.all(target <= points.max(axis=0)+1e-8))
            self.assertTrue(np.all(target >= points.min(axis=0)-1e-8))
        with self.assertRaises(ValueError):
            compute_standby_target([0, 0, 0], [[np.nan]*3]*3)

    def test_disabled_and_found_action_are_object_identity(self):
        actions = torch.randn(4, 3)
        self.assertIs(apply_standby_action(actions, None, None, enabled=False), actions)
        self.assertIs(apply_standby_action(actions, None, None, enabled=True, target_found=True), actions)

    def test_active_only_replaces_executor_and_preserves_dtype(self):
        actions = torch.arange(12, dtype=torch.float32).reshape(4, 3)/12
        before = actions.clone()
        result = apply_standby_action(actions, [0, 0, 0], [10, -10, 1], enabled=True)
        self.assertTrue(torch.equal(actions, before))
        self.assertTrue(torch.equal(result[:3], actions[:3]))
        self.assertEqual(result.dtype, actions.dtype)
        np.testing.assert_array_equal(result[3].numpy(), [1, -1, .5])
        np.testing.assert_array_equal(standby_controller([1, 1, 1], [1, 1, 1]), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
