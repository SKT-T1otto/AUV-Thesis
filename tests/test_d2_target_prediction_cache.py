"""Exact scalar reference, mutation and fallback coverage for prediction reuse."""
import ast
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from core.env import target_motion as motion


class PredictionCacheTests(unittest.TestCase):
    def test_physical_integrator_and_collision_helpers_frozen(self):
        names = {"TargetState", "advance_target_state", "segment_aabb_first_hit",
                 "_boundary_first_hit", "_expanded_boxes", "_vec3"}
        source = ast.parse(Path(motion.__file__).read_text(encoding="utf-8"))
        nodes = [node for node in source.body if getattr(node, "name", None) in names]
        tree = ast.Module(body=nodes, type_ignores=[])
        digest = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
        # Exact definitions from the fixed old source, not a regenerated fixture.
        self.assertEqual(digest, "9ec9e3186568b72fc00f984a33d91a72e4bcc0cc9863c6557367a0d28f695ecd")

    def setUp(self):
        motion._prediction_step.cache_clear()
        self.addCleanup(motion._prediction_step.cache_clear)
        self.state = motion.TargetState([2, 3, 4], [1.7, .4, -.3], 8,
                                       "constant_velocity_reflect_v1",
                                       metadata={"source": [1, (2, 3)]})
        self.kw = dict(dt=.25, bounds=[20., 20., 20.],
                       obstacles=[dict(center=[10., 3., 3.], size=[2., 2., 2.])])

    def reference(self, state, steps, **kw):
        result = motion.TargetState.from_payload(state.to_payload())
        for _ in range(steps):
            result = motion.advance_target_state(result, **kw)
        return result

    def exact(self, actual, expected):
        self.assertEqual(actual.position.tobytes(), expected.position.tobytes())
        self.assertEqual(actual.velocity.tobytes(), expected.velocity.tobytes())
        self.assertEqual(actual.to_payload(), expected.to_payload())

    def test_prefixes_reflections_and_overlapping_start(self):
        for steps in (1, 10, 100, 500, 2, 111):
            self.exact(motion.predict_target_state(self.state, steps, **self.kw),
                       self.reference(self.state, steps, **self.kw))
        info = motion._prediction_step.cache_info()
        self.assertEqual(info.misses, 500)
        self.assertGreater(info.hits, 200)
        advanced = self.reference(self.state, 40, **self.kw)
        self.exact(motion.predict_target_state(advanced, 100, **self.kw),
                   self.reference(advanced, 100, **self.kw))
        self.assertEqual(motion._prediction_step.cache_info().misses, 500)
        obstacle_state = motion.TargetState([2, 3, 3], [2, 0, 0], 0,
                                           "constant_velocity_reflect_v1")
        bounced = motion.predict_target_state(obstacle_state, 20, **self.kw)
        self.exact(bounced, self.reference(obstacle_state, 20, **self.kw))
        self.assertEqual(bounced.reflection_count, 1)
        self.assertEqual(bounced.velocity[0], -2.)
        corner_state = motion.TargetState([2, 2, 2], [2, 2, 2], 0,
                                         "constant_velocity_reflect_v1")
        kw = dict(self.kw, obstacles=[])
        self.exact(motion.predict_target_state(corner_state, 100, **kw),
                   self.reference(corner_state, 100, **kw))

    def test_mutations_and_physical_context(self):
        expected = self.reference(self.state, 100, **self.kw)
        result = motion.predict_target_state(self.state, 100, **self.kw)
        result.position[:] = 0
        result.velocity[:] = 0
        result.metadata["source"].append("changed")
        self.exact(motion.predict_target_state(self.state, 100, **self.kw), expected)
        variants = [dict(dt=.2), dict(bounds=[18., 19., 20.]),
                    dict(clearance=.15), dict(max_reflections=5),
                    dict(obstacles=[dict(center=[9., 3., 3.], size=[1., 2., 2.])])]
        for variant in variants:
            kw = dict(self.kw, **variant)
            before = motion._prediction_step.cache_info().misses
            self.exact(motion.predict_target_state(self.state, 100, **kw),
                       self.reference(self.state, 100, **kw))
            self.assertGreater(motion._prediction_step.cache_info().misses, before)
        self.kw["obstacles"][0]["size"][0] = 3.
        self.state.position[0] += .125
        self.state.metadata["new"] = "preserved"
        self.state.obstacle_layout_id = "changed-layout"
        self.exact(motion.predict_target_state(self.state, 100, **self.kw),
                   self.reference(self.state, 100, **self.kw))

    def test_generator_and_custom_metadata_fallback(self):
        kw1 = dict(self.kw, obstacles=iter(self.kw["obstacles"]))
        kw2 = dict(self.kw, obstacles=iter(self.kw["obstacles"]))
        self.exact(motion.predict_target_state(self.state, 50, **kw1),
                   self.reference(self.state, 50, **kw2))
        self.assertEqual(motion._prediction_step.cache_info().currsize, 0)
        self.state.metadata["array"] = np.array([1., 2.])
        result = motion.predict_target_state(self.state, 50, **self.kw)
        expected = self.reference(self.state, 50, **self.kw)
        np.testing.assert_array_equal(result.position, expected.position)
        np.testing.assert_array_equal(result.metadata["array"], expected.metadata["array"])
        self.assertEqual(motion._prediction_step.cache_info().currsize, 0)

    def test_validation_and_reflection_limit_preserved(self):
        for invalid in (-1, True, 1.5, 501):
            with self.assertRaises(ValueError):
                motion.predict_target_state(self.state, invalid, **self.kw)
        self.exact(motion.predict_target_state(self.state, 0, dt=-1,
                                              bounds=None, obstacles=object()), self.state)
        for call in (motion.predict_target_state, self.reference):
            with self.assertRaisesRegex(RuntimeError, "reflection limit exceeded"):
                call(self.state, 100, **dict(self.kw, max_reflections=0))
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            motion.predict_target_state(self.state, 1, **dict(self.kw,
                obstacles=[dict(center=[10, 3, 3], size=[-1, 2, 2])]))

    def test_rng_signed_zero_and_instrumented_fallback(self):
        self.state.velocity[2] = -0.0
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(7921)
        rng = np.random.get_state()
        expected = self.reference(self.state, 100, **self.kw)
        self.exact(motion.predict_target_state(self.state, 100, **self.kw), expected)
        after = np.random.get_state()
        self.assertEqual(rng[0], after[0])
        np.testing.assert_array_equal(rng[1], after[1])
        self.assertEqual(rng[2:], after[2:])
        with patch.object(motion, "advance_target_state", wraps=motion.advance_target_state) as spy:
            self.exact(motion.predict_target_state(self.state, 100, **self.kw), expected)
            self.assertEqual(spy.call_count, 100)
        self.assertEqual(motion._prediction_step.cache_info().maxsize, 16384)


if __name__ == "__main__":
    unittest.main()
