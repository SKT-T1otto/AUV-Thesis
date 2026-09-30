"""Target-observation telemetry preserves original calls and real update stamps."""
import random
import importlib.util
from types import SimpleNamespace
import unittest

from chapter3_bser.experiments.safe_search_v1.search_coverage import CoverageObserver


class Planner:
    def __init__(self):
        self.target_last_observed_step = [-1] * 8
        self.flat_valid_mask = [True] * 8
        self.target_revisit_half_life_steps = 30.0
        self.runtime_step = 0
        self.belief_enabled = True
        self.calls = []
        self.token = object()

    def update_belief_negative(self, cells, *, sensor_ranges=None, fail=False):
        self.calls.append((cells, sensor_ranges))
        if fail:
            raise LookupError("original update failure")
        if self.belief_enabled:
            for i in cells:
                if self.flat_valid_mask[i]:
                    self.target_last_observed_step[i] = self.runtime_step
        return self.token


def fixture():
    planner = Planner()
    env = SimpleNamespace(map_module=planner, step_count=0, task_found=False, found_step=None)
    return SimpleNamespace(env=SimpleNamespace(unwrapped=env)), env, planner


def advance(env, planner, step, cells):
    env.step_count = planner.runtime_step = step
    return planner.update_belief_negative(cells)


class CoverageTests(unittest.TestCase):
    def test_original_once_same_arguments_result_rng_and_restore(self):
        runtime, env, planner = fixture()
        original = planner.update_belief_negative
        positions, ranges = [0, 1], [2.0]
        rng = random.getstate()
        with CoverageObserver(runtime) as observer:
            env.step_count = planner.runtime_step = 1
            result = planner.update_belief_negative(positions, sensor_ranges=ranges)
            self.assertIs(result, planner.token)
            self.assertIs(planner.calls[0][0], positions)
            self.assertIs(planner.calls[0][1], ranges)
            self.assertEqual(observer.drain()["observed_cell_indices"], [0, 1])
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(planner.update_belief_negative, original)
        self.assertNotIn("update_belief_negative", vars(planner))
        self.assertEqual(random.getstate(), rng)

    def test_repeated_static_cells_and_half_life_revisit(self):
        runtime, env, planner = fixture()
        with CoverageObserver(runtime) as observer:
            advance(env, planner, 1, [0, 1])
            self.assertEqual(observer.drain()["new_cell_observations"], 2)
            advance(env, planner, 2, [0, 1])
            repeated = observer.drain()
            self.assertEqual(repeated["repeated_cell_observations"], 2)
            self.assertEqual(repeated["effective_observation_steps"], 0)
            advance(env, planner, 31, [0, 1])
            self.assertEqual(observer.drain()["aged_revisit_cell_observations"], 0)
            advance(env, planner, 61, [0, 1])
            self.assertEqual(observer.drain()["aged_revisit_cell_observations"], 2)
        result = observer.result()
        self.assertEqual(result["unique_observed_cells"], 2)
        self.assertEqual(result["new_cell_observations"], 2)
        self.assertEqual(result["repeated_cell_observations"], 6)
        self.assertEqual(result["effective_observation_steps"], 2)

    def test_union_multiple_calls_same_step_not_false_repeated_observation(self):
        runtime, env, planner = fixture()
        with CoverageObserver(runtime) as observer:
            advance(env, planner, 1, [0, 1])
            planner.update_belief_negative([1, 2])
            row = observer.drain()
        self.assertEqual(row["observed_cell_indices"], [0, 1, 2])
        self.assertEqual(row["observation_calls"], 2)
        self.assertEqual(row["observation_update_steps"], 1)
        self.assertEqual(row["new_cell_observations"], 3)
        self.assertEqual(row["repeated_cell_observations"], 0)

    def test_initialization_counts_cells_but_never_physical_steps(self):
        runtime, env, planner = fixture()
        planner.target_last_observed_step[0] = 0
        with CoverageObserver(runtime) as observer:
            initial = observer.result()
            self.assertEqual(initial["initial_observed_cells"], 1)
            self.assertEqual(initial["effective_observation_steps"], 0)
            advance(env, planner, 1, [0, 1])
            row = observer.drain()
        self.assertEqual(row["new_cell_observations"], 1)
        self.assertEqual(observer.result()["unique_observed_cells"], 2)

    def test_found_transition_included_then_postfound_updates_excluded(self):
        runtime, env, planner = fixture()
        with CoverageObserver(runtime) as observer:
            advance(env, planner, 1, [0, 1])
            env.task_found, env.found_step = True, 1
            self.assertEqual(observer.drain()["pre_found_exposure_steps"], 1)
            advance(env, planner, 2, [2, 3])
            row = observer.drain()
        self.assertEqual(row["observed_cell_indices"], [])
        self.assertEqual(row["pre_found_exposure_steps"], 0)
        self.assertEqual(observer.result()["unique_observed_cells"], 2)

    def test_collision_terminal_transition_without_update_adds_no_coverage(self):
        runtime, env, planner = fixture()
        with CoverageObserver(runtime) as observer:
            env.step_count = 1
            row = observer.drain()
        self.assertEqual(row["step"], 1)
        self.assertEqual(row["pre_found_exposure_steps"], 1)
        self.assertEqual(row["observed_cell_indices"], [])
        self.assertEqual(row["effective_observation_steps"], 0)

    def test_disabled_belief_does_not_fabricate_footprint(self):
        runtime, env, planner = fixture()
        planner.belief_enabled = False
        with CoverageObserver(runtime) as observer:
            advance(env, planner, 1, [0, 1])
            self.assertEqual(observer.drain()["observed_cell_indices"], [])

    def test_actual_valid_stamps_not_obstacle_map_or_positions_are_counted(self):
        runtime, env, planner = fixture()
        planner.flat_valid_mask[1] = False
        with CoverageObserver(runtime) as observer:
            advance(env, planner, 1, [0, 1])
            self.assertEqual(observer.drain()["observed_cell_indices"], [0])

    def test_missing_footprint_is_honestly_unavailable(self):
        runtime, env, planner = fixture()
        del planner.target_last_observed_step
        with CoverageObserver(runtime) as observer:
            env.step_count = 1
            row = observer.drain()
        self.assertIsNone(row["observed_cell_indices"])
        self.assertIsNone(observer.result()["effective_observation_steps"])
        self.assertFalse(observer.result()["available"])

    def test_original_exception_unchanged_and_tap_restored(self):
        runtime, env, planner = fixture()
        original = planner.update_belief_negative
        with self.assertRaisesRegex(LookupError, "original update failure"):
            with CoverageObserver(runtime):
                env.step_count = planner.runtime_step = 1
                planner.update_belief_negative([], fail=True)
        self.assertEqual(planner.update_belief_negative, original)
        self.assertEqual(len(planner.calls), 1)

    def test_observer_failure_cannot_replace_original_success(self):
        runtime, env, planner = fixture()
        with CoverageObserver(runtime) as observer:
            env.step_count = 1
            # The fake forgot its context update; telemetry must not alter return.
            result = planner.update_belief_negative([0])
            self.assertIs(result, planner.token)
            self.assertFalse(observer.drain()["available"])
            self.assertIsNone(observer.result()["unique_observed_cells"])

    def test_existing_instance_override_restored_by_identity(self):
        runtime, env, planner = fixture()
        override = planner.update_belief_negative
        planner.update_belief_negative = override
        with CoverageObserver(runtime):
            pass
        self.assertIs(planner.update_belief_negative, override)

    def test_nested_tap_rejected_without_breaking_outer(self):
        runtime, env, planner = fixture()
        with CoverageObserver(runtime) as observer:
            with self.assertRaisesRegex(RuntimeError, "already active"):
                with CoverageObserver(runtime):
                    pass
            advance(env, planner, 1, [0])
            self.assertEqual(observer.drain()["observed_cell_indices"], [0])


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
class ExactAttributionTests(unittest.TestCase):
    def fixture(self, positions, mismatch=False):
        import torch

        class TensorPlanner:
            def __init__(self):
                self.dtype, self.device = torch.float32, "cpu"
                self.cell_dx = self.cell_dy = 1.0
                self.eps = 1e-8
                self.flat_xyz_centers = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                                                      [5., 0., 0.], [20., 0., 0.]])
                self.flat_valid_mask = torch.ones(4, dtype=torch.bool)
                self.target_last_observed_step = torch.full((4,), -1)
                self.target_revisit_half_life_steps = 30.0
                self.belief_enabled = True
                self.runtime_step = 0

            def update_belief_negative(self, search_positions, sensor_ranges=None):
                for index, position in enumerate(search_positions):
                    radius = float(sensor_ranges[min(index, len(sensor_ranges) - 1)])
                    distance = torch.linalg.vector_norm(self.flat_xyz_centers - position, dim=1)
                    self.target_last_observed_step[(distance <= max(radius, self.eps)) &
                                                  self.flat_valid_mask] = self.runtime_step
                if mismatch:
                    self.target_last_observed_step[3] = self.runtime_step
                return self.target_last_observed_step

        planner = TensorPlanner()
        env = SimpleNamespace(map_module=planner, step_count=0, task_found=False, found_step=None,
                              _agent_pos=torch.tensor(positions, dtype=torch.float32))
        return SimpleNamespace(env=SimpleNamespace(unwrapped=env)), env, planner

    def test_exact_static_overlap_counted_each_update(self):
        runtime, env, planner = self.fixture([[0, 0, 0], [0, 0, 0], [5, 0, 0]])
        with CoverageObserver(runtime) as observer:
            for step in (1, 2):
                env.step_count = planner.runtime_step = step
                planner.update_belief_negative(env._agent_pos, sensor_ranges=[0.25])
                row = observer.drain()
        self.assertEqual(row["per_agent_observed_cell_indices"], {"0": [0], "1": [0], "2": [2]})
        result = observer.result()
        self.assertEqual(result["repeated_cell_observations"], 2)
        self.assertEqual(result["inter_agent_overlap"]["redundant_agent_cell_observations"], 2)
        self.assertAlmostEqual(result["inter_agent_overlap"]["redundant_agent_cell_fraction"], 1 / 3)

    def test_disjoint_has_zero_overlap_and_short_range_extension(self):
        runtime, env, planner = self.fixture([[0, 0, 0], [1, 0, 0], [5, 0, 0]])
        with CoverageObserver(runtime) as observer:
            env.step_count = planner.runtime_step = 1
            planner.update_belief_negative(env._agent_pos, sensor_ranges=[0.0, 0.25])
            row = observer.drain()
        self.assertEqual(row["observed_cell_indices"], [0, 1, 2])
        self.assertEqual(observer.result()["inter_agent_overlap"]["redundant_agent_cell_fraction"], 0)

    def test_union_mismatch_keeps_actual_coverage_but_nulls_attribution(self):
        runtime, env, planner = self.fixture([[0, 0, 0], [1, 0, 0], [5, 0, 0]], mismatch=True)
        with CoverageObserver(runtime) as observer:
            env.step_count = planner.runtime_step = 1
            planner.update_belief_negative(env._agent_pos, sensor_ranges=[0.25])
            row = observer.drain()
        self.assertTrue(row["available"])
        self.assertEqual(row["observed_cell_indices"], [0, 1, 2, 3])
        self.assertIsNone(row["per_agent_observed_cell_indices"])
        self.assertIsNone(observer.result()["inter_agent_overlap"])
        self.assertIn("differs from actual", observer.result()["attribution_errors"][0])

    def test_wrong_agent_order_never_gets_canonical_agent_labels(self):
        runtime, env, planner = self.fixture([[0, 0, 0], [1, 0, 0], [5, 0, 0]])
        with CoverageObserver(runtime) as observer:
            env.step_count = planner.runtime_step = 1
            planner.update_belief_negative(env._agent_pos.flip(0), sensor_ranges=[0.25])
            observer.drain()
        self.assertTrue(observer.result()["available"])
        self.assertIsNone(observer.result()["per_agent_coverage"])


if __name__ == "__main__":
    unittest.main()
