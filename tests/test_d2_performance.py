"""Performance-only contracts: fresh safety geometry and durable bounded logs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


class JournalTests(unittest.TestCase):
    def test_incremental_records_and_atomic_array_materialization(self):
        from chapter3_bser.experiments.d2_performance.logging import JsonlJournal, read_records
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_metrics.jsonl"
            journal = JsonlJournal(path)
            rows = [dict(optimizer_update=i, actor_loss=[i / 7, -i / 8]) for i in range(7)]
            for row in rows[:3]:
                journal.append(row)
            prefix = path.read_bytes()
            for row in rows[3:]:
                journal.append(row)
            self.assertTrue(path.read_bytes().startswith(prefix))
            self.assertEqual(list(read_records(path)), rows)
            self.assertEqual(len(journal), 7)
            journal.materialize()
            self.assertEqual(json.loads(path.with_suffix(".json").read_text()), rows)
            self.assertEqual(list(read_records(path)), rows)
            with self.assertRaises(FileExistsError):
                JsonlJournal(path)

    def test_crash_tail_is_ignored_but_completed_corruption_is_not(self):
        from chapter3_bser.experiments.d2_performance.logging import read_records
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.jsonl"
            path.write_bytes(b'{"step": 1}\n{"step":')
            self.assertEqual(list(read_records(path)), [dict(step=1)])
            path.write_bytes(b'{"step": 1}\ninvalid\n')
            with self.assertRaises(ValueError):
                list(read_records(path))

    def test_nonfinite_input_does_not_damage_previous_records(self):
        from chapter3_bser.experiments.d2_performance.logging import JsonlJournal, read_records
        with tempfile.TemporaryDirectory() as tmp:
            journal = JsonlJournal(Path(tmp) / "episodes.jsonl")
            journal.append(dict(step=1))
            with self.assertRaises(ValueError):
                journal.append(dict(loss=float("nan")))
            self.assertEqual(list(read_records(journal.path)), [dict(step=1)])


class RuntimeOptionsTests(unittest.TestCase):
    def test_cuda_adapter_preserves_original_cpu_noise_draws(self):
        from core.algorithms.noise import OUNoise
        from chapter3_bser.experiments.d2_performance.compute import CpuExplorationNoise
        reference = OUNoise(3, sigma=.18)
        adapted = CpuExplorationNoise(3, sigma=.18)
        adapted.to(device='cuda')
        self.assertEqual(adapted.state.device.type, 'cpu')
        torch.manual_seed(127)
        expected = [reference.sample().clone() for _ in range(20)]
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(127)
        actual = [adapted.sample().clone() for _ in range(20)]
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)

    def test_explicit_settings_do_not_follow_cuda_availability(self):
        from chapter3_bser.experiments.d2_performance.options import performance_options
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(performance_options({}), dict(learner_device="cpu", cpu_threads=1))
        for settings in (dict(cpu_threads=True), dict(cpu_threads=0), dict(cpu_threads=1.5),
                         dict(learner_device="auto"), dict(batch_size=16)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                performance_options(dict(performance=settings))

    def test_prepare_freezes_explicit_compute_for_each_arm(self):
        from chapter3_bser.experiments.d2_suite_v1.plan import prepare, read
        from tests.test_d2_suite import inputs
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            root = directory / 'collision_terminal/compute'
            plan = prepare(root, **inputs(directory), learner_device='cuda', cpu_threads=2)
            for job in plan['jobs']:
                config = read(root / job['config'])
                if job['arm'] == 'D2':
                    continue
                self.assertEqual(config['performance']['cpu_threads'], 2)
                self.assertEqual(config['performance']['learner_device'],
                                 'cuda' if job['arm'] in ('D2_B2', 'D2_B3') else 'cpu')
            self.assertFalse((root / 'jobs').exists())


class LiveGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_live_projection_tracks_occupancy_positions_without_graph_build(self):
        from types import SimpleNamespace
        from core.mapping.planning_state import extract_planning_state
        from chapter3_bser.experiments.baselines.common.runtime import BaselineMissionRuntime
        from chapter3_bser.experiments.baselines.common.model import build_model
        from chapter3_bser.experiments.d2_performance.planning import D2PlanningViews
        from chapter3_bser.experiments.safe_search_v1.public_geometry import PublicGeometry
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs/chapter3/d2_v1/direct_mc_train.json").read_text())
        scene = json.loads((root / "configs/scenarios/e0_equivalence/M20_MOVING_UNKNOWN_MULTI.json").read_text())["scenarios"][0]
        runtime = BaselineMissionRuntime(config, scene, seed=117)
        try:
            model = build_model(config)
            for _ in range(3):
                if runtime.terminal:
                    break
                runtime.advance(model, explore=False)
                reference = extract_planning_state(runtime.env)
                # A new view rules out reuse of a forced-refresh full state.
                views = D2PlanningViews(runtime, runtime.planning_views.clearance)
                with patch("core.mapping.planning_state.build_planning_graph", side_effect=AssertionError("expensive graph")):
                    actual = views.live()
                    self.assertIs(actual, views.live())
                self.assertEqual(actual.step, reference.step)
                self.assertEqual(actual.agents, reference.agents)
                np.testing.assert_array_equal(actual.occupancy.occupied_mask, reference.occupancy.occupied_mask)
                self.assertFalse(actual.occupancy.occupied_mask.flags.writeable)
                left, right = PublicGeometry(actual, views.clearance), PublicGeometry(reference, views.clearance)
                np.testing.assert_array_equal(left.occupied_lower, right.occupied_lower)
                for a in reference.agents:
                    self.assertEqual(left.segment_free(a.position, a.current_navigation_target),
                                     right.segment_free(a.position, a.current_navigation_target))
                # Same step with changed occupancy must invalidate the content cache.
                planner = runtime.env.unwrapped.map_module
                original = planner.known_occupied_mask.clone()
                try:
                    planner.known_occupied_mask.reshape(-1)[0] = ~planner.known_occupied_mask.reshape(-1)[0]
                    changed = views.live()
                    self.assertNotEqual(bool(changed.occupancy.occupied_mask[0]), bool(actual.occupancy.occupied_mask[0]))
                finally:
                    planner.known_occupied_mask.copy_(original)
        finally:
            runtime.close()


if __name__ == "__main__":
    unittest.main()
