"""Runner input, paired-population and prefix contracts; no simulator or training."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from chapter3_bser.experiments.safe_search_v1 import run_paired as runner


def scene(index):
    return dict(scenario_id=f"scene_{index}", scenario_seed=100+index,
                scenario_split="validation", scenario_profile="M20_MOVING_UNKNOWN_MULTI", max_steps=400)


class RequestTests(unittest.TestCase):
    def test_original_indices_and_seed_are_not_subset_indices(self):
        manifest = dict(scenarios=[scene(i) for i in range(30)])
        original = copy.deepcopy(manifest)
        selected = runner.validate_manifest(manifest, dict(profile="M20_MOVING_UNKNOWN_MULTI", max_steps=400), "0,3,28", 12729)
        self.assertEqual([i for i, _ in selected], [0, 3, 28])
        self.assertEqual([12729+i for i, _ in selected], [12729, 12732, 12757])
        selected[0][1]["max_steps"] = 100
        self.assertEqual(manifest, original)

    def test_full_horizon_requires_explicit_flag(self):
        base = dict(baseline="B0_search_prior", variants="V0,V4", seed=12729, verify_v0=True)
        for steps, full in ((400, False), (100, True), (0, False), (401, True), (True, False)):
            with self.subTest(steps=steps, full=full), self.assertRaises(ValueError):
                runner.validate_request(**base, steps=steps, full_episodes=full)
        self.assertEqual(runner.validate_request(**base, steps=400, full_episodes=True), ["V0", "V4"])

    def test_invalid_indices_variants_and_unselected_manifest_tail_rejected(self):
        for value in ("0,0", "-1", "3", "", "hello"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.select_indices(value, 3)
        for value in ("V0,V0", "V6", "", "V0,V1,"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.parse_variants(value)
        manifest = dict(scenarios=[scene(0), scene(1)])
        manifest["scenarios"][1]["scenario_split"] = "train"
        with self.assertRaisesRegex(ValueError, "validation"):
            runner.validate_manifest(manifest, dict(profile="M20_MOVING_UNKNOWN_MULTI", max_steps=400), "0", 12729)

    def test_scenario_horizon_and_seed_bounds(self):
        config = dict(profile="M20_MOVING_UNKNOWN_MULTI", max_steps=400)
        manifest = dict(scenarios=[scene(0), scene(1)])
        with self.assertRaisesRegex(ValueError, "seed"):
            runner.validate_manifest(manifest, config, "all", 2**32-1)
        manifest["scenarios"][1]["max_steps"] = 100
        with self.assertRaisesRegex(ValueError, "horizon"):
            runner.validate_manifest(manifest, config, "0", 12729)

    def test_new_output_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "scene.json"
            source.write_text("{}", encoding="utf-8")
            for output in (root, root / "outputs" / "new", root / "3090结果" / "new", source):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    runner.output_directory(output, [source], root=root)
            output = root / "runs" / "new"
            self.assertEqual(runner.output_directory(output, [source], root=root), output)
            output.mkdir(parents=True)
            (output / "retained.txt").write_text("evidence", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                runner.output_directory(output, [source], root=root)

    def test_cli_defaults_are_bounded_and_manifest_required(self):
        args = runner.parser().parse_args(["--manifest", "scene.json", "--output-dir", "runs/new"])
        self.assertEqual(args.steps, 100)
        self.assertFalse(args.full_episodes)
        self.assertEqual(args.seed, 12729)
        self.assertEqual(args.episode_indices, "all")


class FakeRuntime:
    def __init__(self, end=None, collision=False, found=None):
        self.step = 0
        self.end = end
        self.collision = collision
        self.found_at = found
        self.closed = False
        self.env = SimpleNamespace(get_episode_result=self.result)

    @property
    def terminal(self):
        return self.end is not None and self.step >= self.end

    @property
    def found_step(self):
        return self.found_at if self.found_at is not None and self.step >= self.found_at else None

    def result(self):
        return dict(termination_reason=("obstacle_collision" if self.collision else "timeout") if self.terminal else None,
                    first_collision_step=self.end if self.terminal and self.collision else None)

    def advance(self):
        if self.terminal:
            raise AssertionError("cannot advance terminal runtime")
        self.step += 1
        return dict(reward=self.step)

    def controller_diagnostics(self):
        return dict(optimizer_update_count=0)

    def close(self):
        self.closed = True


class FakeTap:
    def __init__(self, service):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def drain(self):
        return dict(counts=[], samples=[])


class FakeObserver:
    def __init__(self):
        self.steps = 0

    def observe(self, before, after, queries, result):
        self.steps += 1
        return dict(before=before, after=after, queries=queries, result=result)

    def result(self):
        return dict(counts=dict(pre_found_exposure_steps=self.steps), agents={})


def fake_dependencies():
    return dict(QueryTap=FakeTap, TransitionObserver=FakeObserver, TravelCostService=object,
        capture_runtime=lambda runtime: dict(step=runtime.step),
        physical_signature=lambda runtime, record: runner.digest(dict(step=runtime.step, record=record)),
        complete_episode_row=lambda runtime, wall: dict(terminal_step=runtime.step))


class ExecutionTests(unittest.TestCase):
    def test_prefix_cutoff_is_not_timeout_or_complete_episode(self):
        runtime = FakeRuntime()
        with tempfile.TemporaryDirectory() as temporary:
            signatures, row = runner.execute_episode(lambda: runtime, Path(temporary), steps=3,
                observed=True, deps=fake_dependencies())
            self.assertEqual(len(signatures), 4)
            self.assertEqual(row["stop_reason"], "budget_cutoff")
            self.assertIsNone(row["complete_episode_row"])
            self.assertFalse(row["full_episode_completed"])
            trace = (Path(temporary) / "step_trace.jsonl").read_text().splitlines()
            self.assertEqual(len(trace), 3)
        self.assertTrue(runtime.closed)

    def test_terminal_stops_early_and_collision_precedes_same_step_discovery(self):
        runtime = FakeRuntime(end=2, collision=True, found=2)
        with tempfile.TemporaryDirectory() as temporary:
            _, row = runner.execute_episode(lambda: runtime, Path(temporary), steps=100,
                observed=True, deps=fake_dependencies())
        self.assertEqual(row["physical_steps"], 2)
        self.assertTrue(row["pre_found_collision"])
        self.assertEqual(row["complete_episode_row"], dict(terminal_step=2))
        self.assertTrue(runtime.closed)

    def test_observation_and_reference_signatures_equal(self):
        with tempfile.TemporaryDirectory() as temporary:
            plain, _ = runner.execute_episode(FakeRuntime, Path(temporary), steps=3,
                observed=False, deps=fake_dependencies())
            observed, _ = runner.execute_episode(FakeRuntime, Path(temporary), steps=3,
                observed=True, deps=fake_dependencies())
        self.assertEqual(plain, observed)

    def test_failure_closes_runtime(self):
        runtime = FakeRuntime()
        deps = fake_dependencies()
        def fail(_runtime):
            raise RuntimeError("observer failure")
        deps["capture_runtime"] = fail
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(RuntimeError, "observer failure"):
            runner.execute_episode(lambda: runtime, Path(temporary), steps=3, observed=True, deps=deps)
        self.assertTrue(runtime.closed)

    def test_paired_metrics_join_original_indices_and_keep_cutoff_rates_null(self):
        base = dict(physical_steps=100, terminal=False, stop_reason="budget_cutoff", pre_found_exposure_steps=100,
            searcher_motion_stall_proxy_agent_steps=30, searcher_hold_agent_steps=0,
            wall_seconds=1.0, found_within_budget=False, pre_found_collision=False)
        rows = [dict(base, variant="V0", original_episode_index=28, scenario_id="scene_28"),
                dict(base, variant="V4", original_episode_index=28, scenario_id="scene_28", found_within_budget=True)]
        summary = runner.summarize(rows, ["V0", "V4"], 1, False)
        self.assertIsNone(summary["variants"]["V4"]["found_rate"])
        self.assertEqual(summary["variants"]["V4"]["found_within_budget_fraction"], 1.0)
        self.assertEqual(summary["paired_vs_v0"]["V4"]["found_within_budget_mean_delta"], 1.0)
        self.assertEqual(summary["paired_vs_v0"]["V4"]["pairs"][0]["original_episode_index"], 28)
        self.assertIsNone(summary["performance_passed"])
        self.assertTrue(summary["all_requested_runs_recorded"])

    def test_full_rates_wait_for_every_requested_episode(self):
        row = dict(variant="V0", physical_steps=18, terminal=True, stop_reason="obstacle_collision",
            original_episode_index=3, scenario_id="scene_3", pre_found_exposure_steps=18,
            searcher_motion_stall_proxy_agent_steps=0, searcher_hold_agent_steps=0,
            wall_seconds=1.0, found_within_budget=False, pre_found_collision=True)
        self.assertIsNone(runner.summarize([row], ["V0"], 2, True)["variants"]["V0"]["found_rate"])
        complete = runner.summarize([row], ["V0"], 1, True)["variants"]["V0"]
        self.assertEqual(complete["found_rate"], 0.0)
        self.assertEqual(complete["pre_found_collision_rate"], 1.0)
        self.assertTrue(complete["full_episode_evaluation_complete"])


if __name__ == "__main__":
    unittest.main()
