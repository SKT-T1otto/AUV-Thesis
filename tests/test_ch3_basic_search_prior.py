"""Small synthetic decision fixtures and one full-horizon integration fixture.

No training or formal manifest generation. Real integration retains the 400-step
task definition; the inherited handoff scene is explicitly not performance data.
"""
import copy
from contextlib import ExitStack
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.ch3_baselines import basic_search_prior as basic, evaluate as ev, provenance as prov
from chapter3_bser.baselines.search_only_allocator import solve_search_only_greedy
from chapter3_bser.objective import build_objective_context, evaluate_objective, cell_detection_probability
from chapter3_bser.online.config import load_phase1b2_config
from chapter3_bser.integration.rmaddpg_bridge import RMADDPGGuidanceBridge
from chapter3_bser.experiments.phase1c_prrac.task_metrics import strict_outcome
from tests.bser_online_test_utils import state_at, shifted_belief, mission_context
from tests.bser_test_utils import synthetic_instance

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "configs/chapter3/hgr_train.json"


def fixture():
    value = json.loads((ROOT / "tests/fixtures/hgr/handoff_manifest.json").read_text(encoding="utf-8"))
    scene = value["scenarios"][0]
    scene.update(scenario_id="basic_prior_synthetic_handoff_interface_only", scenario_split="validation",
                 scenario_role="validation", max_steps=400)
    value["purpose"] = "synthetic bounded interface verification; no performance evidence; original 400-step task rules"
    return value


def forbidden_learning():
    stack = ExitStack()
    for target in (
        "chapter3_bser.online.allocator.solve_joint_greedy",
        "chapter3_bser.greedy_solver.solve_joint_greedy",
        "chapter3_bser.online.allocator.generate_standby_candidates",
        "chapter3_bser.candidate_generator.generate_standby_candidates",
        "core.mapping.map_module.ProbabilisticTaskMapPlanner.plan_executor_standby",
        "core.mapping.path_planner.ObstacleAwareTaskMapPlanner.plan_executor_standby",
        "chapter3_bser.models.hgr.policy.HandoffPolicy.actions",
        "torch.nn.Module._call_impl", "torch.randn", "torch.randn_like",
        "torch.distributions.Normal.sample", "torch.distributions.Normal.rsample",
        "torch.optim.SGD.step", "torch.optim.Adam.step",
    ):
        stack.enter_context(patch(target, side_effect=AssertionError("forbidden joint/learning call: " + target)))
    return stack


class SearchOnlyTests(unittest.TestCase):
    def allocator(self, state=None):
        state = state or state_at(0)
        return basic.SearchPriorAllocator(state.agents[3].position)

    def test_math_and_response_weight_independence(self):
        state, config, generated, context = synthetic_instance()
        first = solve_search_only_greedy(generated.search_candidates, context)
        expected = float(np.sum(context.belief * cell_detection_probability(first.selected, context)))
        self.assertAlmostEqual(first.objective, expected, places=14)
        weights = {k: np.full_like(v, 0.00001) for k, v in context.response_weight_by_id.items()}
        changed = replace(context, response_weight_by_id=weights)
        second = solve_search_only_greedy(generated.search_candidates, changed)
        self.assertEqual([c.key for c in first.selected], [c.key for c in second.selected])
        self.assertEqual(first.objective, second.objective)
        self.assertIsNone(first.standby)

    def test_full_none_standby_and_partial_preserve_routes(self):
        allocator = self.allocator()
        with forbidden_learning():
            current = allocator.allocate(state_at(0))
            self.assertEqual({a.agent_id for a in current.search_assignments}, {0, 1, 2})
            state = shifted_belief(20)
            proposed, ok, reason = allocator.allocate_partial(state, current, affected_searcher_ids=(0,), trigger_reason="TEST")
            self.assertTrue(ok, reason)
            for old in current.search_assignments:
                if old.agent_id != 0:
                    self.assertIs(next(a for a in proposed.search_assignments if a.agent_id == old.agent_id), old)
            self.assertAlmostEqual(proposed.objective_value, allocator.score(state, proposed.search_assignments), places=14)
            self.assertEqual(proposed.executor_assignment, current.executor_assignment)

    def test_response_tau_does_not_change_full_or_partial_decisions(self):
        results = []
        for tau in (0.0001, 1e9):
            allocator = self.allocator()
            allocator.config = copy.deepcopy(allocator.config)
            allocator.config["objective"]["tau_executor"] = tau
            full = allocator.allocate(state_at(0))
            partial, ok, _ = allocator.allocate_partial(shifted_belief(20), full, affected_searcher_ids=(1,), trigger_reason="TEST")
            self.assertTrue(ok)
            results.append((full.allocation_sha256, partial.allocation_sha256))
        self.assertEqual(*results)

    def test_controller_initialization_replanning_found_and_legal_handoff(self):
        state = state_at(0)
        allocator = self.allocator(state)
        controller = basic.SearchPriorController(load_phase1b2_config(), allocator)
        with forbidden_learning():
            controller.initialize(state, mission_context(state))
            # Force a genuine existing stale event by moving Searcher 0 to its waypoint.
            old = controller.current_allocation
            waypoint = next(a.waypoint for a in old.search_assignments if a.agent_id == 0)
            moved = replace(state, step=20, agents=tuple(replace(a, position=waypoint) if a.agent_id == 0 else a for a in state.agents))
            result = controller.step(moved, mission_context(moved))
            self.assertGreater(allocator.counts["partial_allocation_attempts"], 0)
            self.assertAlmostEqual(result.allocation.objective_value, allocator.score(moved, result.allocation.search_assignments))
            found = replace(shifted_belief(21), target_found=True)
            waiting = controller.step(found, mission_context(found, executor_knows_target=False))
            self.assertEqual(waiting.allocation.executor_assignment.target_region, allocator.anchor)
            bridge = RMADDPGGuidanceBridge()
            guidance = bridge.compile_guidance(waiting.allocation, found, mission_context(found))
            self.assertFalse(guidance.executor_assignment.execution_request)
            received = replace(found, step=22)
            context = mission_context(received, executor_knows_target=True)
            delivered = controller.step(received, context)
            self.assertEqual(delivered.allocation.executor_assignment.target_region, context.executor_navigation_target)
            self.assertTrue(delivered.allocation.search_frozen)
            self.assertTrue(bridge.compile_guidance(delivered.allocation, received, context).executor_assignment.execution_request)

    def test_anchor_fallback_does_not_use_belief_peak_or_reset_anchor(self):
        allocator = self.allocator()
        current = allocator.allocate(state_at(0))
        with patch.object(allocator.execution, "assign_belief_peak", side_effect=AssertionError("belief fallback")):
            for state in (shifted_belief(20), replace(shifted_belief(21), target_found=True)):
                result, ok, _ = allocator.allocate_partial(state, current, executor_affected=True, trigger_reason="EXECUTOR_INVALID")
                self.assertTrue(ok)
                self.assertEqual(result.executor_assignment.target_region, allocator.anchor)
                unreachable = replace(result.executor_assignment, reachable=False, path=(), failure_reason="fixture_unreachable")
                with patch.object(allocator.execution, "_assignment", return_value=unreachable):
                    retried = allocator.reassign_invalid_executor(state, current)
                    self.assertEqual(retried.executor_assignment.target_region, allocator.anchor)
                    self.assertFalse(retried.executor_assignment.reachable)
        with self.assertRaises(AttributeError):
            allocator.anchor = (0, 0, 0)

    def test_missing_candidates_and_partial_recovery_are_explicit(self):
        allocator = self.allocator()
        with patch.object(allocator, "_candidates", return_value=()):
            empty = allocator.allocate(state_at(0))
            self.assertEqual(empty.search_assignments, ())
            self.assertEqual(empty.executor_assignment.target_region, allocator.anchor)
            _, ok, reason = allocator.allocate_partial(state_at(1), empty, affected_searcher_ids=(0,), trigger_reason="TEST")
            self.assertFalse(ok)
            self.assertEqual(reason, "ATOMIC_REJECT_MISSING_SEARCH_ROUTE")
        recovered, ok, reason = allocator.allocate_partial(state_at(2), empty, affected_searcher_ids=(0,), trigger_reason="TEST")
        self.assertTrue(ok, reason)
        self.assertEqual([a.agent_id for a in recovered.search_assignments], [0])

    def test_hidden_truth_is_not_a_decision_input(self):
        class World:
            def __init__(self, target):
                self.hidden_target = target
                self.receiver_view = state_at(0)
        decisions = []
        for target in ((0, 0, 0), (99, 99, 99)):
            world = World(target)
            allocator = self.allocator(world.receiver_view)
            full = allocator.allocate(world.receiver_view)
            invalid = allocator.reassign_invalid_executor(world.receiver_view, full)
            decisions.append((full.allocation_sha256, invalid.allocation_sha256))
        self.assertEqual(*decisions)


class EvaluatorContractTests(unittest.TestCase):
    def test_reference_resolution_and_invalid_config(self):
        config, resolved = ev.resolve_config(REFERENCE, 1, 12729)
        self.assertEqual(resolved["reference_kind"], "default_config_reference")
        self.assertEqual(resolved["method"], basic.METHOD)
        self.assertEqual(resolved["reference_training_method"], "ch3_hgr")
        self.assertTrue(resolved["common_task_conditions"]["environment_config"]["use_residual_prior"])
        self.assertFalse(resolved["effective_environment_config"]["pse_use_standby"])
        self.assertNotIn("pse_use_standby", resolved["common_task_conditions"]["environment_config"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"bad.json"
            for changes in ({"max_steps": 8}, {"profile": "S00_STATIC_CLEAR"}, {"early_discovery": {"enabled": True}}, {"collision_terminal_reward": -1}):
                prov.write_json(path, {**config, **changes})
                with self.assertRaises(ValueError):
                    ev.resolve_config(path, 1, 12729)

    def test_manifest_guards_and_output_protection(self):
        config, _ = ev.resolve_config(REFERENCE, 1, 12729)
        sources = prov.capture_sources()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root/"fixture.json"
            original = fixture()
            prov.write_json(manifest, original)
            selected = ev.select_manifest(config, 1, 12729, manifest, sources)
            self.assertEqual(selected["scenarios"], original["scenarios"])
            for invalid in (None, root/"missing.json"):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    ev.select_manifest(config, 1, 12729, invalid, sources)
            for changes in ({"scenario_seed": True}, {"scenario_split": "train"}, {"max_steps": 399}, {"scenario_profile": "S00_STATIC_CLEAR"}):
                prov.write_json(manifest, {"scenarios": [{**original["scenarios"][0], **changes}]})
                with self.assertRaises(ValueError):
                    ev.select_manifest(config, 1, 12729, manifest, sources)
            prov.write_json(manifest, {"scenarios": original["scenarios"] * 2})
            with self.assertRaises(ValueError):
                ev.select_manifest(config, 1, 12729, manifest, sources)
            occupied = root/"collision_terminal/keep"
            occupied.mkdir(parents=True)
            keep = occupied/"keep.txt"
            keep.write_text("unchanged")
            with self.assertRaises(FileExistsError):
                ev.evaluate(REFERENCE, manifest, occupied, episodes=1)
            self.assertEqual(keep.read_text(), "unchanged")
            with self.assertRaises(ValueError):
                ev.evaluate(REFERENCE, manifest, root/"wrong_protocol", episodes=1)

    def test_module_help_and_windows_launcher_exit_code(self):
        result = subprocess.run([sys.executable, "-B", "-m", "tools.ch3_baselines.evaluate", "--help"],
                                cwd=ROOT, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--reference-training-config", result.stdout)
        self.assertNotIn("--resume", result.stdout)
        if os.name != "nt":
            return  # Windows-only launcher; Linux shell validation is separate.
        environment = os.environ.copy()
        environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment["PATH"]
        with tempfile.TemporaryDirectory(prefix="basic launcher space ") as temporary:
            for arg, expected in (("--help", 0), ("--unknown-option", 2)):
                command = '"' + str(ROOT/"scripts/run_ch3_basic_prior_eval.bat") + '" ' + arg
                result = subprocess.run(command, shell=True, cwd=temporary, env=environment,
                                        capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)

    def test_independent_source_change_detection(self):
        before = prov.capture_sources()
        prov.require_unchanged(before, prov.capture_sources())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("tools/ch3_baselines/a.py", "scripts/run_ch3_basic_prior_eval.bat", "scripts/linux/run_ch3_basic_prior_eval.sh"):
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original")
            first = prov.baseline_source_identity(root)
            (root/"tools/ch3_baselines/a.py").write_text("changed")
            with self.assertRaises(ValueError):
                prov.require_unchanged(first, prov.baseline_source_identity(root))

    def test_partial_and_three_terminal_outcome_summaries(self):
        rows = []
        for reason in ("success", "obstacle_collision", "timeout"):
            row = dict(task_protocol="collision_terminal_v1", termination_reason=reason,
                success=reason == "success", safe_success=reason == "success", found=reason == "success",
                collision_episode=reason == "obstacle_collision", terminated=True, truncated=False,
                first_collision_agent_ids=[3] if reason == "obstacle_collision" else [],
                first_collision_phase="Search", episode_length=400, wall_seconds=1,
                team_discounted_return=-2.0, team_undiscounted_return=-3.0)
            self.assertNotEqual(strict_outcome(row), "INCOMPLETE")
            rows.append(row)
        final = ev.summary(rows, 3, finalized=True, wall_seconds=3, actual_steps=1200)
        self.assertEqual([final[k] for k in ("n_success", "n_collision_failure", "n_timeout")], [1, 1, 1])
        partial = ev.summary(rows, 4, finalized=False, wall_seconds=3, actual_steps=1201)
        self.assertIsNone(partial["safe_success_rate"])
        self.assertIsNone(partial["mean_team_discounted_return"])
        self.assertFalse(partial["evaluation_complete"])
        with self.assertRaises(ValueError):
            strict_outcome({**rows[0], "collision_episode": True})


class RealIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_horizon_fixture_zero_residual_without_actor_and_outputs(self):
        before = prov.capture_sources()
        with tempfile.TemporaryDirectory(prefix="basic-prior-interface-") as temporary:
            root = Path(temporary)
            path = root/"synthetic_manifest.json"
            prov.write_json(path, fixture())
            output = root/"collision_terminal/synthetic_only"
            with forbidden_learning():
                result = ev.evaluate(REFERENCE, path, output, episodes=1, seed=12729)
            self.assertTrue(result["evaluation_complete"])
            self.assertLessEqual(result["actual_environment_steps"], 400)
            row = json.loads((output/"episodes.json").read_text())[0]
            diagnostic = json.loads((output/"controller_diagnostics.json").read_text())["episodes"][0]
            self.assertIsNotNone(row["found_step"])
            # The authoritative handoff_step is publication time, not delivery.
            self.assertEqual(row["handoff_event_step"], row["found_step"])
            self.assertGreater(row["handoff_decision_step"], row["found_step"])
            self.assertEqual(row["handoff_decision_step"], row["executor_target_received_step"])
            self.assertEqual(diagnostic["residual_steps_checked"], row["episode_length"])
            for key in ("actor_forward_calls", "action_sampling_calls", "optimizer_update_count", "residual_action_max_abs", "physical_residual_acceleration_max_abs"):
                self.assertEqual(diagnostic[key], 0)
            self.assertGreater(diagnostic["physical_prior_acceleration_max_abs"], 0)
            identity = json.loads((output/"evaluation_identity.json").read_text())
            self.assertEqual(identity["sources_before"]["production"], identity["sources_after"]["production"])
            self.assertNotIn("pairing_sha256", identity)
            self.assertTrue((output/"run_console.log").read_text().strip())
        prov.require_unchanged(before, prov.capture_sources())

    def test_real_collision_stops_before_capture_and_never_resumes(self):
        config, _ = ev.resolve_config(REFERENCE, 1, 12729)
        with forbidden_learning():
            runtime = basic.BasicSearchPriorRuntime(config, fixture()["scenarios"][0], seed=12729)
            try:
                physics = runtime.env.unwrapped
                self.assertFalse(physics.pse_use_standby)
                # Exercise the otherwise eligible background standby path.
                physics.step_count = physics.pse_standby_start_step
                physics._update_pse_executor_standby(force=True)
                physics.step_count = 0
                # Synthetic physical-contact fault, separate from any evaluation
                # scenario: retain prior/flow and the strict collision algorithm.
                physics.obstacles = [{"center": physics._agent_pos[3].tolist(), "size": [1, 1, 1]}]
                physics.ground_truth_obstacles = copy.deepcopy(physics.obstacles)
                physics._build_obstacle_tensors()
                with patch.object(physics, "_update_capture", side_effect=AssertionError("capture after collision")):
                    runtime.advance()
                self.assertTrue(runtime.terminal)
                self.assertEqual(runtime.env.get_episode_result()["termination_reason"], "obstacle_collision")
                self.assertEqual(runtime.env.reward_accounting.last["team_reward"], -2.0)
                with self.assertRaises(RuntimeError):
                    runtime.advance()
            finally:
                runtime.close()

    def test_program_failure_is_not_timeout(self):
        with tempfile.TemporaryDirectory(prefix="basic-prior-fault-") as temporary:
            root = Path(temporary)
            path = root/"synthetic_manifest.json"
            prov.write_json(path, fixture())
            output = root/"collision_terminal/fault"
            with patch.object(basic.BasicSearchPriorRuntime, "advance", side_effect=RuntimeError("injected program fault")):
                with self.assertRaisesRegex(RuntimeError, "injected program fault"):
                    ev.evaluate(REFERENCE, path, output, episodes=1)
            summary = json.loads((output/"summary.json").read_text())
            self.assertFalse(summary["evaluation_complete"])
            self.assertEqual(summary["n_timeout"], 0)
            self.assertEqual(summary["n_missing_or_abnormal_episodes"], 1)
            self.assertIsNone(summary["safe_success_rate"])
            self.assertIn("basic_prior_synthetic", (output/"evaluation_failure.json").read_text())


if __name__ == "__main__":
    unittest.main()
