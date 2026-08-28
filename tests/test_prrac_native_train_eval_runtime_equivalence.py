from types import SimpleNamespace
import unittest
from unittest import mock

from chapter3_bser.events.event_types import EventDetection
from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.experiments.phase1c_prrac import train_phase1c_prrac as trainer
from chapter3_bser.experiments.phase1c_prrac.execution_continuity import NavigationMode
from chapter3_bser.experiments.phase1c_prrac.execution_continuity import event_detector, planner
from chapter3_bser.experiments.phase1c_prrac import runtime_factory
from chapter3_bser.online.types import (
    BSERActionAssignment,
    ExecutorAssignment,
    InitialBSERAllocation,
    OnlineAllocation,
    SearchAssignment,
)
from tests.execution_continuity_test_support import FakeTravelCostService, state as execution_state


def _detection():
    return EventDetection((), 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, True, ())


def _allocation():
    search = tuple(
        SearchAssignment(index, f"search-{index}", (float(index), 0.0, 0.0), (), 0.0)
        for index in range(3)
    )
    executor = ExecutorAssignment(3, (0.0, 0.0, 0.0), (), 0.0, "LEGACY", True)
    return OnlineAllocation(search, executor, 1.0, 0.5, 0.0, "LEGACY")


class _FakeLegacyController:
    def __init__(self, config):
        self.config = config
        self.detector = SimpleNamespace(dynamic_public_target_enabled=True)
        self.current_allocation = _allocation()
        self.execution_target = None
        self.replan_count = 0

    def initialize(self, state, context):
        self.current_allocation = _allocation()
        return InitialBSERAllocation(self.current_allocation, ())

    def step(self, state, context):
        return BSERActionAssignment(
            int(state.step), False, (), self.current_allocation, (), "LEGACY_SEARCH", _detection()
        )


def _state(step, *, executor_position=(0.0, 0.0, 0.0), reachable=()):
    value = execution_state(executor_position=executor_position)
    value.step = int(step)
    value.travel_service = FakeTravelCostService(reachable)
    return value


def _context(step, *, found=False, target=None):
    return SimpleNamespace(
        step=int(step), target_found=bool(found), finder_id=0,
        executor_knows_target=bool(found), handoff_step=step if found else None,
        mission_complete=False, executor_navigation_target=target,
        target_known_by_agent=(found,) * 4, searcher_finished_flags=(False,) * 3,
    )


class PRRACNativeTrainEvalRuntimeEquivalenceTests(unittest.TestCase):
    def assert_plan_equal(self, left, right):
        for field in (
            "navigation_mode", "semantic_target", "navigation_endpoint", "reachable",
            "safe_hold", "path_cell_indices", "planning_cost", "path",
            "preserved_from_previous",
        ):
            self.assertEqual(getattr(left, field), getattr(right, field), field)

    def assert_result_equal(self, left, right):
        self.assertEqual(left.decision_reason, right.decision_reason)
        self.assertEqual(left.allocation.executor_assignment, right.allocation.executor_assignment)

    def test_identical_public_state_sequence_has_identical_native_b1_semantics(self):
        config = trainer._load_config(
            trainer.ROOT / "configs/chapter3/bser_phase1c_prrac_s1_train.json"
        )
        phase1b_config = {"online": {"state_refresh_interval": 20}}
        service_factory = lambda state: state.travel_service
        with mock.patch.object(runtime_factory, "OnlineBSERController", _FakeLegacyController), mock.patch.object(planner, "TravelCostService", side_effect=service_factory), mock.patch.object(event_detector, "TravelCostService", side_effect=service_factory):
            train_controller = trainer._build_episode_controller(phase1b_config, config)
            eval_controller = evaluator._build_episode_controller(
                phase1b_config,
                config,
                execution_variant="B1_ATOMIC_LAST_VALID",
                runtime_integration_mode="native",
                checkpoint_runtime_revision=config["execution_runtime_revision"],
            )

            search_state = _state(0)
            search_context = _context(0)
            train_initial = train_controller.initialize(search_state, search_context)
            eval_initial = eval_controller.initialize(search_state, search_context)
            self.assertEqual(train_initial.allocation, eval_initial.allocation)
            self.assertIsNone(train_controller.current_plan)
            self.assertIsNone(eval_controller.current_plan)

            exact_target = (3.0, 0.0, 0.0)
            exact_state = _state(20, reachable=(exact_target,))
            exact_context = _context(20, found=True, target=exact_target)
            train_exact = train_controller.step(exact_state, exact_context)
            eval_exact = eval_controller.step(exact_state, exact_context)
            self.assert_plan_equal(train_controller.current_plan, eval_controller.current_plan)
            self.assertEqual(train_controller.current_plan.navigation_mode, NavigationMode.EXACT_PUBLIC_TARGET)
            self.assert_result_equal(train_exact, eval_exact)

            train_controller.current_plan = None
            eval_controller.current_plan = None
            hold_state = _state(21, executor_position=(0.5, 0.0, 0.0))
            hold_context = _context(21, found=True, target=(9.0, 0.0, 0.0))
            train_hold = train_controller.step(hold_state, hold_context)
            eval_hold = eval_controller.step(hold_state, hold_context)
            self.assert_plan_equal(train_controller.current_plan, eval_controller.current_plan)
            self.assertEqual(train_controller.current_plan.navigation_mode, NavigationMode.SAFE_HOLD)
            self.assertEqual(train_controller.current_plan.navigation_endpoint, (0.5, 0.0, 0.0))
            self.assertEqual(train_controller.current_plan.path, ())
            self.assert_result_equal(train_hold, eval_hold)

            refreshed_target = (3.0, 0.0, 0.0)
            refreshed_state = _state(40, reachable=(refreshed_target,))
            refreshed_context = _context(40, found=True, target=refreshed_target)
            train_refreshed = train_controller.step(refreshed_state, refreshed_context)
            eval_refreshed = eval_controller.step(refreshed_state, refreshed_context)
            self.assert_plan_equal(train_controller.current_plan, eval_controller.current_plan)
            self.assertEqual(train_controller.current_plan.navigation_mode, NavigationMode.EXACT_PUBLIC_TARGET)
            self.assert_result_equal(train_refreshed, eval_refreshed)

            shifted_target = (9.0, 0.0, 0.0)
            last_valid_state = _state(60, reachable=(refreshed_target,))
            last_valid_context = _context(60, found=True, target=shifted_target)
            train_last = train_controller.step(last_valid_state, last_valid_context)
            eval_last = eval_controller.step(last_valid_state, last_valid_context)
            self.assert_plan_equal(train_controller.current_plan, eval_controller.current_plan)
            self.assertEqual(train_controller.current_plan.navigation_mode, NavigationMode.LAST_VALID_ROUTE)
            self.assertTrue(train_controller.current_plan.preserved_from_previous)
            self.assert_result_equal(train_last, eval_last)


if __name__ == "__main__":
    unittest.main()
