from tests.beds_test_support import navigation, parameters
"""Pure public-graph/control-boundary tests. No simulator episodes/checkpoints."""
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import subprocess
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from chapter3_bser.controllers.path_tracker import PathTracker
from chapter3_bser.experiments.phase1c_prrac.beds import BEDSEpisodeAdapter, resolve_beds, write_diagnostics
from chapter3_bser.experiments.phase1c_prrac.standby_diagnostics import StandbyDiagnostics
from chapter3_bser.integration.control_context import AgentAssignmentContextV1, ExecutorAssignmentContextV1, BSERControlContextV1
from chapter3_bser.integration.rmaddpg_bridge import get_tracking_targets
from chapter3_bser.online.safe_executor_standby import SafeStandbyNavigation, StandbyNavigationParameters, resolve_safe_route
from chapter3_bser.online.executor_standby import compute_standby_target
from core.mapping.travel_cost_service import TravelCostService
from core.env.uav_env import UAVEnv
from tests.bser_test_utils import synthetic_state


def public_state():
    old = synthetic_state()
    scale = lambda point: tuple(float(v)*4 for v in point)
    graph = replace(old.planning_graph, cell_centers=old.grid.cell_centers*4,
        endpoint_connectors=tuple(replace(e, point=scale(e.point)) for e in old.planning_graph.endpoint_connectors))
    return replace(old, grid=replace(old.grid, cell_centers=old.grid.cell_centers*4,
        origin=scale(old.grid.origin), spacing=scale(old.grid.spacing)), planning_graph=graph,
        agents=tuple(replace(a, position=scale(a.position), current_navigation_target=scale(a.current_navigation_target)) for a in old.agents))




def guidance(state):
    assignments = tuple(AgentAssignmentContextV1(a.agent_id, a.role, 'search', 'original',
        a.position, (), a.position, a.position, False, True, False) for a in state.agents)
    executor = ExecutorAssignmentContextV1(3, 'original', state.agents[3].position, (),
        state.agents[3].position, state.agents[3].position, False, True, False)
    return BSERControlContextV1('bser.control_context.v1', 'v1:test', 'test', state.step,
        'EXECUTION' if state.target_found else 'SEARCH', assignments, executor, 'test')




def adapter(state, *, enabled=True, diagnostics=True):
    value = BEDSEpisodeAdapter({'executor_standby': {'enabled': enabled, 'gain': .5},
        'standby_diagnostics_enabled': diagnostics}, 'public-fixture', 0)
    if enabled:
        value.navigation = navigation(state)
    return value


class SafeStandbyTests(unittest.TestCase):
    def test_planner_query_exception_is_fail_safe_hold(self):
        state = public_state()
        service = Mock()
        service.query.side_effect = RuntimeError('public endpoint unavailable')
        value = navigation(state, service_factory=lambda state: service)
        value.prepare(state)
        self.assertTrue(value.hold)
        self.assertIsNone(value.safe_target)
        self.assertEqual(value.path, ())
        self.assertIn('PUBLIC_PLANNER_FAILURE', value.details['target_update_reason'])
        np.testing.assert_array_equal(value.apply_residual_safety(np.ones((4, 3)), state)[3], np.zeros(3))

    def test_early_only_adapter_matches_previous_commit(self):
        root = Path(__file__).resolve().parents[1]
        source = subprocess.check_output(['git', '-c', 'safe.directory='+root.as_posix(), 'show',
            '4f11e65e41147e5ced953be0c70c5b0eab13dde4:chapter3_bser/experiments/phase1c_prrac/beds.py'], cwd=root).decode()
        previous = ModuleType('old_beds')
        exec(compile(source, 'old_beds.py', 'exec'), previous.__dict__)
        config = {'early_discovery': {'enabled': True, 'lambda': .01}, 'executor_standby': {'enabled': False}}
        ranking = SimpleNamespace(ranking_diagnostics=[dict(step=0, candidate_count=2, mean_time_discount=.7,
                                                          top_candidate_changed=True)], candidate_diagnostics=[])
        old = previous.BEDSEpisodeAdapter(config, 'same', 0, copy.deepcopy(ranking))
        new = BEDSEpisodeAdapter(config, 'same', 0, copy.deepcopy(ranking))
        state = public_state()
        original, actions = guidance(state), torch.randn(4, 3)
        self.assertIs(old.prepare_guidance(original, state), new.prepare_guidance(original, state))
        self.assertIs(old.before_action(actions, state), new.before_action(actions, state))
        for name in ('early_discovery_diagnostics.csv', 'early_discovery_candidates.csv'):
            self.assertEqual(old.payload()[name], new.payload()[name])

    def test_prior_slow_radius_decreases_actual_navigation_acceleration(self):
        runtime = SimpleNamespace(_agent_pos=torch.zeros(4, 3), _agent_acc=torch.zeros(4, 3),
            _agent_vel=torch.zeros(4, 3), eps=1e-6, _v_xy_max=torch.ones(4), _v_z_max=torch.ones(4),
            _a_xy_max=torch.ones(4), _a_z_max=torch.ones(4), prior_slow_radius_xy=2.4,
            prior_slow_radius_z=1.2, prior_kv_xy=1., prior_kv_z=1.)
        norms = []
        for distance in (4., 1.2, .3, 0.):
            runtime._nav_targets = torch.tensor([[distance, 0., 0.]]*4)
            norms.append(float(torch.linalg.vector_norm(UAVEnv._compute_waypoint_prior_acc(runtime)[3])))
        self.assertTrue(all(a > b for a, b in zip(norms, norms[1:])))

    def test_navigation_binding_uses_only_unknown_map_and_runtime_thresholds(self):
        from core.mapping.path_planner import OnlineUnknownMapTaskPlanner
        planner = Mock(spec=OnlineUnknownMapTaskPlanner)
        planner.planner_obstacle_clearance = .4
        runtime = SimpleNamespace(map_module=planner, use_residual_prior=True, _prior_strength=[1]*4,
            safe_dist=1.6, prior_slow_radius_xy=2.4, prior_slow_radius_z=1.2, executor_hold_radius=.8)
        value = adapter(public_state())
        value.bind_navigation(SimpleNamespace(unwrapped=runtime), .75)
        self.assertEqual(value.navigation.parameters.hold_radius, .8)
        self.assertEqual(value.navigation.parameters.safe_dist, runtime.safe_dist)
        runtime.map_module = object()
        with self.assertRaisesRegex(ValueError, 'unknown-map'):
            value.bind_navigation(SimpleNamespace(unwrapped=runtime), .75)

    def test_disabled_guidance_and_actions_are_identity(self):
        state = public_state()
        value = adapter(state, enabled=False)
        original, actions = guidance(state), torch.randn(4, 3)
        self.assertIs(value.prepare_guidance(original, state), original)
        self.assertIs(value.before_action(actions, state), actions)

    def test_early_source_is_frozen_at_requested_commit(self):
        root = Path(__file__).resolve().parents[1]
        path = 'chapter3_bser/online/early_discovery.py'
        original = subprocess.check_output(['git', '-c', 'safe.directory='+root.as_posix(), 'show',
            '4f11e65e41147e5ced953be0c70c5b0eab13dde4:'+path], cwd=root).decode().replace('\r\n', '\n')
        self.assertEqual((root/path).read_text(encoding='utf-8'), original)

    def test_raw_median_remains_geometric_median(self):
        np.testing.assert_allclose(compute_standby_target([8, 9, 1], [[0, 0, 1], [2, 0, 1], [1, np.sqrt(3), 1]]),
                                   [1, np.sqrt(3)/3, 1], atol=1e-8)

    def test_nonconnector_raw_is_repaired_by_existing_public_graph(self):
        state = public_state()
        raw = (5.7, 5.7, 4.)
        service = TravelCostService(state)
        self.assertFalse(service.query(state.agents[3].position, raw, state.agents[3]).reachable)
        target, route, reason = resolve_safe_route(state, raw, 1.6, lambda a, b: True)
        self.assertEqual(target, (6., 6., 4.))
        self.assertEqual(reason, 'PUBLIC_ENDPOINT_REPAIR')
        self.assertTrue(route.reachable)

    def test_occupied_raw_target_is_never_selected(self):
        state = public_state()
        valid = np.asarray(state.planning_graph.valid_mask).copy()
        valid[4] = False
        graph = replace(state.planning_graph, valid_mask=valid)
        state = replace(state, planning_graph=graph)
        target, _, _ = resolve_safe_route(state, (6., 6., 4.), 1.6,
            lambda a, b: not np.array_equal(b, [6., 6., 4.]))
        self.assertNotEqual(target, (6., 6., 4.))

    def test_no_truth_or_future_information_is_needed(self):
        state = public_state()  # No env, target truth or obstacle boxes exist here.
        value = navigation(state)
        value.prepare(state)
        self.assertTrue(value.path)
        self.assertFalse(hasattr(state, 'true_target_position'))

    def test_failed_planner_holds_and_never_p_commands_raw_target(self):
        state = public_state()
        value = adapter(state)
        value.navigation = navigation(state, segment_clear=lambda a, b: False)
        changed = value.prepare_guidance(guidance(state), state)
        self.assertTrue(changed.executor_assignment.hold_state)
        self.assertEqual(get_tracking_targets(changed)[3], state.agents[3].position)
        actions = torch.ones(4, 3)
        applied = value.before_action(actions, state)
        self.assertTrue(torch.equal(applied[3], torch.zeros(3)))
        self.assertTrue(torch.equal(applied[:3], actions[:3]))

    def test_planned_path_passes_to_real_path_tracker(self):
        state = public_state()
        value = navigation(state)
        with patch.object(value.tracker, 'tracking_target', wraps=value.tracker.tracking_target) as track:
            value.prepare(state)
        self.assertIsInstance(value.tracker, PathTracker)
        self.assertEqual(track.call_args.args[2], value.path)
        self.assertGreater(len(value.path), 1)

    def test_local_tracking_target_not_final_raw_target(self):
        state = public_state()
        value = navigation(state)
        service = TravelCostService(state)
        query = service.query(state.agents[3].position, (6., 6., 4.), state.agents[3])
        query = replace(query, path_points=np.array([(10., 10., 4.), (10., 6., 4.), (6., 6., 4.)]))
        with patch('chapter3_bser.online.safe_executor_standby.resolve_safe_route', return_value=((6., 6., 4.), query, 'TEST_ROUTE')):
            value.prepare(state)
        self.assertEqual(value.tracking_target, (10., 6., 4.))
        self.assertNotEqual(value.tracking_target, value.safe_target)
        self.assertNotEqual(value.tracking_target, value.raw_target)

    def test_small_target_shift_retains_route_and_tracker(self):
        state = public_state()
        fresh = Mock(return_value=state)
        value = navigation(state)
        value.state_factory = fresh
        value.prepare(state)
        route = value.path
        for step in (1, 2, 10, 20):
            changed = replace(state, step=step, agents=tuple(replace(a, position=tuple(np.asarray(a.position)+.01))
                if a.agent_id < 3 else a for a in state.agents))
            value.prepare(changed)
        self.assertEqual(fresh.call_count, 1)
        self.assertIs(value.path, route)

    def test_update_requires_interval_and_material_shift(self):
        state = public_state()
        value = navigation(state)
        value.state_factory = Mock(return_value=state)
        value.prepare(state)
        shifted = tuple(replace(a, position=tuple(np.asarray(a.position)+[3, 0, 0])) if a.agent_id < 3 else a for a in state.agents)
        value.prepare(replace(state, step=9, agents=shifted))
        self.assertEqual(value.state_factory.call_count, 1)
        value.prepare(replace(state, step=10, agents=shifted))
        self.assertEqual(value.state_factory.call_count, 2)

    def test_live_map_invalidation_holds_without_replan_every_step(self):
        state = public_state()
        value = navigation(state)
        value.prepare(state)
        value.state_factory = Mock(return_value=state)
        value.segment_clear = lambda a, b: False
        value.prepare(replace(state, step=1))
        self.assertTrue(value.hold)
        self.assertEqual(value.path, ())
        value.state_factory.assert_not_called()

    def test_slowdown_fades_residual_without_replacing_direction(self):
        state = public_state()
        value = navigation(state)
        value.prepare(state)
        current = np.asarray(value.safe_target)+[1.5, 0, 0]
        moved = replace(state, step=1, agents=tuple(replace(a, position=tuple(current)) if a.agent_id == 3 else a for a in state.agents))
        value.tracker.reset()
        value.path = (value.safe_target,)
        value.prepare(moved)
        self.assertAlmostEqual(value.residual_factor, 1.5/2.4)
        raw = torch.tensor([[1., -1., .4]]*4)
        result = value.apply_residual_safety(raw, moved)
        self.assertTrue(torch.equal(result[:3], raw[:3]))
        torch.testing.assert_close(result[3], raw[3]*(1.5/2.4))

    def test_arrival_hold_has_no_active_residual(self):
        state = public_state()
        value = navigation(state)
        value.prepare(state)
        moved = replace(state, step=1, agents=tuple(replace(a, position=value.safe_target) if a.agent_id == 3 else a for a in state.agents))
        value.path = (value.safe_target,)
        value.tracker.reset()
        value.prepare(moved)
        self.assertTrue(value.hold)
        self.assertEqual(value.residual_factor, 0)
        np.testing.assert_array_equal(value.apply_residual_safety(np.ones((4, 3)), moved)[3], np.zeros(3))

    def test_existing_waypoint_prior_brakes_with_zero_hold_target(self):
        # Exercise the original method on tensors, without constructing an env.
        runtime = Mock()
        runtime._agent_pos = torch.zeros(4, 3)
        runtime._nav_targets = torch.zeros(4, 3)
        runtime._agent_acc = torch.zeros(4, 3)
        runtime._agent_vel = torch.ones(4, 3)*.2
        runtime.eps = 1e-6
        runtime._v_xy_max = runtime._v_z_max = torch.ones(4)
        runtime._a_xy_max = runtime._a_z_max = torch.ones(4)
        runtime.prior_slow_radius_xy, runtime.prior_slow_radius_z = 2.4, 1.2
        runtime.prior_kv_xy = runtime.prior_kv_z = 1.
        acceleration = UAVEnv._compute_waypoint_prior_acc(runtime)
        self.assertTrue(torch.all(acceleration[3] < 0))
        torch.testing.assert_close(runtime._agent_vel, torch.ones(4, 3)*.2)

    def test_searcher_on_local_segment_forces_hold(self):
        state = public_state()
        value = navigation(state)
        value.prepare(state)
        midpoint = tuple((np.asarray(state.agents[3].position)+value.tracking_target)/2)
        changed = replace(state, step=1, agents=tuple(replace(a, position=midpoint) if a.agent_id == 0 else a for a in state.agents))
        value.prepare(changed)
        self.assertTrue(value.hold)
        self.assertEqual(value.details['target_update_reason'], 'LOCAL_CLEARANCE_HOLD')

    def test_found_releases_guidance_action_and_tracker_without_velocity_reset(self):
        state = public_state()
        value = adapter(state)
        value.prepare_guidance(guidance(state), state)
        found = replace(state, target_found=True, agents=tuple(replace(a, velocity=(.2, -.4, .1)) for a in state.agents))
        velocity = found.agents[3].velocity
        original, actions = guidance(found), torch.randn(4, 3)
        self.assertIs(value.prepare_guidance(original, found), original)
        self.assertIs(value.before_action(actions, found), actions)
        self.assertIsNone(value.standby_target)
        self.assertIsNone(value.navigation.safe_target)
        self.assertEqual(value.navigation.path, ())
        self.assertIsNone(value.navigation.tracker.snapshot(3).current_target)
        self.assertEqual(found.agents[3].velocity, velocity)

    def test_collision_diagnostics_partition_by_pre_transition_found(self):
        state = public_state()
        diag = StandbyDiagnostics('fixture', 0, True)
        actions = np.zeros((4, 3))
        for step, before_found, after_found, flags in ((1, False, False, [True, False, False, True]),
                (2, False, True, [False, True, False, True]), (3, True, True, [False, False, False, True])):
            before, after = replace(state, step=step-1, target_found=before_found), replace(state, step=step, target_found=after_found)
            diag.before_action(before, actions, None)
            diag.observe_transition(before, after, actions, flags)
        self.assertEqual(diag.pre_count, 2)
        self.assertEqual(diag.post_count, 1)
        self.assertEqual(diag.searcher_counts, [1, 1, 0])

    def test_found_state_speed_latched_once_on_actual_transition(self):
        state = public_state()
        value = adapter(state)
        value.prepare_guidance(guidance(state), state)
        actions = value.before_action(torch.ones(4, 3), state)
        found = replace(state, step=1, target_found=True, agents=tuple(replace(a, velocity=(3., 4., 0.)) if a.agent_id == 3 else a for a in state.agents))
        value.observe_transition(state, found, actions, [False]*4, 7.)
        row = value.payload()['executor_standby_found_state.csv'][0]
        self.assertEqual(row['executor_speed_at_found'], 5.)
        self.assertEqual(row['executor_distance_to_target_at_found'], 7.)
        self.assertTrue(row['standby_active_before_found'])
        self.assertAlmostEqual(row['executor_action_norm_at_found'], np.linalg.norm(actions[3]))
        value.observe_transition(found, replace(found, step=2), actions, [False]*4)
        self.assertEqual(len(value.payload()['executor_standby_found_state.csv']), 1)
        self.assertFalse(value.navigation.active)

    def test_missing_collision_data_is_na_not_zero(self):
        state = public_state()
        diag = StandbyDiagnostics('fixture', 0, True)
        diag.before_action(state, np.zeros((4, 3)), None)
        diag.observe_transition(state, replace(state, step=1), np.zeros((4, 3)), None)
        self.assertIsNone(diag.rows[0]['executor_collision_event'])
        self.assertIsNone(diag.rows[0]['executor_pre_found_collision_count'])
        diag.before_action(replace(state, step=1), np.zeros((4, 3)), None)
        diag.observe_transition(replace(state, step=1), replace(state, step=2), np.zeros((4, 3)), [False]*4)
        self.assertIsNone(diag.rows[-1]['executor_pre_found_collision_count'])

    def test_diagnostics_disabled_does_not_change_control(self):
        state = public_state()
        left, right = adapter(state, diagnostics=True), adapter(state, diagnostics=False)
        self.assertEqual(left.prepare_guidance(guidance(state), state), right.prepare_guidance(guidance(state), state))
        actions = torch.randn(4, 3)
        self.assertTrue(torch.equal(left.before_action(actions, state), right.before_action(actions, state)))
        self.assertEqual(right.payload()['executor_standby_diagnostics.csv'], [])

    def test_old_config_resolves_inherited_defaults_deterministically(self):
        old = {'executor_standby': {'enabled': True, 'gain': .5}}
        first = resolve_beds(old)
        self.assertEqual(first, resolve_beds(copy.deepcopy(old)))
        self.assertEqual(first['executor_standby']['update_interval'], 10)
        self.assertEqual(first['executor_standby']['target_shift_threshold'], .75)
        self.assertEqual(old, {'executor_standby': {'enabled': True, 'gain': .5}})

    def test_input_public_arrays_unchanged(self):
        state = public_state()
        before = hashlib.sha256(state.grid.cell_centers.tobytes()+state.occupancy.occupied_mask.tobytes()).hexdigest()
        value = navigation(state)
        value.prepare(state)
        after = hashlib.sha256(state.grid.cell_centers.tobytes()+state.occupancy.occupied_mask.tobytes()).hexdigest()
        self.assertEqual(before, after)

    def test_diagnostic_missing_fields_serialize_na_and_keep_other_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retained = root/'historical.csv'
            retained.write_bytes(b'retained\n')
            write_diagnostics(root, 'executor_standby_found_state.csv', [dict(scenario_id='s', episode=0, step=1)])
            self.assertIn('NA', (root/'executor_standby_found_state.csv').read_text())
            self.assertEqual(retained.read_bytes(), b'retained\n')


if __name__ == '__main__':
    unittest.main()
