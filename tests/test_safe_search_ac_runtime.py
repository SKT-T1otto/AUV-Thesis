"""A+C composition and the historical five-arm default remain independent."""
from collections import Counter
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from chapter3_bser.hysteresis.policy import ReplanningPolicy
from chapter3_bser.experiments.safe_search_v1 import run_paired, runtime
from chapter3_bser.experiments.safe_search_v1 import run_development
from chapter3_bser.experiments.safe_search_v1.safe_guidance import SafetyBridge
from tests.test_safe_search_runtime_review import FakeInner, config


class ACCompositionTests(unittest.TestCase):
    def build(self):
        inner = FakeInner()
        original_policy = ReplanningPolicy(config())
        inner.policy = original_policy
        state = NS(step=10, target_found=False, searcher_ids=(0,))
        owner = NS(safe_options=dict(runtime.VARIANTS['V5']), safe_pending={0},
                   safe_counts=Counter(), state=state,
                   provider=NS(last_snapshot_was_full_refresh=False))
        owner.planning_views = NS(refresh=lambda: state)
        return runtime.SearchController(inner, owner), owner, inner, original_policy

    def test_ac_repairs_geometry_pending_without_installing_failure_policy(self):
        controller, owner, inner, original_policy = self.build()
        self.assertIs(inner.policy, original_policy)
        with patch.object(controller, '_geometry_pending', return_value={0}), \
             patch.object(owner.planning_views, 'refresh', return_value=owner.state) as refresh, \
             patch.object(runtime, 'start_status', return_value='ready'):
            controller.step(owner.state, None)
        refresh.assert_called_once_with()
        self.assertEqual(inner.step_calls, 1)
        self.assertEqual(inner.attempt_steps, [10])
        self.assertNotIn('recovery_replans', owner.safe_counts)

    def test_ac_refreshes_stale_start_on_original_planning_opportunity(self):
        controller, owner, inner, original_policy = self.build()
        with patch.object(controller, '_geometry_pending', return_value=set()), \
             patch.object(owner.planning_views, 'refresh', return_value=owner.state) as refresh, \
             patch.object(runtime, 'start_status', return_value='stale_start_endpoint'):
            controller.step(owner.state, None)
        refresh.assert_called_once_with()
        self.assertIs(inner.policy, original_policy)

    def test_ac_found_delegates_without_search_preflight(self):
        controller, owner, inner, original_policy = self.build()
        with patch.object(owner.planning_views, 'refresh', side_effect=AssertionError('search refresh after Found')), \
             patch.object(controller, '_geometry_pending', side_effect=AssertionError('search safety after Found')):
            controller.step(NS(step=10, target_found=True), None)
        self.assertIs(inner.policy, original_policy)
        self.assertEqual(inner.detect_calls, 0)
        self.assertEqual(inner.step_calls, 1)
        self.assertFalse(owner.safe_pending)

    def test_ac_builds_original_safety_bridge_and_disabled_remains_original(self):
        owner = runtime.SafetyMixin()
        owner.safe_options = dict(runtime.VARIANTS['V5'])
        self.assertIsInstance(owner._build_guidance_bridge(), SafetyBridge)
        with patch.object(runtime, 'BasicSearchPriorRuntime') as native:
            runtime.make_runtime({}, {}, variant='V0', seed=7)
        native.assert_called_once_with({}, {}, seed=7, episode_id=0)

    def test_explicit_ac_does_not_change_frozen_five_arm_population(self):
        old = ('V0', 'V1', 'V2', 'V3', 'V4')
        self.assertEqual(run_paired.VARIANTS, old)
        self.assertEqual(run_development.VARIANTS, old)
        args = run_paired.parser().parse_args(['--manifest', 'm', '--output-dir', 'o'])
        self.assertEqual(args.variants, ','.join(old))
        self.assertEqual(run_paired.parse_variants('V5'), ['V5'])
        for bad in ('V6', 'V5,V5', 'V5,'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                run_paired.parse_variants(bad)


if __name__ == '__main__':
    unittest.main()
