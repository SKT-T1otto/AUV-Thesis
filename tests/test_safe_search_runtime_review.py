"""Synthetic controller review contracts; no environment construction or physics."""
from collections import Counter
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.experiments.safe_search_v1 import runtime as safe
from chapter3_bser.experiments.safe_search_v1.failure_policy import SearchFailurePolicy


def config():
    return dict(mechanism_version="phase1b2_corrected", hysteresis=dict(
        minimum_gain_threshold=.01, minimum_relative_gain=.01,
        cooldown_steps=20, waypoint_stale_cooldown_steps=5,
        belief_cooldown_steps=20, obstacle_cooldown_steps=5))


class FakeInner:
    def __init__(self):
        self.config = config()
        self.policy = None
        self.current_allocation = NS(search_assignments=())
        self.current_context = None
        self.cache = NS(current=None)
        self.detector = NS(detect=self.detect)
        self.attempt_steps = []
        self.detect_calls = 0
        self.step_calls = 0

    def detect(self, *args, **kwargs):
        self.detect_calls += 1
        return NS(stale_searcher_ids=(0,), events=(BSEREvent.WAYPOINT_STALE,))

    def step(self, state, context):
        self.step_calls += 1
        invoked = not self.policy.event_cooldown_remaining(BSEREvent.WAYPOINT_STALE, state.step)
        if invoked:
            self.attempt_steps.append(state.step)
        return NS(replanned=False, diagnostics=NS(optimizer_invoked=invoked))


class ControllerReviewTests(unittest.TestCase):
    def owner(self):
        owner = NS(safe_options=dict(safe.VARIANTS["V2"]), safe_pending=set(), safe_counts=Counter(),
                   provider=NS(last_snapshot_was_full_refresh=False), state=None, refresh_steps=[])
        def refresh():
            owner.refresh_steps.append(owner.state.step)
            owner.provider.last_snapshot_was_full_refresh = True
            return owner.state
        owner.planning_views = NS(refresh=refresh)
        return owner

    def test_failed_retry_does_not_force_fresh_snapshot_on_every_next_step(self):
        owner, inner = self.owner(), FakeInner()
        controller = safe.SearchController(inner, owner)
        with patch.object(safe, "start_status", return_value="stale_start_endpoint"), \
             patch.object(safe, "topology_key", return_value="unchanged"):
            for step in (10, 11, 12):
                owner.state = NS(step=step, target_found=False, searcher_ids=(0,))
                owner.provider.last_snapshot_was_full_refresh = False
                controller.step(owner.state, None)
        self.assertEqual(inner.attempt_steps, [10])
        self.assertEqual(owner.refresh_steps, [10])

    def test_persistently_unsafe_path_does_not_bypass_failed_retry_backoff(self):
        owner, inner = self.owner(), FakeInner()
        owner.safe_options = dict(safe.VARIANTS["V4"])
        controller = safe.SearchController(inner, owner)
        with patch.object(safe, "start_status", return_value="ready"), \
             patch.object(safe, "topology_key", return_value="unchanged"), \
             patch.object(controller, "_geometry_pending", return_value={0}):
            for step in (10, 11, 12):
                owner.state = NS(step=step, target_found=False, searcher_ids=(0,))
                owner.provider.last_snapshot_was_full_refresh = False
                controller.step(owner.state, None)
        self.assertEqual(inner.attempt_steps, [10])
        self.assertEqual(owner.refresh_steps, [10])

    def test_found_branch_disables_search_hooks_and_delegates_once(self):
        owner, inner = self.owner(), FakeInner()
        controller = safe.SearchController(inner, owner)
        owner.safe_pending.add(0)
        owner.planning_views.refresh = lambda: self.fail("must not refresh after Found")
        controller.step(NS(step=20, target_found=True), None)
        self.assertFalse(inner.policy.search_enabled)
        self.assertFalse(owner.safe_pending)
        self.assertEqual(inner.detect_calls, 0)
        self.assertEqual(inner.step_calls, 1)

    def test_v0_uses_exact_original_classes_and_passes_seed_episode_id(self):
        native_config, scenario = {}, {}
        for baseline, name in (("B0_search_prior", "BasicSearchPriorRuntime"), ("B1_bser_prior", "BSERPriorRuntime")):
            with self.subTest(baseline=baseline), patch.object(safe, name) as constructor:
                result = safe.make_runtime(native_config, scenario, baseline=baseline, variant="V0", seed=12757, episode_id=28)
                constructor.assert_called_once_with(native_config, scenario, seed=12757, episode_id=28)
                self.assertIs(result, constructor.return_value)

    def test_failed_attempts_preserve_nonsearch_event_cooldowns(self):
        policy = SearchFailurePolicy(config())
        policy.mark_attempt(30, BSEREvent.BELIEF_SHIFT)
        self.assertEqual(policy.event_cooldown_remaining(BSEREvent.BELIEF_SHIFT, 31), 19)
        policy.search_enabled = False
        policy.mark_attempt(35, BSEREvent.OBSTACLE_DISCOVERED)
        self.assertEqual(policy.event_cooldown_remaining(BSEREvent.OBSTACLE_DISCOVERED, 36), 4)


if __name__ == "__main__":
    unittest.main()
