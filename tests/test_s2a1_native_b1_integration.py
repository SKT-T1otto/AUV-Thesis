from __future__ import annotations

from dataclasses import replace
import json
import unittest
from unittest import mock

import numpy as np
import torch

from chapter3_bser.controllers.state_provider import OnlinePlanningStateProvider
from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.experiments.phase1c_prrac.execution_continuity import ExecutionVariant
from chapter3_bser.experiments.phase1c_prrac.search_collision_recovery import (
    SearchRecoveryVariantV2,
    apply_search_recovery_guidance,
    build_search_recovery_controller,
)
from chapter3_bser.integration.rmaddpg_bridge import RMADDPGGuidanceBridge
from chapter3_bser.online.config import execution_runtime_config, load_phase1b2_config
from tests.prrac_evaluation_support import evaluation_config


class NativeB1IntegrationTests(unittest.TestCase):
    def test_native_b1_force_timing_actions_executor_divergence_and_cleanup(self):
        config = evaluation_config(scenario_count=1)
        config["max_steps"] = 5
        config.update({
            "checkpoint_runtime_revision": evaluator.NATIVE_B1_RUNTIME_REVISION,
            "evaluation_runtime_revision": evaluator.NATIVE_B1_RUNTIME_REVISION,
            "runtime_integration_mode": "native",
            "controller_factory_version": "prrac.controller_factory.v1",
        })
        reward = json.loads((evaluator.ROOT / "configs/chapter3/bser_phase1c_prrac_train.json").read_text(encoding="utf-8"))["reward"]
        scenario = evaluator._build_evaluation_manifest(config)[0][0]
        c0_env = evaluator._make_env(config, reward)
        c2_env = evaluator._make_env(config, reward)
        try:
            c0_env.reset(scenario=scenario, episode_id=0, episode_index=0)
            c2_env.reset(scenario=scenario, episode_id=0, episode_index=0)
            phase_config = load_phase1b2_config()
            runtime = execution_runtime_config(config)
            phase_config["execution_runtime"] = runtime
            provider = OnlinePlanningStateProvider(
                c2_env, refresh_interval=20,
                refresh_on_executor_handoff=runtime["refresh_on_executor_handoff"],
                refresh_on_public_target_shift=runtime["refresh_on_public_target_shift"],
                public_target_update_distance=runtime["public_target_update_distance"],
                public_target_update_min_steps=runtime["public_target_update_min_steps"],
            )
            state = provider.initialize()
            context = evaluator._public_context(c2_env, state)
            native = evaluator._build_episode_controller(
                phase_config, config, execution_variant=ExecutionVariant.B1_ATOMIC_LAST_VALID,
                runtime_integration_mode="native",
                checkpoint_runtime_revision=evaluator.NATIVE_B1_RUNTIME_REVISION,
            )
            initialized = native.initialize(state, context)
            base = RMADDPGGuidanceBridge().compile_guidance(
                initialized.allocation, state, context, decision_reason="INITIALIZE"
            )
            c0_env.install_guidance(base)
            c2_env.install_guidance(base)
            c0_env.refresh_observation_after_guidance()
            c2_env.refresh_observation_after_guidance()
            residual = torch.zeros((4, 3), dtype=torch.float32)
            c0_env.step(residual.clone())
            c2_env.step(residual.clone())
            self.assertIsNotNone(c2_env.last_prrac_transition_metadata)

            c0_recovery = build_search_recovery_controller(SearchRecoveryVariantV2.S2A1_C0_BASELINE)
            with mock.patch.object(provider, "snapshot", wraps=provider.snapshot) as c0_snapshot:
                if c0_recovery is not None and c0_recovery.force_refresh_requested:
                    provider.snapshot(force=True)
                c0_snapshot.assert_not_called()

            recovery = build_search_recovery_controller(SearchRecoveryVariantV2.S2A1_C2_LOCAL_CONNECTOR)
            collision_state = replace(state, step=1)
            recovery.observe_transition(
                stage_before=0, planning_state_before=state, planning_state_after=collision_state,
                installed_guidance_before=base, collision_flags=(True, False, False, False),
            )
            self.assertTrue(recovery.force_refresh_requested)
            original_snapshot = provider.snapshot

            def timed_snapshot(*, force=False):
                if force:
                    self.assertIsNotNone(c2_env.last_prrac_transition_metadata)
                return original_snapshot(force=force)

            with mock.patch.object(provider, "snapshot", side_effect=timed_snapshot) as snapshot:
                provider.snapshot(force=True)
                self.assertEqual(snapshot.call_count, 1)
                snapshot.assert_called_once_with(force=True)
            recovery.prepare_next_guidance(collision_state, base)
            collision_state2 = replace(collision_state, step=2)
            recovery.observe_transition(
                stage_before=0, planning_state_before=collision_state,
                planning_state_after=collision_state2, installed_guidance_before=base,
                collision_flags=(True, False, False, False),
            )
            recovery.prepare_next_guidance(collision_state2, base)
            overlay = apply_search_recovery_guidance(base, collision_state2, recovery)
            recovery.observe_activation(base, overlay)
            self.assertNotEqual(base.assignment_for(0).tracking_waypoint, overlay.assignment_for(0).tracking_waypoint)
            self.assertEqual(base.executor_assignment, overlay.executor_assignment)
            self.assertGreater(recovery.summary()["recovery_plan_active_step_count"], 0)

            c0_env.install_guidance(base)
            c2_env.install_guidance(overlay)
            c0_env.refresh_observation_after_guidance()
            c2_env.refresh_observation_after_guidance()
            for _ in range(3):
                c0_env.step(residual.clone())
                c2_env.step(residual.clone())
            c0_positions = np.asarray(c0_env.get_agent_state().positions, dtype=np.float64)
            c2_positions = np.asarray(c2_env.get_agent_state().positions, dtype=np.float64)
            self.assertGreater(float(np.linalg.norm(c0_positions[0] - c2_positions[0])), 0.0)
            np.testing.assert_allclose(c0_positions[3], c2_positions[3], atol=0.0, rtol=0.0)
            np.testing.assert_array_equal(residual.numpy(), residual.clone().numpy())

            recovery.observe_transition(
                stage_before=0, planning_state_before=collision_state2,
                planning_state_after=replace(collision_state2, step=3, target_found=True),
                installed_guidance_before=overlay, collision_flags=(False, False, False, False),
            )
            self.assertEqual(recovery.snapshot().active_agent_ids, ())
            payload = evaluator._json_safe({"activation": recovery.activation_rows()})
            self.assertFalse(evaluator._contains_tensor(payload))
            self.assertFalse(any(isinstance(value, np.ndarray) for row in payload["activation"] for value in row.values()))
        finally:
            c0_env.close()
            c2_env.close()


if __name__ == "__main__":
    unittest.main()
