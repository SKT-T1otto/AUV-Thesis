from __future__ import annotations

import inspect
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.experiments.phase1c_prrac.evaluation_trace import failure_trace_row
from chapter3_bser.integration.control_context import (
    AgentAssignmentContextV1,
    BSERControlContextV1,
    ExecutorAssignmentContextV1,
)
from chapter3_bser.models.prrac.phase_routed_actor import PRRACActorOutput


class _RecordingActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shapes = []

    def forward(self, observation):
        self.shapes.append(tuple(observation.shape))
        batch = observation.shape[0]
        zeros3 = torch.zeros(batch, 3, dtype=observation.dtype, device=observation.device)
        probabilities = torch.full_like(zeros3, 1.0 / 3.0)
        return PRRACActorOutput(
            zeros3,
            zeros3,
            probabilities,
            torch.zeros(batch, 3, 3, dtype=observation.dtype, device=observation.device),
            torch.zeros(batch, 1, dtype=observation.dtype, device=observation.device),
            torch.zeros(batch, 1, dtype=observation.dtype, device=observation.device),
            zeros3,
        )


def _guidance() -> BSERControlContextV1:
    assignments = tuple(
        AgentAssignmentContextV1(
            agent_id=index,
            role="executor" if index == 3 else "searcher",
            assignment_kind="execution" if index == 3 else "search",
            assignment_id=f"assignment-{index}",
            final_waypoint=(1.0, 2.0, 3.0),
            planned_path=(),
            tracking_waypoint=(1.0, 2.0, 3.0),
            hold_position=(0.0, 0.0, 0.0),
            hold_state=False,
            reachable=True,
            execution_request=index == 3,
        )
        for index in range(4)
    )
    executor = ExecutorAssignmentContextV1(
        executor_id=3,
        source="public",
        target_region=(1.0, 2.0, 3.0),
        planned_path=(),
        tracking_waypoint=(1.0, 2.0, 3.0),
        hold_position=(0.0, 0.0, 0.0),
        hold_state=False,
        reachable=True,
        execution_request=True,
    )
    return BSERControlContextV1(
        schema_version="bser.control_context.v1",
        allocation_version="v1",
        allocation_hash="hash",
        step=1,
        mission_phase="EXECUTION",
        agent_assignments=assignments,
        executor_assignment=executor,
        decision_reason="PUBLIC",
    )


class _GuidanceEnv:
    def __init__(self) -> None:
        self.events = []

    def install_guidance(self, context):
        self.events.append(("install", context.decision_reason))

    def refresh_observation_after_guidance(self):
        self.events.append(("refresh", None))
        return "public-observations"


class PRRACEvaluationInformationBoundaryTests(unittest.TestCase):
    def test_actor_receives_only_four_public_28d_observations(self) -> None:
        modules = [_RecordingActor() for _ in range(4)]
        actor = SimpleNamespace(agents=[SimpleNamespace(actor=module) for module in modules])
        observations = tuple(np.zeros(28, dtype=np.float32) for _ in range(4))

        outputs = evaluator._policy_outputs(actor, observations, torch.device("cpu"))

        self.assertEqual(len(outputs), 4)
        self.assertEqual([module.shapes for module in modules], [[(1, 28)]] * 4)
        source = inspect.getsource(evaluator._policy_outputs)
        self.assertEqual(
            tuple(inspect.signature(evaluator._policy_outputs).parameters),
            ("actor", "observations", "device"),
        )
        self.assertNotIn("true_target", source)
        self.assertNotIn("torch.cat", source)

    def test_oracle_tracking_is_installed_after_public_policy_observation(self) -> None:
        env = _GuidanceEnv()
        observations, installed = evaluator._install_next_guidance(
            env,
            _guidance(),
            mode="oracle_current_target_diagnostic",
            true_target=(9.0, 8.0, 7.0),
        )

        self.assertEqual(observations, "public-observations")
        self.assertEqual(
            env.events,
            [
                ("install", "PUBLIC"),
                ("refresh", None),
                ("install", "PRIVILEGED_ORACLE_CURRENT_TARGET_DIAGNOSTIC_ONLY"),
            ],
        )
        self.assertEqual(installed.executor_assignment.tracking_waypoint, (9.0, 8.0, 7.0))

    def test_oracle_mode_and_failure_trace_are_privileged_diagnostic_only(self) -> None:
        info = evaluator._checkpoint_info(
            evaluator.ROOT / "checkpoint.pt",
            {
                "schema": evaluator.CHECKPOINT_SCHEMA,
                "metadata": {"config_hash": "hash"},
                "completed_episode": 1,
            },
            "oracle_current_target_diagnostic",
        )
        trace = failure_trace_row(true_target_position=[1.0, 2.0, 3.0])
        self.assertTrue(info["diagnostic_only"])
        self.assertTrue(info["privileged_oracle"])
        self.assertTrue(trace["diagnostic_only"])
        self.assertTrue(trace["privileged_oracle"])
        self.assertIn("true_target_position", trace)


if __name__ == "__main__":
    unittest.main()
