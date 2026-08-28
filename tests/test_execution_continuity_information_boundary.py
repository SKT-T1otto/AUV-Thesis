import inspect
from pathlib import Path
import unittest

import numpy as np
import torch

from chapter3_bser.experiments.phase1c_prrac import (
    ARCHITECTURE_VERSION, CHECKPOINT_SCHEMA,
)
from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.experiments.phase1c_prrac.execution_continuity import planner
from tests.test_prrac_evaluation_information_boundary import _RecordingActor


class ExecutionContinuityInformationBoundaryTests(unittest.TestCase):
    def test_proxy_planner_has_no_true_target_input_or_lookup(self):
        signature = inspect.signature(planner.assign_reachable_public_proxy)
        self.assertEqual(tuple(signature.parameters), ("state", "semantic_target", "service"))
        source = Path(planner.__file__).read_text(encoding="utf-8").lower()
        self.assertNotIn("true_target", source)
        self.assertNotIn("target_state", source)

    def test_actor_input_and_checkpoint_architecture_contracts_are_unchanged(self):
        modules = [_RecordingActor() for _ in range(4)]
        actor = type("Actor", (), {"agents": [type("Agent", (), {"actor": module})() for module in modules]})()
        evaluator._policy_outputs(
            actor, tuple(np.zeros(28, dtype=np.float32) for _ in range(4)), torch.device("cpu")
        )
        self.assertEqual([module.shapes for module in modules], [[(1, 28)]] * 4)
        self.assertEqual(CHECKPOINT_SCHEMA, "bser.phase1c.prrac.training_state.v1")
        self.assertEqual(ARCHITECTURE_VERSION, "prrac.phase_routed_residual.v1")


if __name__ == "__main__":
    unittest.main()
