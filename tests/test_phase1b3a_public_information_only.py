import ast
from pathlib import Path
import unittest

from chapter3_bser.diagnostics.event_semantics import (
    mission_phase,
    public_target_lock_violation,
)
from tests.bser_online_test_utils import mission_context, state_at


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_FILES = (
    ROOT / "chapter3_bser" / "diagnostics" / "event_semantics.py",
    ROOT / "chapter3_bser" / "experiments" / "phase1b3a_diagnosis" / "run_diagnosis.py",
)
FORBIDDEN_IDENTIFIERS = {
    "true_target",
    "target_position",
    "obstacle_list",
    "hidden_communication_state",
    "future_state",
    "ground_truth",
    "unwrapped",
}


class Phase1B3APublicInformationOnlyTest(unittest.TestCase):
    def test_ast_has_no_forbidden_private_information_names(self):
        for path in DIAGNOSTIC_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            identifiers = {
                node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
            } | {
                node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
            }
            self.assertFalse(FORBIDDEN_IDENTIFIERS & identifiers, path)

    def test_phase_uses_public_context_flags(self):
        initial = state_at(0)
        self.assertEqual(mission_phase(mission_context(initial)), "SEARCH")
        found = state_at(1, target_found=True)
        waiting = mission_context(found, target_found=True, executor_knows_target=False)
        executing = mission_context(found, target_found=True, executor_knows_target=True)
        self.assertEqual(mission_phase(waiting), "WAIT_PUBLIC_HANDOFF")
        self.assertEqual(mission_phase(executing), "EXECUTE_PUBLIC_TARGET")

    def test_public_target_lock_checks_committed_public_assignment(self):
        locked = (1.0, 2.0, 3.0)
        self.assertFalse(
            public_target_lock_violation(locked, locked, "PUBLIC_HANDOFF_TARGET")
        )
        self.assertTrue(
            public_target_lock_violation(locked, (1.0, 2.0, 4.0), "PUBLIC_HANDOFF_TARGET")
        )
        self.assertTrue(public_target_lock_violation(locked, locked, "standby:y02"))


if __name__ == "__main__":
    unittest.main()
