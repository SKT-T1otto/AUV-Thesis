from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path

from chapter3_bser.experiments.phase1c_prrac.search_collision_recovery import (
    plan_forced_route_refresh,
    plan_local_connector,
)


class S2A1InformationBoundaryTests(unittest.TestCase):
    def test_v2_planner_and_controller_have_no_oracle_access(self):
        root = Path(__file__).resolve().parents[1] / "chapter3_bser/experiments/phase1c_prrac/search_collision_recovery"
        forbidden = {
            "unwrapped", "obstacles", "ground_truth_obstacles", "true_target",
            "reward", "success", "future_trajectory", "scenario_id",
            "evaluation_outcome",
        }
        for name in ("planner_v2.py", "controller_v2.py"):
            tree = ast.parse((root / name).read_text(encoding="utf-8"))
            identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            identifiers.update(node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute))
            self.assertFalse(identifiers & forbidden, name)

    def test_planner_runtime_interfaces_accept_only_public_state_inputs(self):
        allowed = {
            "state", "agent_id", "candidate_id", "semantic_waypoint",
            "base_tracking_waypoint", "base_planned_path", "last_safe",
            "failed_endpoint_cells", "failed_direction", "attempt_id", "service",
        }
        for function in (plan_forced_route_refresh, plan_local_connector):
            self.assertLessEqual(set(inspect.signature(function).parameters), allowed)


if __name__ == "__main__":
    unittest.main()
