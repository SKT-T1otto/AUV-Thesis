from __future__ import annotations

import ast
import unittest
from pathlib import Path


class InformationBoundaryTests(unittest.TestCase):
    def test_recovery_package_has_no_forbidden_oracle_access(self):
        root=Path(__file__).resolve().parents[1]/"chapter3_bser/experiments/phase1c_prrac/search_collision_recovery"
        forbidden={"unwrapped","obstacles","ground_truth_obstacles","true_target","reward","scenario_id","future_trajectory"}
        for path in root.glob("*.py"):
            tree=ast.parse(path.read_text(encoding="utf-8")); attrs={node.attr for node in ast.walk(tree) if isinstance(node,ast.Attribute)}; names={node.id for node in ast.walk(tree) if isinstance(node,ast.Name)}
            self.assertFalse((attrs|names)&forbidden, path.name)


if __name__ == "__main__": unittest.main()
