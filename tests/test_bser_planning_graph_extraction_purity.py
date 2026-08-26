import ast
from pathlib import Path
import unittest

class PlanningGraphPurityTest(unittest.TestCase):
    def test_adapter_restores_cache_in_finally_and_has_no_environment_step(self):
        tree=ast.parse((Path(__file__).resolve().parents[1]/"core/mapping/planning_graph.py").read_text())
        attrs={node.attr for node in ast.walk(tree) if isinstance(node,ast.Attribute)}
        self.assertIn("_geodesic_cache",attrs); self.assertNotIn("step",attrs); self.assertNotIn("reset",attrs)
