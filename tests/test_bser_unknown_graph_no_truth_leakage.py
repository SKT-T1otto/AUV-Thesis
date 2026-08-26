import ast
from pathlib import Path
import unittest

class NoTruthLeakagePhase1A1Test(unittest.TestCase):
    def test_unknown_graph_adapter_does_not_read_obstacle_truth(self):
        tree=ast.parse((Path(__file__).resolve().parents[1]/"core/mapping/planning_graph.py").read_text()); attrs={node.attr for node in ast.walk(tree) if isinstance(node,ast.Attribute)}; self.assertFalse(attrs & {"obstacles","ground_truth_obstacles","_truth_occupancy_mask"})
