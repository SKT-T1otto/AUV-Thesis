import ast
from pathlib import Path
import unittest

class NoLayoutLeakageTest(unittest.TestCase):
    def test_algorithm_modules_never_access_obstacle_layout_id(self):
        root=Path(__file__).resolve().parents[1]; paths=[root/"chapter3_bser/candidate_generator.py",root/"chapter3_bser/detection_model.py",root/"chapter3_bser/objective.py",root/"chapter3_bser/greedy_solver.py",root/"chapter3_bser/lazy_greedy_solver.py",root/"chapter3_bser/exact_solver.py"]
        for path in paths: self.assertNotIn("obstacle_layout_id",{node.attr for node in ast.walk(ast.parse(path.read_text())) if isinstance(node,ast.Attribute)})
