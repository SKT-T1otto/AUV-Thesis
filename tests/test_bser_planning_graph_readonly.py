import unittest
from tests.bser_test_utils import synthetic_state

class PlanningGraphReadonlyTest(unittest.TestCase):
    def test_arrays_and_adjacency_are_readonly(self):
        graph=synthetic_state().planning_graph
        with self.assertRaises(ValueError): graph.valid_mask[0]=False
        with self.assertRaises((AttributeError,TypeError)): graph.searcher_adjacency[0][0].destination=8
