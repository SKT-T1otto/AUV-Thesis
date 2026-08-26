import unittest

from chapter3_bser.online.controller import OnlineBSERController
from tests.bser_online_test_utils import state_at


class NoOverplanningTest(unittest.TestCase):
    def test_no_event_does_not_replan_each_step(self):
        controller = OnlineBSERController()
        controller.initialize(state_at(0))
        for step in range(1, 20):
            self.assertFalse(controller.step(state_at(step)).replanned)
        self.assertEqual(controller.replan_count, 1)


if __name__ == "__main__": unittest.main()
