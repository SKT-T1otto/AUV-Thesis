import unittest

from chapter3_bser.events.event_detector import EventDetector
from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.online.config import load_phase1b_config
from tests.bser_online_test_utils import discovered_obstacle, state_at


class ObstacleTriggerTest(unittest.TestCase):
    def test_new_obstacle_uses_occupancy_belief_only(self):
        detection = EventDetector(load_phase1b_config()).detect(state_at(0), discovered_obstacle())
        self.assertEqual(detection.new_obstacle_cells, 1)
        self.assertGreater(detection.new_obstacle_probability_mass, 0.5)
        self.assertIn(BSEREvent.OBSTACLE_DISCOVERED, detection.events)


if __name__ == "__main__": unittest.main()
