import unittest

from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.hysteresis.policy import ReplanningPolicy
from chapter3_bser.online.config import load_phase1b1_config


class EventSpecificCooldownTest(unittest.TestCase):
    def test_belief_cooldown_does_not_block_obstacle(self):
        policy = ReplanningPolicy(load_phase1b1_config())
        policy.mark_replan(0, BSEREvent.BELIEF_SHIFT)
        self.assertFalse(policy.decide((BSEREvent.BELIEF_SHIFT,), 2.0, 1.0, 10).should_replan)
        self.assertTrue(policy.decide((BSEREvent.OBSTACLE_DISCOVERED,), 1.0, 1.0, 10).should_replan)


if __name__ == "__main__": unittest.main()
