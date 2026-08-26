import unittest

from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.hysteresis.policy import ReplanningPolicy
from chapter3_bser.online.config import load_phase1b_config


class HysteresisTest(unittest.TestCase):
    def test_cooldown_gain_and_critical_rules(self):
        policy = ReplanningPolicy(load_phase1b_config())
        policy.mark_replan(0)
        self.assertFalse(policy.decide((BSEREvent.BELIEF_SHIFT,), 1.0, 0.0, 10).should_replan)
        self.assertFalse(policy.decide((BSEREvent.BELIEF_SHIFT,), 0.005, 0.0, 20).should_replan)
        self.assertTrue(policy.decide((BSEREvent.BELIEF_SHIFT,), 0.02, 0.0, 20).should_replan)
        self.assertTrue(policy.decide((BSEREvent.TARGET_FOUND,), 0.0, 1.0, 1).should_replan)


if __name__ == "__main__": unittest.main()
