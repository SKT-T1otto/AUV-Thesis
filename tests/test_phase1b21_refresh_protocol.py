import unittest

from chapter3_bser.experiments.phase1b1_pilot.run_pilot import (
    _config_for_method,
    _periodic_replan_due,
    _state_refresh_interval,
)


class Phase1B21RefreshProtocolTest(unittest.TestCase):
    def test_event_methods_use_the_same_twenty_step_refresh(self):
        phase1b1 = _config_for_method("Event-BSER-phase1b1")
        phase1b2 = _config_for_method("Event-BSER-phase1b2_corrected")
        self.assertEqual(_state_refresh_interval("Event-BSER-phase1b1", phase1b1), 20)
        self.assertEqual(
            _state_refresh_interval("Event-BSER-phase1b2_corrected", phase1b2),
            20,
        )
        self.assertFalse(_periodic_replan_due("Event-BSER-phase1b2_corrected", 20, 20))
        self.assertTrue(_periodic_replan_due("Periodic-BSER", 20, 20))


if __name__ == "__main__":
    unittest.main()
