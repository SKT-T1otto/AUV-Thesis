import tempfile
import unittest
from pathlib import Path

from chapter3_bser.experiments.phase1b1_pilot.run_pilot import run


class PilotSmokeTest(unittest.TestCase):
    def test_four_methods_complete_bounded_cpu_smoke(self):
        with tempfile.TemporaryDirectory() as raw:
            summary = run(Path(raw), smoke=True, workers=4)
            self.assertTrue(summary["engineering_passed"])
            self.assertEqual(summary["completed_condition_episode_count"], 4)
            self.assertEqual(summary["failure_count"], 0)


if __name__ == "__main__": unittest.main()
