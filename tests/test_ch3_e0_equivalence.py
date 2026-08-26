import csv
import json
import unittes
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/chapter3/e0_equivalence"


class E0DeliveryTests(unittest.TestCase):
    def test_full_e0_passed(self):
        summary = json.loads((OUTPUT / "equivalence_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "PASS_CH3_E0_EQUIVALENCE")
        self.assertEqual(summary["completed_trajectory_count"], 60)
        self.assertEqual(summary["passed_trajectory_count"], 60)
        self.assertEqual(summary["task_event_mismatch_count"], 0)
        self.assertTrue(summary["legacy_unchanged"])
        self.assertTrue(summary["allow_bser_phase1a"])
        with (OUTPUT / "per_trajectory_results.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 60)
        self.assertTrue(all(row["passed"] == "True" for row in rows))


if __name__ == "__main__":
    unittest.main()
