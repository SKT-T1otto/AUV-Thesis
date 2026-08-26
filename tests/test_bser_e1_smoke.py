import tempfile
from pathlib import Path
import unittest
from chapter3_bser.experiments.run_e1_offline import run


class E1SmokeTest(unittest.TestCase):
    def test_smoke_writes_explicit_temporary_output(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = run(Path(directory), smoke=True)
            self.assertTrue(summary["passed"]); self.assertGreater(summary["valid_instance_count"], 0); self.assertTrue((Path(directory) / "e1_summary.json").is_file())


if __name__ == "__main__": unittest.main()
