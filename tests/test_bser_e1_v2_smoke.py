from pathlib import Path
import tempfile
import unittes
from chapter3_bser.experiments.run_e1_v2 import run

class E1V2SmokeTest(unittest.TestCase):
    def test_smoke_has_four_exact_protocol_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            summary=run(Path(directory),smoke=True,run_sensitivity=False); self.assertTrue(summary["passed"]); self.assertEqual(summary["protocol_request_count"],4); self.assertTrue((Path(directory)/"protocol_to_unique_state_map.csv").is_file())
