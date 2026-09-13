"""Current bounded behavior checks; original TestCase bodies retained."""


import tempfile
from pathlib import Path
import unittest
from chapter3_bser.experiments.run_e1_offline import run


class E1SmokeTest(unittest.TestCase):
    def test_smoke_writes_explicit_temporary_output(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = run(Path(directory), smoke=True)
            self.assertTrue(summary["passed"]); self.assertGreater(summary["valid_instance_count"], 0); self.assertTrue((Path(directory) / "e1_summary.json").is_file())




import unittest
from chapter3_bser.experiments.instance_builder import ENVIRONMENT_MAX_STEPS
from chapter3_bser.config import load_bser_phase1a1_config

class Step50ProtocolTest(unittest.TestCase):
    def test_step50_is_requested_under_400_step_environment(self):
        self.assertEqual(load_bser_phase1a1_config()["e1_v2"]["snapshot_steps"], [0,10,25,50])
        self.assertEqual(ENVIRONMENT_MAX_STEPS, 400)


import unittest
from chapter3_bser.config import load_bser_phase1a1_config

class RequestCountTest(unittest.TestCase):
    def test_protocol_declares_exactly_240_requests(self):
        config=load_bser_phase1a1_config(); self.assertEqual(4*5*3*len(config["e1_v2"]["snapshot_steps"]),240); self.assertEqual(config["e1_v2"]["protocol_request_count"],240)
