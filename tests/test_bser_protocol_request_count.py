import unittest
from chapter3_bser.config import load_bser_phase1a1_config

class RequestCountTest(unittest.TestCase):
    def test_protocol_declares_exactly_240_requests(self):
        config=load_bser_phase1a1_config(); self.assertEqual(4*5*3*len(config["e1_v2"]["snapshot_steps"]),240); self.assertEqual(config["e1_v2"]["protocol_request_count"],240)
