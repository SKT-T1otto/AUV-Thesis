import unittes
from chapter3_bser.experiments.instance_builder import ENVIRONMENT_MAX_STEPS
from chapter3_bser.config import load_bser_phase1a1_config

class Step50ProtocolTest(unittest.TestCase):
    def test_step50_is_requested_under_400_step_environment(self):
        self.assertEqual(load_bser_phase1a1_config()["e1_v2"]["snapshot_steps"], [0,10,25,50])
        self.assertEqual(ENVIRONMENT_MAX_STEPS, 400)
