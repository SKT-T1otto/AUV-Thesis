from dataclasses import replace
import unittes
from core.mapping.planning_state import planning_state_sha256
from tests.bser_test_utils import synthetic_state

class UniqueStateDedupTest(unittest.TestCase):
    def test_step_and_experiment_metadata_do_not_change_state_hash(self):
        state=synthetic_state(); self.assertEqual(planning_state_sha256(state),planning_state_sha256(replace(state,step=50)))
