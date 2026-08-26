import unittest
import numpy as np
from tests.bser_test_utils import synthetic_instance


class DetectionProbabilityTest(unittest.TestCase):
    def test_probabilities_are_finite_bounded_and_deterministic(self):
        _, _, generated, context = synthetic_instance()
        values = context.detection_by_id[generated.search_candidates[0].candidate_id]
        self.assertTrue(np.all(np.isfinite(values))); self.assertTrue(np.all((0 <= values) & (values <= 1))); self.assertFalse(values.flags.writeable)


if __name__ == "__main__": unittest.main()
