import unittest
from tests.bser_test_utils import synthetic_instance


class CandidateGenerationTest(unittest.TestCase):
    def test_candidates_reachable_unique_and_partitioned(self):
        _, _, generated, _ = synthetic_instance()
        self.assertEqual(generated.search_count_by_agent, {0: 2, 1: 2, 2: 2})
        self.assertEqual(len({candidate.candidate_id for candidate in generated.search_candidates}), 6)
        self.assertTrue(all(candidate.path_points.shape[1] == 3 for candidate in generated.search_candidates))


if __name__ == "__main__": unittest.main()
