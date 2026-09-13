import unittest

from chapter3_bser.experiments.phase1b3a_diagnosis.run_diagnosis import CASES


class Phase1B3ACaseManifestTest(unittest.TestCase):
    def test_case_manifest_is_exactly_preregistered(self):
        self.assertEqual(
            [list(value) for value in CASES],
            [[2729, 2], [2731, 1], [2731, 3], [2732, 3], [2733, 0]],
        )


if __name__ == "__main__":
    unittest.main()
