import unittest
from pathlib import Path


class NoTargetTruthTest(unittest.TestCase):
    def test_corrected_modules_use_only_public_context(self):
        root = Path(__file__).resolve().parents[1]
        names = ("mission_context.py", "controller.py", "execution_manager.py", "route_impact.py")
        source = "\n".join((root/"chapter3_bser"/"online"/name).read_text(encoding="utf-8") for name in names)
        forbidden = (
            "get_"+"target_state", "env."+"unwrapped", "_task_"+"target",
            "true_"+"target", "target_"+"truth", "ground_truth_"+"obstacle",
        )
        self.assertEqual([token for token in forbidden if token in source], [])


if __name__ == "__main__": unittest.main()
