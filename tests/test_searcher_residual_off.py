import unittest

import torch

from chapter3_bser.experiments.phase1c_prrac.evaluate_prrac_checkpoints import _apply_residual_mode
from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.experiments.phase1c_prrac.evaluation_metrics import recommend_checkpoint
from chapter3_bser.models.prrac.stage_mapping import PRRACStage
from tests.prrac_evaluation_support import checkpoint_payload


class SearcherResidualOffTests(unittest.TestCase):
    def test_search_only_copy_suppression_preserves_executor_bitwise(self):
        raw = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        saved = raw.clone()
        applied = _apply_residual_mode(raw, "searcher_residual_off", PRRACStage.SEARCH)
        self.assertTrue(torch.equal(applied[:3], torch.zeros_like(applied[:3])))
        self.assertTrue(torch.equal(applied[3], raw[3]))
        self.assertTrue(torch.equal(raw, saved))
        for stage in (PRRACStage.INTERCEPT, PRRACStage.HOLD):
            self.assertTrue(torch.equal(_apply_residual_mode(raw, "searcher_residual_off", stage), raw))

    def test_existing_modes_are_unchanged(self):
        raw = torch.ones((4, 3))
        self.assertTrue(torch.equal(_apply_residual_mode(raw, "full_prrac"), raw))
        executor_off = _apply_residual_mode(raw, "executor_residual_off")
        self.assertTrue(torch.equal(executor_off[:3], raw[:3]))
        self.assertTrue(torch.equal(executor_off[3], torch.zeros(3)))
        self.assertTrue(torch.equal(_apply_residual_mode(raw, "all_residual_off"), torch.zeros_like(raw)))

    def test_searcher_off_is_diagnostic_only_and_not_recommended(self):
        info = evaluator._checkpoint_info(
            evaluator.ROOT / "diagnostic.pt",
            checkpoint_payload(),
            "searcher_residual_off",
        )
        self.assertTrue(info["diagnostic_only"])
        result = recommend_checkpoint([{**info, "success_rate": 1.0}])
        self.assertIsNone(result["recommended_checkpoint"])


if __name__ == "__main__":
    unittest.main()
