from pathlib import Path
import tempfile
import unittest
from unittest import mock

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.experiments.phase1c_prrac.evaluation_metrics import aggregate_checkpoint
from chapter3_bser.experiments.phase1c_prrac.execution_continuity import ExecutionVariant
from tests.test_searcher_residual_paired_evaluation import row as search_row


class SearchContinuityOutputTests(unittest.TestCase):
    def test_five_outputs_are_registered_and_atomic_writers_leave_no_temp(self):
        names = {
            "search_continuity_episode.csv", "search_continuity_summary.csv",
            "paired_searcher_residual_comparison.csv", "search_failure_funnel.csv",
            "search_continuity_summary.json",
        }
        self.assertTrue(names.issubset(set(evaluator.OUTPUT_FILES)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator._write_csv(root / "search_continuity_episode.csv", [{"found": True}])
            evaluator._write_json(root / "search_continuity_summary.json", {"schema": "test"})
            self.assertFalse(any(path.name.startswith(".") and path.name.endswith(".tmp") for path in root.iterdir()))

    def test_write_outputs_materializes_search_tables_and_pair(self):
        rows = []
        for mode in ("full_prrac", "searcher_residual_off"):
            value = search_row(mode, "scenario-a", True, mode == "full_prrac")
            value.update(
                checkpoint_episode=12,
                checkpoint_schema=evaluator.CHECKPOINT_SCHEMA,
                search_continuity_diagnostics_schema="bser.phase1c.prrac.search_continuity.v1",
                router_confusion_matrix=[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                collision_episode=False,
                failure_stage="SUCCESS" if value["success"] else "FOUND_NO_CONTACT",
            )
            rows.append(value)
        summary = [aggregate_checkpoint([rows[0]], {key: rows[0].get(key) for key in rows[0]})]
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(evaluator, "_plot"), mock.patch.object(evaluator, "_plot_execution_variants"):
            root = Path(directory)
            evaluator._write_outputs(
                root,
                checkpoint_paths=[Path("checkpoint.pt")],
                scenarios=[{"scenario_id": "scenario-a"}],
                modes=("full_prrac", "searcher_residual_off"),
                execution_variants=(ExecutionVariant.B1_ATOMIC_LAST_VALID,),
                episode_rows=rows,
                summary_rows=summary,
                trace_rows=[],
                trace_index=[],
                progress={"completed": []},
            )
            for name in (
                "search_continuity_episode.csv", "search_continuity_summary.csv",
                "paired_searcher_residual_comparison.csv", "search_failure_funnel.csv",
                "search_continuity_summary.json",
            ):
                self.assertTrue((root / name).is_file(), name)
            paired = evaluator._read_csv(root / "paired_searcher_residual_comparison.csv")
            self.assertEqual(len(paired), 1)
            self.assertEqual(paired[0]["paired_scenario_count"], 1)


if __name__ == "__main__":
    unittest.main()
