import unittest

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.experiments.phase1c_prrac.execution_continuity import (
    ExecutionVariant, paired_execution_variant_comparisons,
)


def row(variant, scenario, success=False):
    return {
        "checkpoint": "checkpoint.pt",
        "checkpoint_config_hash": "config",
        "checkpoint_episode": 1,
        "evaluation_mode": "full_prrac",
        "execution_variant": variant.value,
        "manifest_sha256": "same-manifest",
        "scenario_id": scenario,
        "found": True,
        "contact_episode": True,
        "hold_episode": success,
        "success": success,
        "collision_episode": False,
        "post_found_collision_count": 0,
        "post_found_step_count": 2,
        "executor_route_active_post_found_steps": 1,
        "executor_invalid_count_post_found": 0,
        "assignment_unreachable_count_post_found": 0,
        "executor_min_distance_to_target": 1.0,
        "executor_final_distance_to_target": 2.0,
        "router_confusion_matrix": [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
    }


class ExecutionAblationEvaluatorTests(unittest.TestCase):
    def test_ablation_config_and_runtime_metadata_are_registered(self):
        config = evaluator._load_config(
            evaluator.ROOT / "configs/chapter3/bser_phase1c_prrac_execution_ablation.json"
        )
        self.assertEqual(config["evaluation_episodes"], 50)
        self.assertEqual(config["execution_variants"], [item.value for item in ExecutionVariant])
        info = evaluator._checkpoint_info(
            evaluator.ROOT / "checkpoint.pt",
            {"schema": evaluator.CHECKPOINT_SCHEMA, "metadata": {
                "config_hash": "hash", "execution_runtime_revision": "dynamic_public_intercept_v2_1"
            }, "completed_episode": 1},
            "full_prrac", ExecutionVariant.B2_REACHABLE_PROXY,
            manifest_sha256="manifest", execution_overlay_config_hash="overlay",
        )
        self.assertTrue(info["runtime_overlay_enabled"])
        self.assertEqual(info["checkpoint_runtime_revision"], "dynamic_public_intercept_v2_1")
        self.assertEqual(info["evaluation_runtime_revision"], "dynamic_public_intercept_v3_reachable_proxy")

    def test_four_variants_produce_six_paired_rows_on_same_manifest(self):
        rows = [row(variant, scenario, success=index % 2 == 0)
                for index, variant in enumerate(ExecutionVariant)
                for scenario in ("s0", "s1")]
        comparisons = paired_execution_variant_comparisons(rows)
        self.assertEqual(len(comparisons), 6)
        self.assertTrue(all(item["manifest_sha256"] == "same-manifest" for item in comparisons))


if __name__ == "__main__":
    unittest.main()
