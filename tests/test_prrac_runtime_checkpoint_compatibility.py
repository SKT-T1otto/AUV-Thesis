import copy
import unittest

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.experiments.phase1c_prrac.runtime_factory import NATIVE_B1_RUNTIME_REVISION
from tests.prrac_evaluation_support import checkpoint_payload
from chapter3_bser.models.prrac.prrac_maddpg import PRRACMADDPG
from chapter3_bser.experiments.phase1c_prrac import train_phase1c_prrac as trainer


class RuntimeCheckpointCompatibilityTests(unittest.TestCase):
    def test_old_checkpoint_is_accepted_by_legacy_and_rejected_by_native(self):
        payload = checkpoint_payload()
        legacy = evaluator._load_config(evaluator.ROOT / "configs/chapter3/bser_phase1c_prrac_s1_search_diag_legacy.json")
        evaluator._validate_checkpoint_payload(payload, legacy)
        native = evaluator._load_config(evaluator.ROOT / "configs/chapter3/bser_phase1c_prrac_s1_search_diag_native.json")
        with self.assertRaisesRegex(ValueError, "execution runtime revision mismatch"):
            evaluator._validate_checkpoint_payload(payload, native)

    def test_native_metadata_contract_is_required(self):
        payload = checkpoint_payload()
        payload["metadata"].update(
            execution_runtime_revision=NATIVE_B1_RUNTIME_REVISION,
            execution_variant="B1_ATOMIC_LAST_VALID",
            runtime_integration_mode="native",
            controller_factory_version="prrac.controller_factory.v1",
        )
        native = evaluator._load_config(evaluator.ROOT / "configs/chapter3/bser_phase1c_prrac_s1_search_diag_native.json")
        evaluator._validate_checkpoint_payload(copy.deepcopy(payload), native)

    def test_new_checkpoint_metadata_records_runtime_origin(self):
        config = trainer._load_config(
            trainer.ROOT / "configs/chapter3/bser_phase1c_prrac_s1_train.json"
        )
        learner = PRRACMADDPG(
            architecture=config["architecture"], loss=config["loss"],
            gamma=config["rl"]["gamma"], tau=config["rl"]["tau"],
        )
        metadata = trainer._checkpoint_metadata(config, 1, learner)
        self.assertEqual(metadata["execution_runtime_revision"], NATIVE_B1_RUNTIME_REVISION)
        self.assertEqual(metadata["execution_variant"], "B1_ATOMIC_LAST_VALID")
        self.assertEqual(metadata["runtime_integration_mode"], "native")
        self.assertEqual(metadata["schema"], trainer.CHECKPOINT_SCHEMA)


if __name__ == "__main__":
    unittest.main()
