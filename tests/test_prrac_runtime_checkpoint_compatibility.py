import copy
from pathlib import Path
import tempfile
import unittest

import torch

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.experiments.phase1c_prrac.runtime_factory import NATIVE_B1_RUNTIME_REVISION
from tests.prrac_evaluation_support import checkpoint_payload
from chapter3_bser.models.prrac.prrac_maddpg import PRRACMADDPG
from chapter3_bser.experiments.phase1c_prrac import train_phase1c_prrac as trainer
from chapter3_bser.experiments.phase1c_prrac.replay_adapter import PRRACReplayAdapter
from tests.test_prrac_checkpoint import _config as legacy_training_config


class RuntimeCheckpointCompatibilityTests(unittest.TestCase):
    @staticmethod
    def _learner(config):
        return PRRACMADDPG(
            architecture=config["architecture"],
            loss=config["loss"],
            gamma=float(config.get("rl", {}).get("gamma", 0.95)),
            tau=float(config.get("rl", {}).get("tau", 0.01)),
        )

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

    def test_native_resume_requires_real_factory_variant_and_integration_fields(self):
        config = trainer._load_config(
            trainer.ROOT / "configs/chapter3/bser_phase1c_prrac_s1_train.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = trainer._save_checkpoint(
                self._learner(config), PRRACReplayAdapter(8), root, config, 1,
                global_step=0, update_step=0, replay_sample_count=0,
                optimizer_update_count=0, episode_rows=[], execution_rows=[],
                prrac_rows=[],
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            trainer._load_checkpoint(
                path, self._learner(config), PRRACReplayAdapter(8), config
            )
            cases = (
                ("controller_factory_version", None, "controller factory version mismatch"),
                ("controller_factory_version", "wrong.factory", "controller factory version mismatch"),
                ("execution_variant", None, "execution variant mismatch"),
                ("runtime_integration_mode", None, "runtime integration mode mismatch"),
            )
            for index, (field, value, message) in enumerate(cases):
                mutated = copy.deepcopy(payload)
                if value is None:
                    mutated["metadata"].pop(field, None)
                else:
                    mutated["metadata"][field] = value
                candidate = root / f"mutated_{index}.pt"
                torch.save(mutated, candidate)
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, message):
                    trainer._load_checkpoint(
                        candidate,
                        self._learner(config),
                        PRRACReplayAdapter(8),
                        config,
                    )

    def test_legacy_resume_allows_missing_runtime_origin_fields(self):
        config = legacy_training_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = trainer._save_checkpoint(
                self._learner(config), PRRACReplayAdapter(8), root, config, 1,
                global_step=0, update_step=0, replay_sample_count=0,
                optimizer_update_count=0, episode_rows=[], execution_rows=[],
                prrac_rows=[],
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            for field in (
                "execution_variant", "runtime_integration_mode", "controller_factory_version"
            ):
                payload["metadata"].pop(field, None)
            candidate = root / "legacy_missing_origin_fields.pt"
            torch.save(payload, candidate)
            restored = trainer._load_checkpoint(
                candidate,
                self._learner(config),
                PRRACReplayAdapter(8),
                config,
            )
            self.assertEqual(restored["completed_episode"], 1)
            native = trainer._load_config(
                trainer.ROOT / "configs/chapter3/bser_phase1c_prrac_s1_train.json"
            )
            with self.assertRaisesRegex(
                ValueError, "checkpoint execution runtime revision mismatch"
            ):
                trainer._load_checkpoint(
                    candidate,
                    self._learner(native),
                    PRRACReplayAdapter(8),
                    native,
                )


if __name__ == "__main__":
    unittest.main()
