from pathlib import Path
import copy
import math
from multiprocessing.reduction import ForkingPickler
import tempfile
import unittest

import torch

from chapter3_bser.experiments.phase1c_prrac.search_value_decision import (
    SearchValueDecisionController,
)
from chapter3_bser.experiments.phase1c_prrac.train_phase1c_prrac import (
    _collect_episode,
    _load_checkpoint,
    _save_checkpoint,
)
from chapter3_bser.experiments.phase1c_prrac.replay_adapter import (
    PRRACReplayAdapter,
)
from chapter3_bser.models.prrac.prrac_maddpg import PRRACMADDPG
from tests.test_prrac_cross_platform_runtime import _worker_jobs
from tests.test_search_value_head import _config, _learner


DECISION_CONFIG = {
    "enabled": True,
    "threshold": 0.35,
    "patience": 20,
    "cooldown": 50,
}


class SearchValueDecisionTests(unittest.TestCase):
    def test_high_value_never_triggers(self):
        controller = SearchValueDecisionController(DECISION_CONFIG)
        decisions = [
            controller.observe(0.8, step=step) for step in range(100)
        ]
        self.assertFalse(any(item.trigger_replan for item in decisions))
        self.assertEqual(controller.trigger_count, 0)
        self.assertEqual(controller.low_value_steps, 0)

    def test_twenty_continuous_low_value_steps_trigger(self):
        controller = SearchValueDecisionController(DECISION_CONFIG)
        for step in range(19):
            self.assertFalse(controller.observe(0.2, step=step).trigger_replan)
        decision = controller.observe(0.2, step=19)
        self.assertTrue(decision.trigger_replan)
        self.assertEqual(controller.trigger_count, 1)
        self.assertEqual(controller.low_value_steps, 20)

    def test_fifty_steps_after_trigger_are_in_cooldown(self):
        controller = SearchValueDecisionController(DECISION_CONFIG)
        for step in range(20):
            first = controller.observe(0.2, step=step)
        self.assertTrue(first.trigger_replan)
        for step in range(20, 70):
            with self.subTest(step=step):
                self.assertFalse(
                    controller.observe(0.2, step=step).trigger_replan
                )
        self.assertTrue(controller.observe(0.2, step=70).trigger_replan)

    def test_disabled_mode_is_behaviorally_baseline(self):
        controller = SearchValueDecisionController(
            {
                "enabled": False,
                "threshold": 0.35,
                "patience": 20,
                "cooldown": 50,
            }
        )
        for step in range(100):
            decision = controller.observe(0.0, step=step)
            self.assertFalse(decision.trigger_replan)
        self.assertEqual(
            controller.summary(),
            {
                "enabled": False,
                "trigger_count": 0,
                "low_value_steps": 0,
                "mean_trigger_value": None,
                "replan_after_trigger": 0,
                "search_value_mean": None,
            },
        )

    def test_old_checkpoint_loads_with_fresh_decision_controller(self):
        search_value = {
            "enabled": True,
            "hidden_dim": 16,
            "horizon": 50,
            "loss_weight": 0.05,
            "threshold": 0.5,
        }
        old_config = _config(search_value)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_path = _save_checkpoint(
                _learner(old_config),
                PRRACReplayAdapter(8, search_value_config=search_value),
                root,
                old_config,
                1,
                global_step=0,
                update_step=0,
                replay_sample_count=0,
                optimizer_update_count=0,
                episode_rows=[],
                execution_rows=[],
                prrac_rows=[],
            )
            payload = torch.load(old_path, map_location="cpu", weights_only=True)
            payload["metadata"].pop("search_value_decision", None)
            payload.pop("search_value_decision_state", None)
            legacy_path = root / "legacy_without_search_value_decision.pt"
            torch.save(payload, legacy_path)

            new_config = copy.deepcopy(old_config)
            new_config["search_value_decision"] = dict(DECISION_CONFIG)
            learner = _learner(new_config)
            replay = PRRACReplayAdapter(8, search_value_config=search_value)
            restored = _load_checkpoint(
                legacy_path, learner, replay, new_config
            )
            self.assertTrue(
                restored["search_value_decision_controller_initialized"]
            )
            state = restored["search_value_decision_state"]
            self.assertTrue(state["config"]["enabled"])
            self.assertEqual(state["trigger_count"], 0)

    def test_worker_uses_low_value_to_request_existing_bser_reassessment(self):
        job = _worker_jobs(1)[0]
        search_value = {
            "enabled": True,
            "hidden_dim": 16,
            "horizon": 50,
            "loss_weight": 0.05,
            "threshold": 0.5,
        }
        job["search_value"] = search_value
        job["search_value_decision"] = {
            "enabled": True,
            "threshold": 0.35,
            "patience": 1,
            "cooldown": 50,
        }
        head_owner = PRRACMADDPG(
            architecture=job["architecture"],
            loss=job["loss"],
            search_value=search_value,
        )
        snapshot = head_owner.search_value_snapshot()
        for value in snapshot.values():
            value.fill(0.0)
        snapshot["network.4.bias"].fill(math.log(0.2 / 0.8))
        job["search_value_snapshot"] = snapshot
        metrics, _, _, _ = _collect_episode(job)
        self.assertEqual(metrics["search_value_trigger_count"], 1)
        self.assertTrue(metrics["search_value_forced_replan"])
        self.assertAlmostEqual(metrics["search_value_mean"], 0.2, places=5)

    def test_search_value_rollout_snapshot_is_spawn_serializable(self):
        search_value = {
            "enabled": True,
            "hidden_dim": 16,
            "horizon": 50,
            "loss_weight": 0.05,
            "threshold": 0.5,
        }
        learner = _learner(_config(search_value))
        snapshot = learner.search_value_snapshot()
        restored = ForkingPickler.loads(ForkingPickler.dumps(snapshot))
        loaded = _learner(_config(search_value))
        loaded.load_search_value_snapshot(restored)
        for key, expected in snapshot.items():
            self.assertTrue((expected == loaded.search_value_snapshot()[key]).all())


if __name__ == "__main__":
    unittest.main()
