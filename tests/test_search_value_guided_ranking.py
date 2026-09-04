from dataclasses import replace
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.experiments.phase1c_prrac.search_value_guidance import (
    SearchValueGuidedBSERAllocator, SearchValueGuidedCandidateScore,
    aggregate_search_value_guidance, resolve_search_value_guidance,
)
from chapter3_bser.greedy_solver import solve_joint_greedy
from chapter3_bser.models.search_value_head import SearchValueHead
from chapter3_bser.models.prrac.prrac_maddpg import PRRACMADDPG
from chapter3_bser.objective import ObjectiveContext, evaluate_objective
from chapter3_bser.online.allocator import BSEROnlineAllocator
from chapter3_bser.types import SearchCandidate, StandbyCandidate
from tests.bser_test_utils import synthetic_state
from tests.prrac_evaluation_support import checkpoint_payload, worker_jobs, write_checkpoint


def fixture():
    state = synthetic_state()
    candidates = tuple(
        SearchCandidate(0, name, (x, 0.5, 1.0), np.array([state.agents[0].position, (x, 0.5, 1.0)]),
                        np.array([0]), x, x, x, "test")
        for name, x in (("a", 1.0), ("b", 3.0))
    )
    standby = StandbyCandidate("y", (2.5, 2.5, 1.0), np.array([[2.5, 2.5, 1.0]]),
                               np.array([0]), 0.0, 0.0, 0.0, "test")
    context = ObjectiveContext(state, candidates, (standby,), np.array([1.0]),
                               {"a": np.array([0.20]), "b": np.array([0.19])},
                               {"y": np.array([1.0])}, {"y": np.array([1.0])}, 1e-12)
    return state, candidates, standby, context


def scorer(config=None):
    head = SearchValueHead(hidden_dim=8)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.network[0].weight[0, 6] = 1.0
        head.network[2].weight[0, 0] = 1.0
        head.network[4].weight[0, 0] = 1.0
        head.network[4].bias[0] = -1.0
    return SearchValueGuidedCandidateScore.from_snapshot(
        {"enabled": True} if config is None else config,
        snapshot={key: value.numpy().copy() for key, value in head.state_dict().items()},
        head_config={"enabled": True, "hidden_dim": 8}, max_steps=400,
    )


class SearchValueGuidedRankingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.state, self.candidates, self.standby, self.context = fixture()

    def _solve(self, value_scorer, context=None):
        context = self.context if context is None else context
        value_scorer.observe_state(np.zeros((4, 28), np.float32), context.state)
        return SearchValueGuidedBSERAllocator(value_scorer)._solve_candidates(
            self.candidates, (self.standby,), context,
        )

    def test_disabled_ranking_is_original_bser(self):
        value_scorer = SearchValueGuidedCandidateScore()
        baseline = solve_joint_greedy(self.candidates, (self.standby,), self.context)
        with patch.object(value_scorer, "estimate_candidate_value", side_effect=AssertionError("disabled inference")):
            result = self._solve(value_scorer)
        self.assertEqual(result.selected_ids, baseline.selected_ids)
        self.assertEqual(result.objective, baseline.objective)
        self.assertIs(result.standby, baseline.standby)

    def test_enabled_changes_actual_greedy_choice_not_objective_or_executor(self):
        value_scorer = scorer()
        result = self._solve(value_scorer)
        self.assertEqual(result.selected_ids, ("b",))
        self.assertEqual(result.objective, evaluate_objective(result.selected, self.standby, self.context))
        self.assertAlmostEqual(result.objective, 0.19)
        self.assertIs(result.standby, self.standby)
        self.assertEqual(value_scorer.metrics()["ranking_changed_count"], 1)
        self.assertEqual(value_scorer.metrics()["allocation_changed_count"], 1)
        self.assertEqual(value_scorer.metrics()["selected_value_rank"], 1)

    def test_zero_weight_is_exact_baseline_without_head(self):
        value_scorer = SearchValueGuidedCandidateScore.from_snapshot(
            {"enabled": True, "weight": 0}, snapshot=None, head_config=None,
        )
        baseline = solve_joint_greedy(self.candidates, (self.standby,), self.context)
        result = self._solve(value_scorer)
        self.assertEqual(result.selected_ids, baseline.selected_ids)
        self.assertEqual(result.objective, baseline.objective)
        self.assertEqual(value_scorer.candidate_count, 0)

    def test_probability_range_and_clip(self):
        value_scorer = scorer({"enabled": True, "clip_min": 0.2, "clip_max": 0.8})
        for scale in (-1e4, 0, 1e4):
            value = value_scorer.estimate_candidate_value(np.full(34, scale))
            self.assertGreaterEqual(value, 0.2)
            self.assertLessEqual(value, 0.8)
        with self.assertRaises(ValueError):
            value_scorer.estimate_candidate_value(np.full(34, np.nan))

    def test_candidate_feature_is_distinct_and_does_not_mutate_actor_observation(self):
        value_scorer = scorer()
        observations = np.arange(112, dtype=np.float32).reshape(4, 28)
        original = observations.copy()
        value_scorer.observe_state(observations, self.state)
        a, b = [value_scorer.candidate_feature(item, self.state) for item in self.candidates]
        np.testing.assert_array_equal(observations, original)
        unchanged = [index for index in range(34) if index not in (*range(6, 12), 15, 17)]
        np.testing.assert_array_equal(a[unchanged], b[unchanged])
        self.assertNotEqual(value_scorer.estimate_candidate_value(a), value_scorer.estimate_candidate_value(b))
        self.assertEqual(a.shape, (34,))

    def test_base_gap_larger_than_auxiliary_bound_cannot_be_overturned(self):
        context = replace(self.context, detection_by_id={"a": np.array([0.5]), "b": np.array([0.19])})
        self.assertEqual(self._solve(scorer(), context).selected_ids, ("a",))

    def test_zero_original_gain_never_becomes_eligible(self):
        context = replace(self.context, detection_by_id={"a": np.array([0.0]), "b": np.array([0.0])})
        self.assertEqual(self._solve(scorer(), context).selected_ids, ())

    def test_after_found_bypasses_head(self):
        value_scorer = scorer()
        context = replace(self.context, state=replace(self.state, target_found=True))
        with patch.object(value_scorer, "estimate_candidate_value", side_effect=AssertionError("post-found inference")):
            self.assertEqual(self._solve(value_scorer, context).selected_ids, ("a",))

    def test_full_and_partial_allocator_paths_and_unaffected_assignments(self):
        value_scorer = scorer()
        value_scorer.observe_state(np.zeros((4, 28)), self.state)
        allocator = SearchValueGuidedBSERAllocator(value_scorer)
        current = allocator.allocate(self.state)
        self.assertGreater(value_scorer.candidate_count, 0)
        before = value_scorer.candidate_count
        result, valid, _ = allocator.allocate_partial(
            self.state, current, affected_searcher_ids=(0,), executor_affected=False, trigger_reason="test",
        )
        self.assertTrue(valid)
        self.assertGreater(value_scorer.candidate_count, before)
        self.assertIs(result.executor_assignment, current.executor_assignment)
        for item in current.search_assignments:
            if item.agent_id != 0:
                self.assertEqual(next(a for a in result.search_assignments if a.agent_id == item.agent_id), item)
        disabled = SearchValueGuidedBSERAllocator(SearchValueGuidedCandidateScore()).allocate(self.state)
        self.assertEqual(disabled.allocation_sha256, BSEROnlineAllocator().allocate(self.state).allocation_sha256)

    def test_snapshot_frozen_exact_and_rng_unchanged(self):
        head = SearchValueHead(hidden_dim=8)
        snapshot = {key: value.numpy().copy() for key, value in head.state_dict().items()}
        rng = torch.get_rng_state().clone()
        value_scorer = SearchValueGuidedCandidateScore.from_snapshot(
            {"enabled": True}, snapshot=snapshot, head_config={"enabled": True, "hidden_dim": 8},
        )
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        value_scorer.estimate_candidate_value(np.zeros(34))
        for key, value in value_scorer.head.state_dict().items():
            np.testing.assert_array_equal(value.numpy(), snapshot[key])
        self.assertFalse(value_scorer.head.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in value_scorer.head.parameters()))

    def test_rejected_proposal_is_not_reported_as_installed_change(self):
        value_scorer = scorer()
        self._solve(value_scorer)
        old = SimpleNamespace(search_assignments=(SimpleNamespace(agent_id=0, candidate_id="a"),))
        new = SimpleNamespace(search_assignments=(SimpleNamespace(agent_id=0, candidate_id="b"),))
        value_scorer.record_installed(old)
        self.assertEqual(value_scorer.accepted_search_change_count, 0)
        self._solve(value_scorer)
        value_scorer.record_installed(new, accepted=False)
        self.assertEqual(value_scorer.accepted_search_change_count, 0)
        self._solve(value_scorer)
        value_scorer.record_installed(new)
        value_scorer.record_installed(new)
        self.assertEqual(value_scorer.accepted_search_change_count, 1)

    def test_checkpoint_head_roundtrip_is_read_only(self):
        payload = checkpoint_payload()
        model = PRRACMADDPG(architecture=payload["metadata"]["architecture"],
                           loss=payload["metadata"]["loss"],
                           search_value={"enabled": True, "hidden_dim": 8})
        payload["metadata"]["search_value"] = model.search_value_config
        payload["prrac_training_state"] = model.training_state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = write_checkpoint(Path(directory) / "head.pt", payload)
            original = path.read_bytes()
            config = {"execution_runtime_revision": "dynamic_public_intercept_v2_1",
                      "search_value_guidance": {"enabled": True}}
            loaded, _ = evaluator.load_prrac_checkpoint(path, config=config)
            frozen = SearchValueGuidedCandidateScore.from_snapshot(
                config["search_value_guidance"], snapshot=loaded.search_value_snapshot(),
                head_config=loaded.search_value_config,
            )
            feature = np.ones(34, np.float32)
            with torch.no_grad():
                expected = model.search_value_head(torch.tensor(feature)).item()
            self.assertEqual(frozen.estimate_candidate_value(feature), expected)
            self.assertEqual(original, path.read_bytes())

    def test_config_defaults_and_validation(self):
        self.assertFalse(resolve_search_value_guidance()["enabled"])
        config = evaluator._load_config(evaluator.ROOT / "configs/chapter3/bser_phase1c_search_value_guided.json")
        self.assertTrue(config["search_value_guidance"]["enabled"])
        self.assertFalse(config["training_update"])
        self.assertEqual((config["observation_dim"], config["action_dim"], config["critic_dim"]), (28, 3, 124))
        for config in ({"weight": 0.11}, {"weight": -0.1}, {"clip_max": 2}, {"clip_min": 0.9, "clip_max": 0.2}, {"weight": float("nan")}):
            with self.assertRaises(ValueError):
                resolve_search_value_guidance(config)

    def test_legacy_checkpoint_still_loads_but_cannot_supply_active_guidance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_checkpoint(Path(directory) / "old.pt")
            learner, _ = evaluator.load_prrac_checkpoint(path)
            self.assertIsNone(learner.search_value_head)
            config = {"execution_runtime_revision": "dynamic_public_intercept_v2_1",
                      "search_value_guidance": {"enabled": True}}
            with self.assertRaisesRegex(ValueError, "checkpoint search_value_head"):
                evaluator.load_prrac_checkpoint(path, config=config)

    def test_worker_uses_checkpoint_head_without_update_and_disabled_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_checkpoint(Path(directory) / "old.pt", checkpoint_payload())
            job = worker_jobs(path, count=1)[0]
            baseline = evaluator._evaluate_episode_job(job)
            disabled = copy.deepcopy(job)
            disabled["config"]["search_value_guidance"] = {"enabled": False}
            off = evaluator._evaluate_episode_job(disabled)
            off["episode"].pop("search_value_guidance")
            self.assertEqual(baseline, off)
            guided = copy.deepcopy(job)
            guided["config"]["search_value_guidance"] = {"enabled": True}
            guided["search_value_config"] = {"enabled": True, "hidden_dim": 8}
            guided["search_value_snapshot"] = {
                key: value.numpy().copy() for key, value in scorer().head.state_dict().items()
            }
            with patch.object(evaluator.PRRACMADDPG, "update", side_effect=AssertionError("no training")):
                result = evaluator._evaluate_episode_job(guided)
            self.assertGreater(result["episode"]["search_value_guidance"]["candidate_count"], 0)
            artifact = aggregate_search_value_guidance([result["episode"]])
            output = Path(directory) / "search_value_guidance_metrics.json"
            evaluator._write_json(output, artifact)
            self.assertEqual(json.loads(output.read_text())["enabled"], True)

    def test_aggregate_is_candidate_weighted_not_episode_weighted(self):
        one = scorer().metrics()
        one.update(candidate_count=1, mean_search_value=0.2, selected_candidate_count=1,
                   mean_selected_candidate_value=0.2, selected_value_rank=2)
        two = dict(one, candidate_count=3, mean_search_value=0.8)
        combined = aggregate_search_value_guidance([{"search_value_guidance": one}, {"search_value_guidance": two}])
        self.assertAlmostEqual(combined["mean_search_value"], 0.65)
        self.assertEqual(combined["selected_value_rank"], 2)

    def test_evaluation_writer_emits_diagnostics_and_csv_roundtrip(self):
        from chapter3_bser.experiments.phase1c_prrac.evaluation_metrics import aggregate_checkpoint
        from chapter3_bser.experiments.phase1c_prrac.execution_continuity import ExecutionVariant
        from tests.test_searcher_residual_paired_evaluation import row as search_row

        row = search_row("full_prrac", "scenario-a", True, True)
        row.update(checkpoint_episode=12, checkpoint_schema=evaluator.CHECKPOINT_SCHEMA,
                   search_continuity_diagnostics_schema=evaluator.SEARCH_CONTINUITY_SCHEMA,
                   router_confusion_matrix=[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                   collision_episode=False, failure_stage="SUCCESS",
                   search_value_guidance=scorer().metrics())
        with tempfile.TemporaryDirectory() as directory, patch.object(evaluator, "_plot"), patch.object(evaluator, "_plot_execution_variants"), patch.object(evaluator, "_plot_search_recovery"):
            output = Path(directory)
            evaluator._write_outputs(
                output, checkpoint_paths=[Path("checkpoint.pt")], scenarios=[{"scenario_id": "scenario-a"}],
                modes=("full_prrac",), execution_variants=(ExecutionVariant.B1_ATOMIC_LAST_VALID,),
                episode_rows=[row], summary_rows=[aggregate_checkpoint([row], row)],
                trace_rows=[], trace_index=[], progress={"completed": []},
            )
            artifact = json.loads((output / evaluator.SEARCH_VALUE_GUIDANCE_OUTPUT).read_text())
            self.assertTrue(artifact["enabled"])
            self.assertEqual(artifact["episodes"][0]["scenario_id"], "scenario-a")
            restored = evaluator._read_csv(output / "episode_evaluation.csv")
            self.assertEqual(restored[0]["search_value_guidance"], row["search_value_guidance"])


if __name__ == "__main__":
    unittest.main()
