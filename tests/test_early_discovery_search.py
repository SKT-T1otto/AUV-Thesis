import copy
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import numpy as np

from chapter3_bser.greedy_solver import solve_joint_greedy
from chapter3_bser.objective import evaluate_objective
from chapter3_bser.online.early_discovery import EarlyDiscoveryAllocator, rank_candidates, estimated_arrival_time, resolve_early_discovery
from tests.test_search_value_guided_ranking import fixture


class EarlyDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.state, self.candidates, self.standby, self.context = fixture()
        self.positions = {a.agent_id: a.position for a in self.state.agents}
        self.scores = {c.key: 0.5 for c in self.candidates}

    def test_disabled_and_zero_lambda_identity_without_arrival_computation(self):
        for settings in ({"enabled": False, "lambda": 2}, {"enabled": True, "lambda": 0}):
            with patch("chapter3_bser.online.early_discovery.estimated_arrival_time", side_effect=AssertionError("identity")):
                ranked = rank_candidates(self.candidates, self.scores, {}, settings)
            for c, row in ranked:
                self.assertIs(row["final_score"], self.scores[c.key])
            original = solve_joint_greedy(self.candidates, (self.standby,), self.context)
            result = EarlyDiscoveryAllocator(settings)._solve_candidates(self.candidates, (self.standby,), self.context)
            self.assertEqual(result.selected_ids, original.selected_ids)
            self.assertEqual(float(result.objective).hex(), float(original.objective).hex())
            self.assertIs(result.standby, original.standby)

    def test_arrival_discount_order_and_stable_ties(self):
        ranked = rank_candidates(self.candidates, self.scores, self.positions, {"enabled": True, "lambda": 1})
        self.assertLess(ranked[0][1]["estimated_arrival_time"], ranked[1][1]["estimated_arrival_time"])
        self.assertGreater(ranked[0][1]["time_discount"], ranked[1][1]["time_discount"])
        zero = rank_candidates(tuple(reversed(self.candidates)), self.scores, {}, {"enabled": True, "lambda": 0})
        self.assertEqual([c.candidate_id for c, _ in zero], ["a", "b"])

    def test_existing_astar_length_precedes_straight_line(self):
        candidate = replace(self.candidates[0], path_length=12)
        self.assertEqual(estimated_arrival_time(candidate, candidate.waypoint, 2), 6)
        candidate = replace(candidate, path_length=None)
        self.assertEqual(estimated_arrival_time(candidate, candidate.waypoint, 2), 0)

    def test_candidate_arrays_are_unchanged(self):
        before = copy.deepcopy(self.candidates)
        rank_candidates(self.candidates, self.scores, self.positions, {"enabled": True, "lambda": 1})
        for old, new in zip(before, self.candidates):
            self.assertEqual(old.key, new.key)
            np.testing.assert_array_equal(old.path_points, new.path_points)
            np.testing.assert_array_equal(old.path_cell_indices, new.path_cell_indices)

    def test_active_changes_actual_selection_but_retains_original_objective(self):
        context = replace(self.context, detection_by_id={"a": np.array([.19]), "b": np.array([.2])})
        original = solve_joint_greedy(self.candidates, (self.standby,), context)
        allocator = EarlyDiscoveryAllocator({"enabled": True, "lambda": 1})
        result = allocator._solve_candidates(self.candidates, (self.standby,), context)
        self.assertEqual(original.selected_ids, ("b",))
        self.assertEqual(result.selected_ids, ("a",))
        self.assertEqual(result.objective, evaluate_objective(result.selected, result.standby, context))
        self.assertIs(result.standby, original.standby)
        self.assertTrue(allocator.ranking_diagnostics[0]["top_candidate_changed"])

    def test_found_restores_baseline(self):
        context = replace(self.context, state=replace(self.state, target_found=True))
        allocator = EarlyDiscoveryAllocator({"enabled": True, "lambda": 1})
        result = allocator._solve_candidates(self.candidates, (self.standby,), context)
        original = solve_joint_greedy(self.candidates, (self.standby,), context)
        self.assertEqual(result.selected_ids, original.selected_ids)
        self.assertEqual(allocator.candidate_diagnostics, [])

    def test_invalid_settings_rejected(self):
        for value in ({"enabled": "false"}, {"lambda": -1}, {"lambda": math.nan}, {"nominal_speed": 0}, {"typo": 1}):
            with self.assertRaises(ValueError):
                resolve_early_discovery(value)


if __name__ == "__main__":
    unittest.main()
