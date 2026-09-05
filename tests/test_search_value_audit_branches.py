import copy
from dataclasses import replace
import inspect
import unittest
import numpy as np

from chapter3_bser.experiments.phase1c_prrac.search_value_audit.runtime_audit import alternative_c, solution_geometry
from chapter3_bser.experiments.phase1c_prrac.search_value_audit.branch_audit import installed_signature, root_decision, replay_equal, validate_branches, needs_root_probe
from chapter3_bser.experiments.phase1c_prrac.search_value_audit.metrics import paired_outcomes
from chapter3_bser.experiments.phase1c_prrac.search_value_audit import features, runtime_audit
from chapter3_bser.greedy_solver import solve_joint_greedy
from chapter3_bser.experiments.phase1c_prrac.search_value_guidance import SearchValueGuidedBSERAllocator
from tests.test_search_value_guided_ranking import fixture, scorer
from tests.search_value_audit_support import synthetic_branch


class BranchTests(unittest.TestCase):
    def test_init_boundary_AA_reverse_order_one_intervention(self):
        forward = {b: synthetic_branch(b, 0) for b in ("A", "B", "C")}
        reverse = {b: synthetic_branch(b, 0) for b in ("C", "B", "A")}
        repeat = synthetic_branch("A", 0)
        self.assertTrue(replay_equal(forward["A"], repeat))
        for b in forward:
            self.assertTrue(replay_equal(forward[b], reverse[b]))
            audit = forward[b]["audit"]
            self.assertEqual(audit["guided_after_root_count"], 0)
            self.assertEqual(audit["guided_root_count"], int(b == "B"))
            self.assertEqual(root_decision(audit, 0)["boundary"], "initialize")
            self.assertTrue(root_decision(audit, 0)["proposal_accepted"])
        self.assertEqual(forward["A"]["audit"]["root_fingerprint"], forward["B"]["audit"]["root_fingerprint"])
        self.assertNotEqual(installed_signature(root_decision(forward["A"]["audit"], 0)), installed_signature(root_decision(forward["B"]["audit"], 0)))

    def test_later_boundary_retains_acceptance_and_original_global_cutoff(self):
        # Real unmodified periodic/event policy at step 100; no forced acceptance.
        a = synthetic_branch("A", 100, cutoff=102)
        c = synthetic_branch("C", 100, cutoff=102)
        self.assertEqual(a["audit"]["root_fingerprint"], c["audit"]["root_fingerprint"])
        decision = root_decision(c["audit"], 100)
        self.assertIsNotNone(decision, (c["status"], c["root_reason"], [(d["step"], d["decision_reason"]) for d in c["audit"]["decisions"]]))
        self.assertEqual(decision["boundary"], "controller.step")
        self.assertFalse(decision["proposal_accepted"])
        self.assertEqual(c["audit"]["guided_after_root_count"], 0)
        # root=399 does not create an extra 400-step suffix in either runner.
        from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
        self.assertIn('range(int(config["max_steps"]))', inspect.getsource(evaluator._evaluate_episode_job))
        cutoff = synthetic_branch("A", 100, cutoff=400)
        self.assertEqual(cutoff["audit"]["observed_until"], 400)
        self.assertEqual(len(cutoff["audit"]["steps"]), 400)

    def test_no_C_does_not_manufacture_proposal(self):
        result = synthetic_branch("C", 0, third=False)
        self.assertEqual(result["status"], "C_UNAVAILABLE")
        self.assertIsNone(root_decision(result["audit"], 0))
        state, candidates, standby, context = fixture()
        baseline = solve_joint_greedy(candidates, (standby,), context)
        value = scorer()
        value.observe_state(np.zeros((4, 28)), state)
        guided = SearchValueGuidedBSERAllocator(value)._solve_candidates(candidates, (standby,), context)
        self.assertIsNone(alternative_c(candidates, baseline, guided, context))

    def test_ids_and_accepted_counter_alone_are_not_delivery(self):
        base = dict(installed_assignment_geometry={"0": dict(waypoint=[1, 0, 0], path=[[1, 0, 0]])},
                    public_guidance_geometry={}, effective_guidance_geometry={}, candidate_id="A", proposal_accepted=True,
                    historical_accepted_search_change_count=1)
        other = dict(base, candidate_id="B", historical_accepted_search_change_count=99)
        self.assertEqual(installed_signature(base), installed_signature(other))
        self.assertTrue(needs_root_probe(dict(proposal_accepted=True, step=20, allocation_proposal_changed=False, proposal_candidate_ids_changed=True)))
        other["effective_guidance_geometry"] = {"0": {"tracking": [2, 0, 0]}}
        self.assertNotEqual(installed_signature(base), installed_signature(other))
        self.assertFalse(validate_branches(dict(root=None), {})["passed"])

    def test_AB_equivalent_root_cannot_pass_delivery_gate(self):
        same = synthetic_branch("A", 0, cutoff=1)
        decision = root_decision(same["audit"], 0)
        root = dict(step=0, original_ON_decision=decision, A_probe_decision=decision,
                    root_fingerprint=same["audit"]["root_fingerprint"], prefix_hash=same["audit"]["prefix_hash"],
                    state_equal=True, prefix_equal=True)
        location = dict(root=root, checks={m: dict(passed=True, historical_reproduced=True) for m in ("OFF", "ON")})
        results = {b: copy.deepcopy(same) for b in ("A", "B", "C", "A_repeat", "A_reverse", "B_reverse", "C_reverse")}
        self.assertFalse(validate_branches(location, results)["checks"]["B_reproduces_ON_delivery"])

    def test_C_preserves_standby_and_best_original_objective(self):
        state, candidates, standby, context = fixture()
        extra = replace(candidates[0], candidate_id="c", waypoint=(2, 2, 1))
        candidates = (*candidates, extra)
        context = replace(context, candidates=candidates, detection_by_id={**context.detection_by_id, "c": np.array([.18])})
        baseline = solve_joint_greedy(candidates, (standby,), context)
        guided = replace(baseline, selected=(candidates[1],), objective=.19)
        c = alternative_c(candidates, baseline, guided, context)
        self.assertEqual(c.selected_ids, ("c",))
        self.assertIs(c.standby, baseline.standby)
        self.assertAlmostEqual(c.objective, .18)

    def test_partial_C_excludes_B_after_frozen_assignment_merge(self):
        from chapter3_bser.online.allocator import BSEROnlineAllocator
        from chapter3_bser.online.types import OnlineAllocation
        state, pair, standby, context = fixture()
        a, b = pair
        c = replace(b, candidate_id="c")  # Different ID, identical actual geometry.
        frozen = replace(a, agent_id=1, candidate_id="f", source="current_assignment")
        candidates = (a, b, c, frozen)
        context = replace(context, candidates=candidates, belief=np.array([.5, .5]),
                          detection_by_id={"a": np.array([1., 0.]), "b": np.array([0., .98]), "c": np.array([0., .97]), "f": np.array([1., 0.])},
                          response_weight_by_id={"y": np.ones(2)}, response_time_by_id={"y": np.ones(2)})
        baseline = solve_joint_greedy(candidates, (standby,), context)
        self.assertEqual(baseline.selected_ids, ("a",))
        guided = replace(baseline, selected=(b, frozen), objective=.99)
        allocator = BSEROnlineAllocator()
        current = OnlineAllocation(tuple(allocator._search_assignment(x) for x in (a, frozen)),
                                   allocator.execution.assign_standby(state, standby), .5, .5, 1., "test")
        self.assertIsNone(alternative_c(candidates, baseline, guided, context, affected={0}, current=current))

    def test_scoring_boundary_is_public_and_team_probabilities_not_summed(self):
        self.assertEqual(list(inspect.signature(features.pool_audit).parameters)[:4], ["candidates", "context", "baseline", "shadow"])
        for function in (features.pool_audit, alternative_c):
            source = inspect.getsource(function)
            self.assertNotIn("unwrapped", source)
            self.assertNotIn("hidden_target", source)
        rows = [dict(scenario_id="s", scenario_seed=1, branch=b, valid_pair=True, treatment_delivered=True,
                     found_50=False, found=True, success=False, searcher_collisions=0, max_collision_streak=0,
                     known_ratio_gain=0., travel_distance=1.) for b in ("A", "B", "C")]
        _, summary = paired_outcomes(rows)
        self.assertEqual(summary["root_count"], 1)
        self.assertEqual(summary["binary"]["found"]["both"], 1)
        self.assertEqual(summary["continuous"]["searcher_collisions"]["ties"], 1)
        self.assertIn("not_applicable", summary["probability_ranking_consistency"])


if __name__ == "__main__":
    unittest.main()
