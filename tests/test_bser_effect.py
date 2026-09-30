"""Mechanism tests with counterexamples, not reimplementations of the solver."""
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from chapter3_bser.objective import ObjectiveContext, evaluate_objective
from chapter3_bser.types import SearchCandidate, StandbyCandidate
from chapter3_bser.online.types import OnlineAllocation, ExecutorAssignment
from chapter3_bser.online.config import load_phase1b2_config
from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.greedy_solver import solve_joint_greedy
from chapter3_bser.experiments.bser_effect_v1.solver import solve
from chapter3_bser.experiments.bser_effect_v1.allocator import EffectAllocator
from chapter3_bser.experiments.bser_effect_v1.controller import EffectPolicy, ScoredWaypoints
from chapter3_bser.online.waypoint_manager import WaypointManager


def candidate(agent, name, x):
    return SearchCandidate(agent, name, (x,0.,0.), np.array([[0.,0.,0.],[x,0.,0.]]),
                           np.array([0,1]), x, x, x, "fixture")


def standby(name, x):
    return StandbyCandidate(name, (x,0.,0.), np.array([[0.,0.,0.],[x,0.,0.]]), np.array([0,1]), x,x,x,"fixture")


def toy():
    a,b,c = candidate(0,"a",1.), candidate(0,"b",3.), candidate(1,"c",2.)
    ys = (standby("y0",0.), standby("y1",5.))
    state = SimpleNamespace(step=12, target_found=False, searcher_ids=(0,1), executor_id=3)
    context = ObjectiveContext(state, (a,b,c), ys, np.array([.6,.4]),
        {"a":np.array([1.,0.]),"b":np.array([0.,1.]),"c":np.array([.05,.05])},
        {"y0":np.array([.05,1.]),"y1":np.array([.9,.1])},
        {"y0":np.array([10.,1.]),"y1":np.array([1.,10.])},1e-12)
    return (a,b,c), ys, context


class SolverTests(unittest.TestCase):
    def test_four_arms_separate_response_weighting_and_standby(self):
        candidates, ys, ctx = toy()
        results = {a:solve(a,candidates,ys[:1] if a in ("D0","D1") else ys,ctx)
                   for a in ("D0","D1","D2","D3")}
        self.assertIn("a",results["D0"].selected_ids)
        self.assertIn("b",results["D1"].selected_ids)
        self.assertEqual(results["D0"].selected_ids,results["D2"].selected_ids)
        self.assertEqual(results["D2"].standby.candidate_id,"y1")
        self.assertEqual(results["D3"].selected_ids,solve_joint_greedy(candidates,ys,ctx).selected_ids)
        self.assertEqual(results["D3"].standby.key,solve_joint_greedy(candidates,ys,ctx).standby.key)

    def test_d2_search_cannot_be_changed_by_response_weights(self):
        cs,ys,ctx=toy()
        changed=replace(ctx,response_weight_by_id={"y0":np.array([0.,1.]),"y1":np.array([0.,.9])})
        self.assertEqual(solve("D2",cs,ys,ctx).selected_ids,solve("D2",cs,ys,changed).selected_ids)
        self.assertNotEqual(solve("D3",cs,ys,ctx).selected_ids,solve("D3",cs,ys,changed).selected_ids)

    def test_frozen_zero_gain_agent_retained_in_all_partial_solves(self):
        cs,ys,ctx=toy()
        ctx=replace(ctx,detection_by_id=dict(ctx.detection_by_id,c=np.zeros(2)))
        for arm in ("D0","D1","D2","D3"):
            result=solve(arm,cs[:2],ys[:1] if arm in ("D0","D1") else ys,ctx,frozen=(cs[2],))
            self.assertIn("c",result.selected_ids)
            self.assertEqual(len({c.agent_id for c in result.selected}),len(result.selected))


class AllocationTests(unittest.TestCase):
    def allocator(self, arm):
        return EffectAllocator(SimpleNamespace(effect_counts=Counter()),None,arm,(0.,0.,0.))

    def test_fixed_anchor_unreachable_stays_unreachable_without_belief_fallback(self):
        cs,ys,ctx=toy()
        invalid=ExecutorAssignment(3,(0.,0.,0.),(),float("inf"),"fixed",False)
        for arm in ("D0","D1"):
            alloc=self.allocator(arm)
            with patch.object(alloc.execution,"_assignment",return_value=invalid), \
                 patch.object(alloc.execution,"assign_belief_peak",side_effect=AssertionError("not allowed")):
                pool,executor=alloc.standby_pool(ctx.state)
                self.assertFalse(executor.reachable)
                self.assertEqual(executor.path,())
                self.assertEqual(pool[0].waypoint,(0.,0.,0.))

    def test_unaffected_executor_route_preserved_even_when_anchor_cannot_be_replanned(self):
        cs,ys,ctx=toy()
        item=ExecutorAssignment(3,(0.,0.,0.),((0.,0.,0.),),0.,"old",True)
        current=OnlineAllocation((),item,0.,0.,0.,"fixture")
        for arm in ("D0","D1","D2","D3"):
            alloc=self.allocator(arm)
            with patch.object(alloc,"anchor_assignment",side_effect=AssertionError("unaffected")):
                pool,executor=alloc.standby_pool(ctx.state,current,movable=False)
                self.assertIs(executor,item)

    def test_partial_omitted_agents_are_fixed_and_missing_candidates_rejected(self):
        cs,ys,ctx=toy()
        alloc=self.allocator("D3")
        executor=alloc.execution.assign_standby(ctx.state,ys[0])
        current=OnlineAllocation(tuple(alloc._search_assignment(c) for c in (cs[0],cs[2])),executor,.8,.8,1.,"old")
        with patch.object(alloc,"search_pool",return_value=((cs[1],),0,())), \
             patch.object(alloc,"standby_pool",return_value=(ys[:1],executor)), \
             patch("chapter3_bser.experiments.bser_effect_v1.allocator.build_objective_context",return_value=ctx):
            proposal,ok,_=alloc.allocate_partial(ctx.state,current,affected_searcher_ids=(0,),trigger_reason="test")
        self.assertTrue(ok)
        self.assertEqual(proposal.search_assignments[1],current.search_assignments[1])
        self.assertEqual(proposal.executor_assignment,current.executor_assignment)
        with patch.object(alloc,"search_pool",return_value=((),0,())):
            same,ok,_=alloc.allocate_partial(ctx.state,current,affected_searcher_ids=(0,),trigger_reason="test")
        self.assertFalse(ok)
        self.assertIs(same,current)

    def test_d2_reselects_standby_after_retaining_old_search_and_rescores(self):
        cs,ys,ctx=toy()
        alloc=self.allocator("D2")
        alloc.last_standbys=ys
        alloc.standby_may_change=True
        old=OnlineAllocation((alloc._search_assignment(cs[1]),),alloc.execution.assign_standby(ctx.state,ys[0]),.4,.4,1.,"old")
        proposal=OnlineAllocation((alloc._search_assignment(cs[0]),),alloc.execution.assign_standby(ctx.state,ys[1]),.6,.6,1.,"new")
        controller=SimpleNamespace(scoring_state=ctx.state,allocator=alloc)
        manager=ScoredWaypoints(controller,WaypointManager(agent_cooldown_steps=20))
        manager.last_switch_step[0]=10
        with patch("chapter3_bser.experiments.bser_effect_v1.allocator.build_objective_context",return_value=ctx):
            actual=manager.stabilize(old,proposal,affected_agent_ids=(0,),step=12)
        self.assertEqual(actual.search_assignments,old.search_assignments)
        self.assertEqual(actual.executor_assignment.target_region,ys[0].waypoint)
        self.assertAlmostEqual(actual.objective_value,.4)
        self.assertNotEqual(actual.objective_value,proposal.objective_value)
        self.assertIs(alloc.rescore(SimpleNamespace(target_found=True),actual),actual)

    def test_acceptance_has_equal_opportunities_but_keeps_cooldown_and_post_found_policy(self):
        policy=EffectPolicy(load_phase1b2_config())
        event=BSEREvent.BELIEF_SHIFT
        self.assertTrue(policy.decide((event,),.001,.9,20).should_replan)
        self.assertFalse(policy.decide((),1.,0.,20).should_replan)
        policy.mark_attempt(20,event)
        self.assertFalse(policy.decide((event,),1.,0.,21).should_replan)
        policy.search_enabled=False
        self.assertFalse(policy.decide((event,),.001,.9,100).should_replan)


if __name__ == "__main__":
    unittest.main()
