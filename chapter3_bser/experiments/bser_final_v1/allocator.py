"""Same-pool D2 reference, installable proposals, unchanged event lifecycle."""
from collections import Counter
from dataclasses import replace
import hashlib
import time

from chapter3_bser.experiments.bser_effect_v1.allocator import EffectAllocator
from chapter3_bser.experiments.bser_effect_v1.solver import solve
from chapter3_bser.objective import build_objective_context, evaluate_objective, response_diagnostics
from chapter3_bser.online.types import OnlineAllocation
from chapter3_bser.online.waypoint_manager import WaypointManager
from .forecast import Incomparable
from .options import ARMS, SETTINGS
from .scoring import Scorer, allocation_key
from .solver import choose


class FinalAllocator(EffectAllocator):
    def __init__(self, owner, config, arm, anchor):
        super().__init__(owner, config, "D2", anchor)
        self.final_arm = arm
        self.final_audit = []
        self.call_current = None
        self.call_affected = ()

    def allocate(self, state, *, trigger_reason="online"):
        controller = getattr(self.owner, "controller", None)
        self.call_current = None if controller is None else controller.current_allocation
        self.call_affected = tuple(state.searcher_ids)
        return super().allocate(state, trigger_reason=trigger_reason)

    def allocate_partial(self, state, current, *, affected_searcher_ids=(), executor_affected=False, trigger_reason):
        self.call_current, self.call_affected = current, tuple(affected_searcher_ids)
        return super().allocate_partial(state, current, affected_searcher_ids=affected_searcher_ids,
            executor_affected=executor_affected, trigger_reason=trigger_reason)

    def preview(self, state, allocation, trigger):
        # Recovery installs its partial allocation directly in the unchanged
        # outer controller; ordinary events pass through WaypointManager.
        if self.call_current is None or trigger == "SAFE_SEARCH_RECOVERY":
            return allocation
        manager = self.owner.controller.inner.waypoints
        return WaypointManager.stabilize(manager, self.call_current, allocation,
            affected_agent_ids=self.call_affected, step=state.step)

    def _proposal(self, state, candidates, frozen, standbys, executor, trigger):
        started = time.perf_counter()
        self.last_standbys, self.standby_may_change = tuple(standbys), executor is None
        context = build_objective_context(state, tuple(candidates)+tuple(frozen), standbys, self.config)
        result = solve("D2", candidates, standbys, context, frozen=frozen)
        raw = OnlineAllocation(tuple(self._search_assignment(c) for c in result.selected),
            executor or self.execution.assign_standby(state, result.standby), result.objective,
            evaluate_objective(result.selected, result.standby, context, search_only=True),
            response_diagnostics(result.selected, result.standby, context).conditional_reachable_response_time,
            trigger, result.status)
        arm = ARMS[self.final_arm]
        scorer = Scorer(self, state, context, standbys, executor, arm)
        reference = self.preview(state, raw, trigger)
        # Precisely reproduce D2's post-stabilization standby re-selection.
        if self.call_current is not None and trigger != "SAFE_SEARCH_RECOVERY":
            reference = scorer.sequential(reference)
        reference = self.rescore(state, reference)
        observer = getattr(self, "_audit_candidate_generation_observer", None)
        if observer is not None:
            observer(state, candidates, standbys, scope="F_proposal", agent_ids=tuple(c.agent_id for c in candidates))
        record = dict(kind="final_proposal", step=int(state.step), trigger=trigger, arm=self.final_arm,
            candidate_counts={str(i): sum(c.agent_id == i for c in candidates) for i in state.searcher_ids},
            standby_count=len(standbys), affected=list(self.call_affected),
            reference_ids=[x.candidate_id for x in reference.search_assignments],
            reference_standby=list(reference.executor_assignment.target_region),
            reference_score=None, selected_score=None, candidate_model_failures={}, rejections={},
            combinations=0, selected_is_reference=True, fallback_reason=None, diagnostic_complete=False)
        selected = reference
        try:
            if {x.agent_id for x in reference.search_assignments} != set(state.searcher_ids):
                raise Incomparable("incomplete_D2_search_assignment")
            ref = scorer.score(reference)
            record["reference_score"] = ref.record()
            if ref.near <= SETTINGS["score_floor"] or (ref.response is not None and ref.response <= SETTINGS["score_floor"]):
                raise Incomparable("near_zero_D2_score")
            groups = []
            frozen_by_id = {x.agent_id: x for x in frozen}
            for i in state.searcher_ids:
                group = (frozen_by_id[i],) if i in frozen_by_id else tuple(sorted(
                    (x for x in candidates if x.agent_id == i), key=lambda c: c.key))
                if not group:
                    raise Incomparable("missing_candidate_group")
                groups.append(group)
            failures = Counter()
            def evaluate(items, y):
                scorer.counts["combination_requests"] += 1
                proposal = OnlineAllocation(tuple(self._search_assignment(c) for c in items),
                    executor or self.execution.assign_standby(state, y), 0., 0., float("inf"), trigger)
                proposal = self.preview(state, proposal, trigger)
                if arm["response"] == "sequential":
                    proposal = scorer.sequential(proposal)
                if arm["search_guard"] and scorer.full(proposal)+SETTINGS["numerical_tolerance"] < ref.full:
                    return None, "full_search_regression"
                try:
                    score = scorer.score(proposal, reference=ref)
                except Incomparable as exc:
                    reason = str(exc)
                    if reason in ("forecast_budget", "response_query_budget", "combination_budget"):
                        raise
                    if reason != "near_search_regression" and not reason.startswith("risk_regression_agent_"):
                        failures[reason] += 1
                    return None, reason
                return (proposal, score), None
            # F2's standby is determined after search stabilization, not a second
            # optimization variable. Avoid evaluating duplicate search sets 4x.
            ys = standbys[:1] if arm["response"] == "sequential" else standbys
            seed_by_id = {c.agent_id: c for c in result.selected}
            winner, rejected, evaluated = choose(groups, ys, ref, evaluate, arm,
                seed_items=tuple(seed_by_id[i] for i in state.searcher_ids))
            record.update(candidate_model_failures=dict(failures), rejections=rejected,
                          combinations=evaluated, diagnostic_complete=True)
            if winner is None:
                record["fallback_reason"] = "no_admissible_material_gain"
            else:
                selected, score = winner
                record["selected_score"] = score.record()
                record["selected_is_reference"] = allocation_key(selected) == allocation_key(reference)
        except Incomparable as exc:
            record["fallback_reason"] = str(exc)
            selected = reference  # Work-budget failures do not select a partial-prefix winner.
        selected = self.rescore(state, selected)
        record.update(selected_ids=[x.candidate_id for x in selected.search_assignments],
            selected_standby=list(selected.executor_assignment.target_region),
            selected_route_signature=hashlib.sha256(repr(allocation_key(selected)).encode()).hexdigest(),
            counts=dict(scorer.counts), timings=dict(scorer.timings),
            total_planning_seconds=time.perf_counter()-started)
        record["combinations"] = scorer.counts["combination_requests"]
        # Save the first-step prediction for a later physical model-error check.
        predicted = {}
        for i in (*state.searcher_ids, state.executor_id):
            item = selected.executor_assignment if i == state.executor_id else next((x for x in selected.search_assignments if x.agent_id == i), None)
            from .scoring import route_key
            f = scorer.forecasts.get((i, route_key(item)))
            if f is not None and not isinstance(f, Exception):
                predicted[str(i)] = dict(position=f.next_position.tolist(), velocity=f.next_velocity.tolist(),
                    disturbance_status=f.disturbance_status, hold=f.hold)
        record["next_step_prediction"] = predicted
        self.final_audit.append(record)
        # Existing controller instrumentation continues to see its proposal row.
        self.audit.append(dict(kind="proposal", step=int(state.step), selected_ids=record["selected_ids"],
            pure_search_ids=record["reference_ids"], search_differs_from_pure=record["selected_ids"] != record["reference_ids"],
            installed=False))
        return selected

    def rescore(self, state, allocation, *, finalize=False):
        # Scores used by the common event controller keep D2's original scale.
        # A selected F standby must not be overwritten by D2 during commit.
        if state.target_found or allocation.search_frozen:
            return allocation
        search = tuple(self._frozen_search_candidate(c) for c in allocation.search_assignments)
        y = self._frozen_standby_candidate(allocation)
        context = build_objective_context(state, search, (y,), self.config)
        value = evaluate_objective(search, y, context, search_only=True)
        return replace(allocation, objective_value=value, detection_probability=value,
            response_time=response_diagnostics(search, y, context).conditional_reachable_response_time)
