"""Optional observer and single-decision intervention for the existing evaluator.

All scoring APIs accept only existing candidates and ObjectiveContext. Privileged
runtime access is confined to replay fingerprints and offline outcome accounting.
"""

from dataclasses import replace
import copy

import numpy as np
import torch

from chapter3_bser.greedy_solver import solve_joint_greedy
from chapter3_bser.objective import evaluate_objective, marginal_gain
from chapter3_bser.online.allocator import BSEROnlineAllocator
from chapter3_bser.experiments.phase1c_prrac.search_value_guidance import SearchValueGuidedCandidateScore
from . import MAX_STEPS
from .features import assignment_geometry, candidate_geometry, compare_feature, exact_feature_id, guidance_geometry, pool_audit
from .provenance import actor_hash, digest, json_value, weights_hash
from .state_fingerprint import component_fingerprint, rng_state, runtime_fingerprint


class BoundaryProbeComplete(Exception):
    """Intentional prefix-only diagnostic probe; never a completed outcome."""


class CUnavailable(Exception):
    """The existing candidate pool contains no distinct legal C proposal."""


class InvalidRoot(Exception):
    """A replayed root no longer generates the required search decision."""


def solution_geometry(solution, *, current=None, affected=None):
    if solution is None:
        return None
    values = {str(c.agent_id): dict(waypoint=list(c.waypoint), path=np.asarray(c.path_points).tolist()) for c in solution.selected}
    if current is not None:
        frozen = assignment_geometry(current)
        for key, value in frozen.items():
            if int(key) not in affected:
                values[key] = value
    return values


def alternative_c(candidates, baseline, guided, context, *, affected=None, current=None):
    if baseline.standby is None:
        return None
    if affected is not None and not set(affected).issubset({c.agent_id for c in baseline.selected}):
        return None  # A single replacement cannot repair a missing required agent.
    best, best_key = None, None
    geometry = lambda proposal: solution_geometry(proposal, current=current, affected=affected)
    a_signature, b_signature = digest(geometry(baseline)), digest(geometry(guided))
    def merged_objective(selected):
        if current is not None:
            chosen = {c.agent_id: c for c in selected}
            frozen = {c.agent_id: c for c in candidates if c.source == "current_assignment"}
            selected = tuple(chosen[a.agent_id] if a.agent_id in affected else frozen[a.agent_id]
                             for a in current.search_assignments)
        return evaluate_objective(selected, baseline.standby, context)
    for old in baseline.selected:
        if old.source == "current_assignment" or (affected is not None and old.agent_id not in affected):
            continue
        prefix = [c for c in baseline.selected if c.agent_id != old.agent_id]
        for candidate in sorted(candidates, key=lambda c: c.key):
            if candidate.agent_id != old.agent_id or candidate.key == old.key:
                continue
            if marginal_gain(prefix, candidate, baseline.standby, context) <= 1e-15:
                continue
            selected = tuple(sorted([*prefix, candidate], key=lambda c: c.key))
            proposal = replace(baseline, selected=selected, objective=merged_objective(selected))
            if digest(geometry(proposal)) in (a_signature, b_signature):
                continue
            key = tuple(c.key for c in selected)
            if best is None or proposal.objective > best.objective + 1e-15 or (abs(proposal.objective-best.objective) <= 1e-15 and key < best_key):
                best, best_key = proposal, key
    return best


class AuditAllocator(BSEROnlineAllocator):
    """Capture the actual generated pool through the existing solver extension.

    Allocation construction, executor assignment, partial merging and controller
    acceptance remain the inherited operations. No generator is called twice.
    """
    def __init__(self, delegate, observer):
        self.delegate, self.observer = delegate, observer
        self.config, self.execution = delegate.config, delegate.execution
        self._audit_candidate_generation_observer = observer.generated_pool
        self.request = {}

    def allocate(self, state, *, trigger_reason="online"):
        self.request = dict(scope="full", current=None, affected=None)
        proposal = super().allocate(state, trigger_reason=trigger_reason)
        self.observer.capture_proposal(proposal)
        return proposal

    def allocate_partial(self, state, current, *, affected_searcher_ids=(), executor_affected=False, trigger_reason):
        self.request = dict(scope="partial", current=current, affected=set(affected_searcher_ids))
        result = super().allocate_partial(state, current, affected_searcher_ids=affected_searcher_ids,
                                          executor_affected=executor_affected, trigger_reason=trigger_reason)
        self.observer.capture_proposal(result[0])
        return result

    def _solve_candidates(self, candidates, standby_candidates, context):
        actual = self.delegate._solve_candidates(candidates, standby_candidates, context)
        # Diagnostic computation on the captured immutable pool. It has no planner
        # service/cache access and does not alter the actual scorer's counters.
        return self.observer.pool(candidates, standby_candidates, context, actual, self.request)


class RuntimeAudit:
    def __init__(self, job, *, shadow=False, intervention=None, root_step=None, probe=False, capture=True):
        if intervention not in (None, "A", "B", "C") or (intervention is not None and (root_step is None or not 0 <= root_step < MAX_STEPS)):
            raise ValueError("single intervention requires A/B/C and a pre-action root in [0,399]")
        self.job, self.shadow_enabled = job, shadow
        self.intervention, self.root_step, self.probe, self.capture = intervention, root_step, probe, capture
        self.identity = dict(scenario_id=str(job["scenario"]["scenario_id"]), scenario_seed=int(job["scenario"]["scenario_seed"]))
        self.bridge = self.recovery = self.action_adapter = None
        self.predictions, self.consistency, self.representation, self.scores, self.decisions = [], [], [], [], []
        self.steps, self.boundaries, self.feature_vectors = [], {}, {}
        self.generations = []
        self.last_result = self.pending = None
        self.found_step, self.observed_until = None, 0
        self.guided_root_count, self.guided_after_root_count = 0, 0
        self.root_fingerprint = None
        self.suffix_collisions = np.zeros(3, dtype=int)
        self.streak, self.max_streak = np.zeros(3, dtype=int), 0
        self.distance, self.root_known, self.known_gain = 0.0, None, 0.0
        self.root_installed_signature, self.guidance_end, self.first_followup_replan = None, None, None
        self.trace_chain, self.prefix_chain = digest([]), None

    def bind(self, *, actor, env, provider, controller, scorer):
        self.actor, self.env, self.provider, self.controller, self.scorer = actor, env, provider, controller, scorer
        # Existing evaluator actors are eval-mode; freeze explicitly for audits.
        for agent in actor.agents:
            agent.actor.eval().requires_grad_(False)
        self.before_actor_hash = actor_hash(actor.policy_snapshot())
        self.before_head_hash = weights_hash(self.job["search_value_snapshot"])
        self.shadow = None
        if self.shadow_enabled or self.capture:
            self.shadow = SearchValueGuidedCandidateScore.from_snapshot(
                {"enabled": True, "weight": .1}, snapshot=self.job["search_value_snapshot"],
                head_config=self.job["search_value_config"], max_steps=MAX_STEPS)
        if self.capture:
            legacy = getattr(controller, "legacy", controller)
            legacy.allocator = AuditAllocator(legacy.allocator, self)

    def attach_guidance_runtime(self, bridge, recovery, action_adapter):
        self.bridge, self.recovery, self.action_adapter = bridge, recovery, action_adapter

    def fingerprint(self, state, observations, context):
        return runtime_fingerprint(self.env, self.provider, self.controller, self.bridge, self.recovery, self.scorer,
                                   context=context, observations=observations, action_adapter=self.action_adapter)

    def before_decision(self, state, observations, context, *, initialize):
        self.state, self.context = state, context
        self.observed_until = int(state.step)
        self.pending = None
        if self.shadow is not None:
            self.shadow.observe_state(observations, state,
                                      collision_flags=None if initialize else self.env.unwrapped.collision_flags)
        fingerprint = self.fingerprint(state, observations, context)
        self.boundaries[int(state.step)] = dict(fingerprint=fingerprint, prefix_hash=self.trace_chain)
        if self.root_step == int(state.step):
            self.root_fingerprint, self.prefix_chain = fingerprint, self.trace_chain
            self.root_known = float(np.mean(state.occupancy.known_mask))
            self.last_positions = np.array([a.position for a in state.agents[:3]])
        self.old_allocation = self.controller.current_allocation
        self.initialize = initialize

    def pool(self, candidates, standby_candidates, context, actual, request):
        if context.state.target_found:
            return actual
        baseline = solve_joint_greedy(candidates, standby_candidates, context)
        # During ON replay 'actual' is the unchanged guided algorithm. During D1
        # OFF it is baseline. No new guided allocator is installed for D1.
        index = len(self.decisions)
        representation, scores, pool_hash = pool_audit(candidates, context, baseline, self.shadow,
                                                      identity=self.identity, decision_index=index)
        self.representation.extend(representation)
        self.scores.extend(scores)
        current, affected = request["current"], request["affected"]
        guided = actual
        at_root = self.root_step == int(context.state.step) and self.intervention is not None
        alternate = alternative_c(candidates, baseline, guided, context, affected=affected, current=current) if at_root else None
        proposals = {"A": baseline, "B": guided, "C": alternate}
        self.pending = dict(**self.identity, step=int(context.state.step), boundary="initialize" if self.initialize else "controller.step",
                            decision_index=index, candidate_pool_hash=pool_hash,
                            previous_assignment_geometry=assignment_geometry(self.old_allocation),
                            proposal_geometries={key: solution_geometry(value, current=current, affected=affected) for key, value in proposals.items()},
                            proposal_candidate_keys={key: [c.key for c in value.selected] if value is not None else None for key, value in proposals.items()},
                            proposal_candidate_ids_changed=baseline.selected_ids != guided.selected_ids,
                            baseline_objective=baseline.objective, actual_objective=actual.objective,
                            ranking_changed=any(r["algorithm"] == "B" and r["ranking_changed_at_this_prefix"] for r in scores),
                            runtime_guidance_active=self.scorer.active,
                            allocation_proposal_changed=digest(solution_geometry(baseline, current=current, affected=affected)) != digest(solution_geometry(guided, current=current, affected=affected)),
                            original_candidate_ids=[c.candidate_id for c in candidates],
                            prefix_hash=self.trace_chain, root_state_hash=self.boundaries[int(context.state.step)]["fingerprint"]["sha256"])
        if at_root:
            selected = proposals[self.intervention]
            if selected is None:
                raise CUnavailable("no distinct legal single-searcher proposal in actual pool")
            self.guided_root_count += int(self.intervention == "B" and self.scorer.active)
            return selected
        if self.root_step is not None and context.state.step > self.root_step and self.scorer.active:
            self.guided_after_root_count += 1
            raise RuntimeError("guided selection after single intervention")
        return actual

    def generated_pool(self, state, candidates, standby_candidates, *, scope, agent_ids):
        """Observe actual generation, including empty pools before early returns."""
        for agent_id in (tuple(agent_ids) or (None,)):
            pool = [c for c in candidates if c.agent_id == agent_id]
            self.generations.append(dict(**self.identity, step=int(state.step), scope=scope, agent_id=agent_id,
                                         generation_index=len(self.generations), candidate_count=len(pool),
                                         standby_count=None if standby_candidates is None else len(standby_candidates),
                                         candidate_ids=[c.candidate_id for c in pool],
                                         geometry_hash=digest([candidate_geometry(c) for c in pool]),
                                         empty_search_pool=not pool if agent_id is not None else None,
                                         scores_available_at_generation=False,
                                         score_reason="objective context is constructed later; see candidate_representation/candidate_scores when solver is reached"))

    def capture_proposal(self, proposal):
        if self.pending is not None:
            self.pending["chosen_proposal_signature"] = digest(assignment_geometry(proposal))
            self.pending["chosen_proposal_objective"] = float(proposal.objective_value)
            self.pending["chosen_proposal_candidate_ids"] = [(a.agent_id, a.candidate_id) for a in proposal.search_assignments]

    def after_decision(self, result, *, initialize):
        if self.root_step == int(self.state.step) and self.intervention is not None and self.pending is None:
            raise InvalidRoot("requested root did not generate a search proposal; no intervention was delivered")
        self.last_result = result
        self.accepted = initialize or bool(result.replanned)
        if self.pending is not None:
            self.pending.update(proposal_accepted=self.accepted,
                                decision_reason="INITIALIZE" if initialize else result.decision_reason,
                                historical_accepted_search_change_count=self.scorer.accepted_search_change_count,
                                installed_assignment_geometry=assignment_geometry(result.allocation),
                                installed_assignment_signature=digest(assignment_geometry(result.allocation)))
        if self.root_step == int(self.state.step) and self.intervention is not None:
            self.scorer.active = False
        elif self.root_step is not None and self.state.step > self.root_step and self.accepted and self.first_followup_replan is None:
            self.first_followup_replan = int(self.state.step)

    def after_install(self, state, observations, public_guidance, installed_guidance):
        self.current_observations = observations
        self.public_guidance, self.installed_guidance = public_guidance, installed_guidance
        public_geometry, effective_geometry = guidance_geometry(public_guidance), guidance_geometry(installed_guidance)
        if self.pending is not None:
            self.pending.update(public_guidance_geometry=public_geometry, effective_guidance_geometry=effective_geometry,
                                installed_guidance_signature=digest(effective_geometry))
            self.decisions.append(self.pending)
            self.pending = None
        if self.shadow is not None and not state.target_found:
            # Same physical state, post-install observations. This does not advance
            # the actual bridge or refresh the provider/map.
            self.shadow.observe_state(observations, state)
            for assignment in self.last_result.allocation.search_assignments:
                agent_id = assignment.agent_id
                actual = self.shadow._features[agent_id].copy()
                candidate = BSEROnlineAllocator._frozen_search_candidate(assignment)
                rebuilt = self.shadow.candidate_feature(candidate, state)
                public = public_guidance.assignment_for(agent_id)
                effective = installed_guidance.assignment_for(agent_id)
                agent = next(a for a in state.agents if a.agent_id == agent_id)
                comparison = compare_feature(actual, rebuilt, observation_step=self.env.get_task_state().step,
                                             public_step=state.step, observation_position=actual[:3], public_position=agent.position,
                                             semantic=assignment.waypoint, public_tracking=public.tracking_waypoint,
                                             effective_tracking=effective.tracking_waypoint,
                                             overlay_active=getattr(getattr(self.recovery, "agents", {}).get(agent_id), "plan", None) is not None)
                av, bv = self.shadow.estimate_candidate_value(actual), self.shadow.estimate_candidate_value(rebuilt)
                self.consistency.append(dict(**self.identity, step=state.step, agent_id=agent_id,
                                             timing="after_guidance_install_before_next_action", **comparison,
                                             actual_feature_id=exact_feature_id(actual), candidate_feature_id=exact_feature_id(rebuilt),
                                             actual_prediction=av, candidate_prediction=bv,
                                             prediction_difference=av-bv if comparison["comparable"] else None,
                                             path_hash=digest(assignment.path), map_revision=state.map_revision,
                                             effective_path_hash=digest(effective.planned_path),
                                             public_hold_state=public.hold_state, effective_hold_state=effective.hold_state,
                                             tracker_cursor=self.bridge.path_tracker.snapshot(agent_id).next_index,
                                             information_update_step=self.provider._last_full_refresh_step))
                self.feature_vectors[exact_feature_id(actual)] = actual.tolist()
                self.feature_vectors[exact_feature_id(rebuilt)] = rebuilt.tolist()
        signature = digest(effective_geometry)
        if self.root_step == int(state.step):
            self.root_installed_signature = signature
            if self.probe:
                raise BoundaryProbeComplete()
        elif self.root_step is not None and state.step > self.root_step and self.guidance_end is None and signature != self.root_installed_signature:
            self.guidance_end = int(state.step)

    def before_action(self, state, observations, installed_guidance):
        fingerprint = self.fingerprint(state, observations, self.context)
        self.current_step_trace = dict(step=int(state.step), state_hash=fingerprint["sha256"],
                                       physical_state_hash=fingerprint["components"]["environment"],
                                       guidance_hash=digest(guidance_geometry(installed_guidance)),
                                       rng_hash=component_fingerprint(dict(rng=rng_state()))["sha256"])
        if self.shadow_enabled and not state.target_found:
            self.shadow.observe_state(observations, state)
            for agent_id in state.searcher_ids:
                feature = self.shadow._features[agent_id].copy()
                feature_id = exact_feature_id(feature)
                self.feature_vectors[feature_id] = feature.tolist()
                self.predictions.append(dict(**self.identity, step=int(self.env.get_task_state().step), public_state_step=int(state.step),
                                             agent_id=int(agent_id), prediction=self.shadow.estimate_candidate_value(feature), feature_id=feature_id,
                                             information_update_step=self.provider._last_full_refresh_step, map_revision=int(state.map_revision),
                                             timing="PRE_FOUND_before_next_action"))

    def action(self, state, actions):
        self.current_step_trace["action_hash"] = digest(json_value(actions))
        self.trace_chain = digest([self.trace_chain, self.current_step_trace])
        self.current_step_trace["prefix_hash"] = self.trace_chain
        self.steps.append(self.current_step_trace)

    def transition(self, state, observations, dones):
        self.observed_until = int(state.step)
        if state.target_found and self.found_step is None:
            self.found_step = int(state.step)
        if self.root_step is not None and state.step > self.root_step:
            collisions = np.asarray(torch.as_tensor(self.env.unwrapped.collision_flags).cpu(), dtype=int)[:3]
            self.suffix_collisions += collisions
            self.streak = (self.streak + 1)*collisions
            self.max_streak = max(self.max_streak, int(self.streak.max()))
            positions = np.array([a.position for a in state.agents[:3]])
            self.distance += float(np.linalg.norm(positions-self.last_positions, axis=1).sum())
            self.last_positions = positions
            self.known_gain = float(np.mean(state.occupancy.known_mask))-self.root_known

    def finish(self, payload):
        if payload is not None:
            self.terminal_fingerprint = self.fingerprint(self.state, self.current_observations, self.context)
        self.weight_validation = dict(actor_before=self.before_actor_hash, actor_after=actor_hash(self.actor.policy_snapshot()),
                                      head_before=self.before_head_hash,
                                      head_after=weights_hash(self.shadow.head.state_dict()) if self.shadow else self.before_head_hash)
        if self.weight_validation["actor_before"] != self.weight_validation["actor_after"] or self.weight_validation["head_before"] != self.weight_validation["head_after"]:
            raise RuntimeError("audit changed model weights")
        if self.scorer.head is not None and weights_hash(self.scorer.head.state_dict()) != self.before_head_hash:
            raise RuntimeError("runtime Head changed")
        self.payload = payload

    def export(self):
        return json_value(dict(identity=self.identity, predictions=self.predictions, feature_consistency=self.consistency,
                               candidate_representation=self.representation, candidate_scores=self.scores, decisions=self.decisions,
                               candidate_generation=self.generations,
                               steps=self.steps, boundaries=self.boundaries, feature_vectors=self.feature_vectors,
                               terminal_fingerprint=getattr(self, "terminal_fingerprint", None),
                               found_step=self.found_step, observed_until=self.observed_until,
                               root_fingerprint=self.root_fingerprint, prefix_hash=self.prefix_chain,
                               suffix_collisions=int(self.suffix_collisions.sum()), max_collision_streak=self.max_streak,
                               travel_distance=self.distance, known_ratio_gain=self.known_gain,
                               first_followup_replan=self.first_followup_replan,
                               guidance_duration=(self.guidance_end or self.observed_until) - self.root_step if self.root_step is not None else None,
                               guided_root_count=self.guided_root_count, guided_after_root_count=self.guided_after_root_count,
                               weights=getattr(self, "weight_validation", None), payload=getattr(self, "payload", None)))
