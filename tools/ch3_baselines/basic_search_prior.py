"""Search-only allocation injected before the first guidance installation.

Only the construction seam and residual source differ from MissionRuntime.
Physics, public event handling, reward wrappers and navigation remain shared.
"""
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import replace

import numpy as np
import torch

from chapter3_bser.baselines.search_only_allocator import solve_search_only_greedy
from chapter3_bser.candidate_generator import generate_search_candidates
from chapter3_bser.objective import build_objective_context, evaluate_objective, cell_detection_probability
from chapter3_bser.online.allocator import BSEROnlineAllocator
from chapter3_bser.online.controller import OnlineBSERController
from chapter3_bser.online.types import OnlineAllocation
from chapter3_bser.online.config import load_phase1b2_config, execution_runtime_config
from chapter3_bser.controllers.state_provider import OnlinePlanningStateProvider
from chapter3_bser.integration.guided_env import GuidedEnv
from chapter3_bser.integration.rmaddpg_bridge import RMADDPGGuidanceBridge
from chapter3_bser.experiments.hgr.runtime import MissionRuntime, seed_innovations
from chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.train_phase1c_v2 import _public_context
from chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.training_env import Phase1CV2TrainingEnv
from chapter3_bser.experiments.phase1c_prrac.training_env import PRRACTrainingEnv
from chapter3_bser.experiments.phase1c_prrac.runtime_factory import runtime_contract
from core.mapping.travel_cost_service import TravelCostService
from core.config.ch3_config import build_ch3_config
from core.env.mission_env import MissionCoreEnv, environment_kwargs_from_config
from core.env.task_protocol import PROTOCOL_FIELDS, validate_task_config

METHOD = "ch3_basic_search_prior_v1"
SPEC = dict(method=METHOD, planner_mode="search_only", controller_mode="prior_only",
            standby_mode="initial_position_hold", training_update=False,
            residual_source="zeros_4x3", policy_weights=None,
            standby_environment_override={"pse_use_standby": False},
            allowed_interventions=["search_scoring", "pre_handoff_standby", "residual_source", "learning_process"])


def baseline_environment_kwargs(config):
    """The reference factory's exact parsing with one declared standby switch.

Disable the background legacy PSE standby optimizer too, at construction time,
so even transient reset planning cannot optimize a hidden waiting position.
This switch is used only by standby planning, not by belief or sensing.
"""
    validate_task_config(config)
    environment = build_ch3_config(config.get("base_candidate", "ch3_v3_full_reference"), config["profile"])
    environment.update({k: config[k] for k in (*PROTOCOL_FIELDS, "collision_terminal_reward") if k in config})
    kwargs = environment_kwargs_from_config(environment, device="cpu", max_steps=config["max_steps"], return_numpy=False)
    kwargs.update(SPEC["standby_environment_override"])
    return kwargs


class SearchPriorAllocator(BSEROnlineAllocator):
    def __init__(self, anchor, phase1a1_config=None):
        super().__init__(phase1a1_config)
        self._anchor = tuple(float(v) for v in anchor)
        if len(self._anchor) != 3 or not np.isfinite(self._anchor).all():
            raise ValueError("anchor must be a finite initial position")
        self.counts = Counter()
        self.reasons = Counter()

    @property
    def anchor(self):
        return self._anchor

    def standby(self, state):
        self.counts["anchor_route_queries"] += 1
        assignment = self.execution._assignment(state, self.anchor, source="INITIAL_POSITION_HOLD")
        if not assignment.reachable:
            self.counts["anchor_unreachable"] += 1
            self.reasons[assignment.failure_reason or "ANCHOR_UNREACHABLE"] += 1
        return assignment

    def _candidates(self, state, ids):
        generation = self.config["candidate_generation"]
        candidates, unreachable, reasons = generate_search_candidates(
            replace(state, searcher_ids=tuple(sorted(ids))), TravelCostService(state),
            k_search=int(generation["k_search_exact"]),
            minimum_separation=float(generation["minimum_separation"]),
            maximum_travel_time=float(generation.get("maximum_physical_travel_time", generation.get("maximum_travel_time"))))
        self.counts["unreachable_search_queries"] += unreachable
        self.counts["candidate_shortage_events"] += sum(":ONLY_" in r for r in reasons)
        self.reasons.update(reasons)
        if unreachable:
            self.reasons["SEARCH_PATH_UNREACHABLE_UNSPECIFIED_BY_GENERATOR"] += unreachable
        return candidates

    def _solve_candidates(self, candidates, standby_candidates, context):
        return solve_search_only_greedy(candidates, context)

    def score(self, state, assignments):
        candidates = tuple(self._frozen_search_candidate(a) for a in assignments)
        context = build_objective_context(state, candidates, (), self.config)
        return evaluate_objective(candidates, None, context, search_only=True)

    def allocate(self, state, *, trigger_reason="online"):
        self.counts["full_search_allocations"] += 1
        candidates = self._candidates(state, state.searcher_ids)
        context = build_objective_context(state, candidates, (), self.config)
        solved = self._solve_candidates(candidates, (), context)
        self.counts["selected_search_assignments"] += len(solved.selected)
        self.counts["unassigned_searchers"] += len(set(state.searcher_ids) - {c.agent_id for c in solved.selected})
        return OnlineAllocation(tuple(self._search_assignment(c) for c in solved.selected),
                                self.standby(state), solved.objective, solved.objective,
                                float("inf"), trigger_reason,
                                solved.status if solved.selected else "NO_POSITIVE_SEARCH_SELECTION")

    def allocate_partial(self, state, current, *, affected_searcher_ids=(),
                         executor_affected=False, trigger_reason):
        self.counts["partial_allocation_attempts"] += 1
        affected = set(affected_searcher_ids)
        if not affected.issubset(state.searcher_ids):
            return current, False, "ATOMIC_REJECT_UNKNOWN_SEARCHER"
        frozen = tuple(a for a in current.search_assignments if a.agent_id not in affected)
        fixed = tuple(self._frozen_search_candidate(a) for a in frozen)
        candidates = self._candidates(state, affected) if affected else ()
        if any(not any(c.agent_id == i for c in candidates) for i in affected):
            self.reasons["ATOMIC_REJECT_MISSING_SEARCH_ROUTE"] += 1
            return current, False, "ATOMIC_REJECT_MISSING_SEARCH_ROUTE"
        context = build_objective_context(state, candidates + fixed, (), self.config)
        # Conditional marginal discovery gain given the retained routes. This
        # preserves the original greedy partition constraint and tie break.
        conditional = replace(context, belief=context.belief * (1 - cell_detection_probability(fixed, context)))
        solved = self._solve_candidates(candidates, (), conditional)
        if any(not any(c.agent_id == i for c in solved.selected) for i in affected):
            self.reasons["ATOMIC_REJECT_MISSING_GREEDY_SELECTION"] += 1
            return current, False, "ATOMIC_REJECT_MISSING_GREEDY_SELECTION"
        assignments = tuple(sorted(frozen + tuple(self._search_assignment(c) for c in solved.selected), key=lambda a: a.agent_id))
        value = evaluate_objective(fixed + solved.selected, None, context, search_only=True)
        executor = self.standby(state) if executor_affected else current.executor_assignment
        self.counts["partial_search_proposals"] += bool(affected)
        return replace(current, search_assignments=assignments, executor_assignment=executor,
                       objective_value=value, detection_probability=value, response_time=float("inf"),
                       trigger_reason=trigger_reason, status=solved.status), True, "ATOMIC_PARTIAL_SEARCH_ONLY_PROPOSAL"

    def reassign_invalid_executor(self, state, current, *, trigger_reason="EXECUTOR_INVALID"):
        return replace(current, executor_assignment=self.standby(state), trigger_reason=trigger_reason)

    def reassign_after_target_found(self, state, *, trigger_reason="TARGET_FOUND"):
        raise RuntimeError("Found is not a legal target delivery; use the corrected public-handoff controller")


class SearchPriorController(OnlineBSERController):
    def step(self, state, mission_context=None):
        # Stabilization can retain old routes. Revalue the actual retained set
        # before the original hysteresis compares it with a fresh proposal.
        old = self.current_allocation
        value = self.allocator.score(state, old.search_assignments)
        self.current_allocation = replace(old, objective_value=value, detection_probability=value)
        result = super().step(state, mission_context)
        value = self.allocator.score(state, result.allocation.search_assignments)
        allocation = replace(result.allocation, objective_value=value, detection_probability=value)
        self.current_allocation = allocation
        return replace(result, allocation=allocation)


class BasicSearchPriorRuntime(MissionRuntime):
    def __init__(self, config, scenario, *, seed, episode_id=0):
        self.config, self.scenario = copy.deepcopy(config), copy.deepcopy(scenario)
        contract = runtime_contract(config)
        if contract.runtime_integration_mode != "legacy":
            raise ValueError("baseline v1 requires the reference legacy v2.1 execution runtime")
        seed_innovations(seed)
        base = MissionCoreEnv(**baseline_environment_kwargs(config))
        guided = GuidedEnv(base, enabled=True)
        self.env = PRRACTrainingEnv(Phase1CV2TrainingEnv(guided, reward_config=config["reward"]),
                                  reward_objective_config=config, gamma=config["rl"]["gamma"])
        try:
            self.env.reset(scenario=scenario, episode_id=episode_id, episode_index=episode_id)
            seed_innovations(seed)
            phase = load_phase1b2_config()
            execution = execution_runtime_config(config)
            phase["execution_runtime"] = copy.deepcopy(execution)
            self.provider = OnlinePlanningStateProvider(self.env, refresh_interval=int(phase["online"]["state_refresh_interval"]),
                **{k: execution[k] for k in ("refresh_on_executor_handoff", "refresh_on_public_target_shift", "public_target_update_distance", "public_target_update_min_steps")})
            self.state = self.provider.initialize()
            if tuple(self.state.searcher_ids) != (0, 1, 2) or self.state.executor_id != 3:
                raise ValueError("baseline requires three Searchers followed by one Executor")
            allocator = SearchPriorAllocator(self.env.get_agent_state().positions[3])
            self.controller = SearchPriorController(phase, allocator=allocator)
            self.controller.prrac_runtime_contract = contract
            context = _public_context(self.env, self.state)
            initialized = self.controller.initialize(self.state, context)
            self.bridge = RMADDPGGuidanceBridge()
            self.guidance = self.bridge.compile_guidance(initialized.allocation, self.state, context, decision_reason="INITIALIZE")
            self.env.install_guidance(self.guidance)
            self.observations = self.env.refresh_observation_after_guidance()
            self.found_step = self.handoff_event_step = self.handoff_decision_step = None
            self.diagnostics = dict(actor_forward_calls=0, action_sampling_calls=0, optimizer_update_count=0,
                residual_action_max_abs=0.0, physical_residual_acceleration_max_abs=0.0,
                physical_prior_acceleration_max_abs=0.0, residual_steps_checked=0,
                standby_anchor=list(allocator.anchor), pre_handoff_max_anchor_deviation=0.0)
            self._record_events()
            self._check_anchor()
        except BaseException:
            self.env.close()
            raise

    def _check_anchor(self):
        if not self.env.get_task_state().executor_knows_target:
            if self.controller.current_allocation.executor_assignment.target_region != self.controller.allocator.anchor:
                raise RuntimeError("pre-handoff target left its initial anchor")
            deviation = float(np.linalg.norm(np.asarray(self.env.get_agent_state().positions[3]) - self.controller.allocator.anchor))
            self.diagnostics["pre_handoff_max_anchor_deviation"] = max(self.diagnostics["pre_handoff_max_anchor_deviation"], deviation)

    def advance(self):
        if self.terminal:
            raise RuntimeError("cannot step a terminal mission")
        before = self.step
        self._check_anchor()
        actions = torch.zeros((4, 3), dtype=torch.float32)
        if actions.shape != (4, 3) or not torch.isfinite(actions).all() or torch.count_nonzero(actions):
            raise RuntimeError("nonzero or invalid residual command")
        if not self.env.unwrapped.use_residual_prior:
            raise RuntimeError("prior control is disabled")
        observations, rewards, dones = self.env.step(actions)
        physical = self.env.unwrapped._last_residual_acc
        if physical.shape != (4, 3) or not torch.isfinite(physical).all() or torch.count_nonzero(physical):
            raise RuntimeError("nonzero or invalid physical residual acceleration")
        self.diagnostics["residual_steps_checked"] += 1
        self.diagnostics["physical_prior_acceleration_max_abs"] = max(
            self.diagnostics["physical_prior_acceleration_max_abs"], float(self.env.unwrapped._last_prior_acc.abs().max()))
        self._check_anchor()
        if self.terminal:
            self.observations = observations
        else:
            self.state = self.provider.snapshot(force=False)
            context = _public_context(self.env, self.state)
            result = self.controller.step(self.state, context)
            self.env.observe_controller_result(result, controller=self.controller, state_provider=self.provider)
            self.guidance = self.bridge.compile_guidance(result.allocation, self.state, context, decision_reason=result.decision_reason)
            self.env.install_guidance(self.guidance)
            self.observations = self.env.refresh_observation_after_guidance()
        self._record_events()
        if self.step != before + 1 or self.terminal != all(bool(d) for d in dones):
            raise RuntimeError("mission clock/done contract mismatch")
        return copy.deepcopy(self.env.reward_accounting.last)

    def controller_diagnostics(self):
        return dict(self.diagnostics, allocation_counts=dict(self.controller.allocator.counts),
                    allocation_reasons=dict(self.controller.allocator.reasons),
                    accepted_replans=self.controller.replan_count,
                    replan_steps=list(self.controller.replan_steps),
                    initial_endpoint_failures=list(self.env.last_initial_endpoint_failures))
