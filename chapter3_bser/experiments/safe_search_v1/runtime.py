"""Independent B0/B1 assembly with public-state repair and zero residuals."""
from collections import Counter
from dataclasses import replace

import numpy as np

from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.experiments.hgr.runtime import MissionRuntime
from chapter3_bser.online.types import BSERActionAssignment
from tools.ch3_baselines.basic_search_prior import BasicSearchPriorRuntime
from tools.ch3_baselines.bser_prior import BSERPriorRuntime, ZeroResidualSource
from .allocators import SafePriorAllocator, SafeJointAllocator
from .failure_policy import SearchFailurePolicy
from .planning_refresh import PlanningViews, start_status, topology_key
from .public_geometry import PublicGeometry


VARIANTS = {
    "V0": dict(planning_state_consistency=False, failure_policy=False, path_safety=False),
    "V1": dict(planning_state_consistency=True, failure_policy=False, path_safety=False),
    "V2": dict(planning_state_consistency=True, failure_policy=True, path_safety=False),
    "V3": dict(planning_state_consistency=False, failure_policy=False, path_safety=True),
    "V4": dict(planning_state_consistency=True, failure_policy=True, path_safety=True),
    "V5": dict(planning_state_consistency=True, failure_policy=False, path_safety=True),
}


class SearchController:
    """Compose the original controller; no replacement event/physics engine."""
    def __init__(self, inner, owner):
        self.inner = inner
        self.owner = owner
        if owner.safe_options["failure_policy"]:
            self.inner.policy = SearchFailurePolicy(inner.config)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def initialize(self, state, context=None):
        return self.inner.initialize(state, context)

    def _geometry_pending(self, state):
        if not self.owner.safe_options["path_safety"]:
            return set()
        geometry = PublicGeometry(self.owner.planning_views.live(), self.owner.planning_views.clearance)
        pending = set(self.owner.safe_pending)
        if hasattr(self.owner, "bridge"):
            for item in self.inner.current_allocation.search_assignments:
                remaining = self.owner.bridge.path_tracker.snapshot(item.agent_id).remaining_path_points
                if not geometry.path_free((state.agents[item.agent_id].position, *remaining)):
                    pending.add(item.agent_id)
        return pending

    def step(self, state, mission_context=None):
        owner, inner = self.owner, self.inner
        if state.target_found:
            owner.safe_pending.clear()
            if isinstance(inner.policy, SearchFailurePolicy):
                inner.policy.search_enabled = False
            return inner.step(state, mission_context)

        # Detection is pure. This preflight can issue real route queries, which
        # are visible in diagnostic counts, but consumes no simulator/RNG step.
        detection = inner.detector.detect(inner.cache.current, state, inner.current_context,
                                          mission_context, assignment=inner.current_allocation)
        stale = set(detection.stale_searcher_ids)
        geometry_pending = self._geometry_pending(state)
        pending = set(owner.safe_pending) | geometry_pending
        if owner.safe_options["failure_policy"]:
            finished = set(i for i, flag in zip(state.searcher_ids,
                getattr(mission_context, "searcher_finished_flags", ())) if flag)
            assigned = {item.agent_id for item in inner.current_allocation.search_assignments}
            pending.update(set(state.searcher_ids) - assigned - finished)
        primary = inner.policy.primary_event(detection.events)
        policy = inner.policy if isinstance(inner.policy, SearchFailurePolicy) else None
        ready = policy.ready(state, topology_key(state), full_refresh=owner.provider.last_snapshot_was_full_refresh) if policy else False
        if policy:
            # Update before testing cooldowns: last tick's recovery readiness
            # must not manufacture a fresh snapshot that bypasses retry delay.
            policy.recovery_ready = ready and bool(pending or stale)
            policy.retry_remaining = 0 if ready or policy.last_failed_step is None else max(
                0, policy.retry_steps - (state.step - policy.last_failed_step))
        opportunity = primary is not None and not inner.policy.event_cooldown_remaining(primary, state.step)
        if owner.safe_options["planning_state_consistency"]:
            need_snapshot = bool((geometry_pending and (not policy or ready)) or (opportunity or (pending and ready)) and
                                 any(start_status(state, i) == "stale_start_endpoint" for i in state.searcher_ids))
            if need_snapshot:
                state = owner.planning_views.refresh()
        owner.state = state
        if policy:
            ready = policy.ready(state, topology_key(state), full_refresh=owner.provider.last_snapshot_was_full_refresh)
            policy.recovery_ready = ready and bool(pending or stale)
            policy.retry_remaining = 0 if ready or policy.last_failed_step is None else max(
                0, policy.retry_steps - (state.step - policy.last_failed_step))

        old = inner.current_allocation
        result = inner.step(state, mission_context)
        if policy:
            affected = (pending | stale) & set(state.searcher_ids)
            invoked = bool(result.diagnostics and result.diagnostics.optimizer_invoked)
            if result.replanned:
                policy.succeeded()
                installed = {item.agent_id for item in result.allocation.search_assignments}
                owner.safe_pending.difference_update(affected & installed)
                owner.safe_pending.update(affected - installed)
            elif affected and ready and not invoked:
                # Retry an unresolved assignment on a fresh usable snapshot,
                # independently of an ordinary gain-only planning event.
                proposed, ok, reason = inner.allocator.allocate_partial(
                    state, inner.current_allocation, affected_searcher_ids=tuple(sorted(affected)),
                    executor_affected=False, trigger_reason="SAFE_SEARCH_RECOVERY")
                if ok and proposed != old:
                    updates = inner.waypoints.updates(old, proposed, reason="SAFE_SEARCH_RECOVERY", step=state.step)
                    inner.current_allocation = proposed
                    inner.policy.mark_replan(state.step, BSEREvent.WAYPOINT_STALE)
                    inner.replan_count += 1
                    inner.replan_steps.append(int(state.step))
                    diag = inner._diagnostics(state=state, events=result.events, optimizer_invoked=True,
                        scope="safe_search_recovery", old=old, proposed=proposed, accepted=True,
                        accept_reason="SAFE_SEARCH_RECOVERY", affected=affected, route_impacted=bool(geometry_pending), updates=updates)
                    result = BSERActionAssignment(state.step, True, result.events, proposed, updates,
                        "SAFE_SEARCH_RECOVERY", result.event_detection, diag)
                    policy.succeeded()
                    owner.safe_pending.difference_update(affected)
                    owner.safe_counts["recovery_replans"] += 1
                else:
                    owner.safe_pending.update(affected)
                    policy.failed(state.step, topology_key(state))
                    owner.safe_counts["recovery_failure:" + reason] += 1
            elif affected and invoked:
                owner.safe_pending.update(affected)
                policy.failed(state.step, topology_key(state))
        return result


class SafetyMixin:
    def _setup_safe(self, scenario, variant):
        if variant not in VARIANTS or variant == "V0":
            raise ValueError("enabled runtime requires V1..V5")
        self.safe_variant = variant
        self.safe_options = dict(VARIANTS[variant])
        self.safe_counts = Counter()
        self.safe_pending = set()
        clearance = float(scenario.get("planner_obstacle_clearance", 0.4))
        if not np.isfinite(clearance) or clearance < 0:
            raise ValueError("invalid public planner clearance")
        self.planning_views = PlanningViews(self, clearance)

    def _build_prior_controller(self, phase, allocator):
        selected = SafePriorAllocator(allocator.anchor, self, allocator.config)
        return SearchController(super()._build_prior_controller(phase, selected), self)

    def _build_online_controller(self, phase_config, config):
        inner = super()._build_online_controller(phase_config, config)
        inner.allocator = SafeJointAllocator(self, inner.allocator.config)
        return SearchController(inner, self)

    def _build_guidance_bridge(self):
        if self.safe_options["path_safety"]:
            from .safe_guidance import SafetyBridge
            return SafetyBridge(self)
        return super()._build_guidance_bridge()

    def controller_diagnostics(self):
        result = super().controller_diagnostics()
        result["safe_search"] = dict(variant=self.safe_variant, options=self.safe_options,
            counts=dict(self.safe_counts), unresolved_searcher_ids=sorted(self.safe_pending),
            safety_status=getattr(self.bridge, "last_safety_status", {}),
            information_boundary="public occupancy only; no truth obstacles or hidden target",
            zero_residual_required=True)
        return result


class SafePriorRuntime(SafetyMixin, BasicSearchPriorRuntime):
    def __init__(self, config, scenario, *, variant, seed, episode_id=0):
        self._setup_safe(scenario, variant)
        super().__init__(config, scenario, seed=seed, episode_id=episode_id)


class SafeJointRuntime(SafetyMixin, BSERPriorRuntime):
    def __init__(self, config, scenario, *, variant, seed, episode_id=0):
        self._setup_safe(scenario, variant)
        # This is an explicitly different experiment, not frozen B1. Reuse its
        # original advance/zero checks while assembling our own search allocator.
        MissionRuntime.__init__(self, config, scenario, seed=seed, episode_id=episode_id)
        self.zero_source = ZeroResidualSource()
        self.diagnostics = dict(actor_forward_calls=0, action_sampling_calls=0,
            optimizer_update_count=0, residual_steps_checked=0, residual_action_max_abs=0.0,
            physical_residual_acceleration_max_abs=0.0, physical_prior_acceleration_max_abs=0.0)


def make_runtime(config, scenario, *, baseline="B0_search_prior", variant="V0", seed=12729, episode_id=0):
    if baseline not in {"B0_search_prior", "B1_bser_prior"} or variant not in VARIANTS:
        raise ValueError("safe search supports only explicit B0/B1 and V0..V5")
    if variant == "V0":
        cls = BasicSearchPriorRuntime if baseline == "B0_search_prior" else BSERPriorRuntime
        return cls(config, scenario, seed=seed, episode_id=episode_id)
    cls = SafePriorRuntime if baseline == "B0_search_prior" else SafeJointRuntime
    return cls(config, scenario, variant=variant, seed=seed, episode_id=episode_id)
