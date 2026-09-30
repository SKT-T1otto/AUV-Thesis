"""Bounded, per-searcher recovery using the existing B0/B1 candidate solvers."""
from collections import deque
import numpy as np

from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.hysteresis.policy import ReplanningPolicy
from chapter3_bser.online.types import BSERActionAssignment
from chapter3_bser.experiments.safe_search_v1.runtime import SearchController
from chapter3_bser.experiments.safe_search_v1.planning_refresh import start_status
from chapter3_bser.experiments.safe_search_v1.public_geometry import PublicGeometry


class RecoveryController(SearchController):
    def __init__(self, inner, owner):
        self.inner, self.owner = inner, owner
        # Native event cooldowns remain ordinary; our per-agent recovery clock
        # is separate and is never reset by a forced planning refresh.
        inner.policy = ReplanningPolicy(inner.config)
        self.next_retry = {}
        self.positions = {}
        self.last_recovery = {}

    def _pending(self, state, context):
        pending = self._geometry_pending(state)
        finished = dict(zip(state.searcher_ids, context.searcher_finished_flags))
        assigned = {a.agent_id: a for a in self.inner.current_allocation.search_assignments}
        settings = self.owner.found_settings
        for agent in state.agents:
            i = agent.agent_id
            if i not in state.searcher_ids or finished.get(i, False):
                continue
            if i not in assigned:
                pending.add(i)
            history = self.positions.setdefault(i, deque(maxlen=settings.stall_window_steps + 1))
            if not history or history[-1][0] != state.step:
                history.append((state.step, np.asarray(agent.position)))
            if len(history) == settings.stall_window_steps + 1 and state.step-history[0][0] == settings.stall_window_steps:
                excursion = max(np.linalg.norm(p-history[0][1]) for _, p in history)
                if excursion < settings.stall_distance:
                    pending.add(i)
            if hasattr(self.owner, "bridge"):
                route = self.owner.bridge.path_tracker.snapshot(i)
                if (i in assigned and route.completed and route.final_waypoint == assigned[i].waypoint
                        and np.linalg.norm(np.asarray(agent.position)-assigned[i].waypoint) <= .75):
                    pending.add(i)
        return pending - {i for i, flag in finished.items() if flag}

    def step(self, state, mission_context=None):
        owner, inner = self.owner, self.inner
        if state.target_found or mission_context.target_found:
            owner.safe_pending.clear()
            self.positions.clear()
            self.last_recovery.clear()
            return inner.step(state, mission_context)
        pending = self._pending(state, mission_context)
        detection = inner.detector.detect(inner.cache.current, state, inner.current_context,
            mission_context, assignment=inner.current_allocation)
        pending.update(detection.stale_searcher_ids)
        primary = inner.policy.primary_event(detection.events)
        opportunity = primary is not None and not inner.policy.event_cooldown_remaining(primary, state.step)
        due = sorted(i for i in pending if state.step >= self.next_retry.get(i, 0))
        # Refresh before BOTH ordinary planning and recovery, so queries and
        # later guidance share authoritative current-position connectors.
        if due or opportunity and any(start_status(state, i) == "stale_start_endpoint" for i in state.searcher_ids):
            state = owner.planning_views.refresh()
        owner.state = state
        old = inner.current_allocation
        result = inner.step(state, mission_context)
        accepted = []
        for agent_id in due:
            previous = next((a for a in old.search_assignments if a.agent_id == agent_id), None)
            installed = next((a for a in inner.current_allocation.search_assignments if a.agent_id == agent_id), None)
            if result.replanned and installed is not None and installed != previous and installed.failure_reason is None:
                geometry = PublicGeometry(owner.planning_views.live(), owner.planning_views.clearance)
                agent = next(a for a in state.agents if a.agent_id == agent_id)
                if geometry.path_free((agent.position, *installed.path)):
                    owner.safe_pending.discard(agent_id)
                    self.positions.pop(agent_id, None)
                    continue
            self.next_retry[agent_id] = state.step + owner.found_settings.recovery_retry_steps
            checked = owner.planning_views.search(state)
            status = start_status(checked, agent_id)
            if status != "ready":
                owner.safe_pending.add(agent_id)
                owner.found_counts["recovery_preflight:" + status] += 1
                self.last_recovery[agent_id] = dict(step=state.step, accepted=False, reason=status,
                                                   retry_at=self.next_retry[agent_id])
                continue
            current = inner.current_allocation
            proposed, ok, reason = inner.allocator.allocate_partial(state, current,
                affected_searcher_ids=(agent_id,), executor_affected=False,
                trigger_reason="FOUND_SEARCH_RECOVERY")
            # Independently scoped calls preserve the native scoring and keep
            # one agent's invalid start from rejecting healthy agents' routes.
            before = next((a for a in current.search_assignments if a.agent_id == agent_id), None)
            after = next((a for a in proposed.search_assignments if a.agent_id == agent_id), None)
            changed = bool(ok and after is not None and after.failure_reason is None and after != before)
            if changed:
                if proposed.executor_assignment != current.executor_assignment:
                    raise RuntimeError("search recovery changed the executor assignment")
                if any(a != next((b for b in proposed.search_assignments if b.agent_id == a.agent_id), None)
                       for a in current.search_assignments if a.agent_id != agent_id):
                    raise RuntimeError("single-agent recovery changed a retained search assignment")
                inner.current_allocation = proposed
                accepted.append(agent_id)
                owner.safe_pending.discard(agent_id)
                self.positions.pop(agent_id, None)
                owner.found_counts["recovery_assignments"] += 1
            else:
                owner.safe_pending.add(agent_id)
                owner.found_counts["recovery_failure:" + reason] += 1
            self.last_recovery[agent_id] = dict(step=state.step, accepted=changed,
                reason=reason if changed or not ok else "UNCHANGED_ASSIGNMENT",
                retry_at=self.next_retry[agent_id])
        if accepted:
            proposed = inner.current_allocation
            updates = inner.waypoints.updates(old, proposed, reason="FOUND_SEARCH_RECOVERY", step=state.step)
            inner.policy.mark_replan(state.step, BSEREvent.WAYPOINT_STALE)
            if not result.replanned:
                inner.replan_count += 1
                inner.replan_steps.append(int(state.step))
            diagnostics = inner._diagnostics(state=state, events=result.events, optimizer_invoked=True,
                scope="found_search_recovery", old=old, proposed=proposed, accepted=True,
                accept_reason="FOUND_SEARCH_RECOVERY", affected=accepted, route_impacted=True, updates=updates)
            result = BSERActionAssignment(state.step, True, result.events, proposed, updates,
                "FOUND_SEARCH_RECOVERY", result.event_detection, diagnostics)
        return result
