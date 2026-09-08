"""Explicit diagnostic command intervention; no alternative physics or tracker.

Each instance is used in one deterministic evaluator replay from step zero.
The ordinary trace sink and runtime fingerprint are read-only observers.
"""

import numpy as np

from .searcher_residual_trace import SearcherResidualTrace, array, vector_fields
from .search_value_audit.provenance import digest
from .search_value_audit.state_fingerprint import component_fingerprint, runtime_fingerprint


def scale_commands(actions, agents, alpha):
    if not 0 <= alpha <= 1 or not agents or any(a not in (0, 1, 2) for a in agents):
        raise ValueError("invalid diagnostic alpha/branch agents")
    if alpha == 1:
        return actions
    result = actions.clone()
    result[list(agents)] *= alpha
    return result


class ResidualBranchDiagnostic(SearcherResidualTrace):
    def __init__(self, scenario, selection, alpha, horizon=10):
        super().__init__(scenario, "full_prrac", selection["transition_type"])
        self.selection, self.alpha, self.horizon = selection, float(alpha), horizon
        self.anchor = int(selection["anchor_step"])
        self.finished = False
        self.anchor_fingerprint = None
        self.boundaries, self.rows = [], []

    def boundary(self, state, env, bridge, guidance, recovery):
        runtime = env.unwrapped
        snap = None if recovery is None else recovery.snapshot()
        return dict(step=int(state.step), positions=array(runtime._agent_pos).tolist(),
                    targets=array(runtime._nav_targets).tolist(),
                    cursors=[bridge.path_tracker.snapshot(a).next_index for a in range(3)],
                    route_hashes=[digest(dict(endpoint=list(guidance.assignment_for(a).final_waypoint),
                                             path=getattr(guidance.assignment_for(a), "planned_path", None))) for a in range(3)],
                    c2_active=[] if snap is None else list(snap.active_agent_ids),
                    c2_modes={} if snap is None else dict(snap.mode_by_agent))

    def before_action(self, *, state, env, bridge, recovery, guidance, observations, raw_actions,
                      actions, provider, controller, scorer, context, action_adapter):
        step = int(state.step)
        self.current = self.boundary(state, env, bridge, guidance, recovery)
        self.raw, self.full_applied = array(raw_actions), array(actions)
        if step == self.anchor:
            # Existing strict inventory includes physics, wrappers, provider caches,
            # allocation, tracker, C2, public observations, and all inventoried RNGs.
            inventory = runtime_fingerprint(env, provider, controller, bridge, recovery, scorer,
                context=context, observations=observations, action_adapter=action_adapter)
            runtime = env.unwrapped
            # Pure original helper; no new prior/action formula and no cache write.
            prior = runtime._compute_waypoint_prior_acc() if hasattr(runtime, "_compute_waypoint_prior_acc") else None
            explicit = component_fingerprint(dict(scenario_id=self.base["scenario_id"], step=step,
                boundary=self.current, mission_stage=state.mission_phase if hasattr(state, "mission_phase") else state.target_found,
                public_observations=observations, waypoint_prior_preview=prior,
                cached_prior=getattr(runtime, "_last_prior_acc", None), raw_residual=raw_actions,
                full_applied_residual=actions))
            self.anchor_fingerprint = dict(sha256=digest([inventory["sha256"], explicit["sha256"]]),
                                           runtime=inventory, explicit=explicit)
        applied = scale_commands(actions, self.selection["branch_agents"], self.alpha) if self.anchor <= step < self.anchor+self.horizon else actions
        self.applied = array(applied)
        return applied

    def after_transition(self, *, state, env, bridge, guidance, recovery):
        step = self.current["step"]
        if step < self.anchor:
            return
        if not self.boundaries:
            self.boundaries.append(self.current)
        endpoint = self.boundary(state, env, bridge, guidance, recovery)
        self.boundaries.append(endpoint)
        diagnostics = {r["agent_id"]: r for r in self.action_rows if r["step"] == step}
        mission = self.mission_rows[-1]
        runtime = env.unwrapped
        for agent in range(3):
            row = dict(diagnostics.get(agent, {}))
            row.update(self.base, **self.selection, alpha=self.alpha, step=step,
                relative_step=step-self.anchor, agent_id=agent, sample_kind="transition",
                waypoint_cursor=self.current["cursors"][agent], route_hash=self.current["route_hashes"][agent],
                c2_active=agent in self.current["c2_active"],
                c2_state_or_tier=self.current["c2_modes"].get(agent),
                found_event=mission["found_event"], collision_event=bool(array(runtime.collision_flags)[agent]),
                navigation_target_change_event=bool(np.linalg.norm(np.array(endpoint["targets"][agent])-self.current["targets"][agent]) > 1e-8),
                waypoint_switch_event=row.get("waypoint_switch_event"),
                path_cursor_advance_event=endpoint["cursors"][agent] != self.current["cursors"][agent],
                raw_residual_norm=row.get("raw_residual_norm", float(np.linalg.norm(self.raw[agent].astype(np.float64)))),
                applied_residual_norm=row.get("applied_residual_norm", float(np.linalg.norm(self.applied[agent].astype(np.float64)))),
                residual_prior_cosine=row.get("residual_prior_cosine"), negative_alignment=row.get("negative_alignment"))
            for prefix, values, axes in (("position", self.current["positions"], "xyz"),
                    ("navigation_target", self.current["targets"], "xyz"), ("raw_residual", self.raw, "012"),
                    ("applied_residual", self.applied, "012"), ("full_applied_residual", self.full_applied, "012"),
                    ("prior_action", array(runtime._last_prior_acc), "012"), ("final_action", array(runtime._agent_acc), "012")):
                row.update(vector_fields(prefix, values[agent], axes))
            self.rows.append(row)
        self.finished = int(state.step) >= self.anchor+self.horizon
        if self.finished:
            # Endpoint is a state-only sample: exactly ten actions, never an 11th.
            for agent in range(3):
                self.rows.append(dict(self.base, **self.selection, alpha=self.alpha, step=int(state.step),
                    relative_step=self.horizon, agent_id=agent, sample_kind="endpoint",
                    waypoint_cursor=endpoint["cursors"][agent], route_hash=endpoint["route_hashes"][agent],
                    c2_active=agent in endpoint["c2_active"], c2_state_or_tier=endpoint["c2_modes"].get(agent),
                    **vector_fields("position", endpoint["positions"][agent], "xyz"),
                    **vector_fields("navigation_target", endpoint["targets"][agent], "xyz")))

    def export(self):
        return dict(scenario_id=self.base["scenario_id"], alpha=self.alpha, selection=self.selection,
                    complete=self.finished, anchor_state_hash=None if self.anchor_fingerprint is None else self.anchor_fingerprint["sha256"],
                    anchor_fingerprint=self.anchor_fingerprint, rows=self.rows,
                    missions=[r for r in self.mission_rows if self.anchor <= r["step"] < self.anchor+self.horizon])
