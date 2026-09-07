"""Read-only trace sink; consumes the authoritative SEARCH diagnostic rows.

No policy, planning, environment update, random sampling, or file writes here.
Action rows describe input state t and the transition t -> t+1. Mission rows
describe the resulting state, labelled by t with state_step=t+1 explicitly.
"""

import hashlib
import json
import math

import numpy as np

SCHEMA = "bser.searcher_residual_trace.v1"
TRACE_DEFINITIONS = {
    "step": "pre-action physical state t; action/collision is transition t->t+1; mission state_step is post-transition",
    "position": "copied pre-action runtime position; navigation target is pre-action installed runtime target",
    "waypoint_cursor": "read-only bridge.path_tracker.snapshot(agent).next_index; base path cursor, not C2 path progress",
    "raw_residual": "gated_residual_action before existing residual mode; dimensionless, same source as episode raw norm",
    "applied_residual": "actual residual command passed to env after existing mode/continuity adapter; dimensionless",
    "prior_action": "runtime cached _last_prior_acc; physical acceleration, not dimensionless residual command",
    "final_action": "runtime cached _agent_acc after prior+scaled residual and acceleration clipping; physical acceleration",
    "residual_prior_cosine": "legacy label: authoritative actor alignment_cosine(raw residual_mix, navigation unit), NOT cosine(applied residual, physical prior acceleration); original route/navigation/mix epsilon=1e-8 eligibility",
    "residual_contribution_ratio": "per-agent only if original diagnostic receives per-agent ratios; currently NA because runtime exposes team scalar",
    "team_residual_contribution_ratio": "mission row, exact runtime last_residual_contribution_ratio_search used by episode aggregate; no new formula and no agent broadcast",
    "c2": "pre-action snapshot active_agent_ids and mode_by_agent; no recovery state update",
    "allocation_change_event": "public BSER assignment identity/kind/final waypoint/planned path/reachable/hold change; excludes pure cursor motion and C2 overlay",
    "truth": "executor_target_distance uses existing evaluator ground-truth diagnostic getter, audit-only; never fed to control",
    "same_state": "time/agent pairing is NOT same-state counterfactual after trajectory divergence",
}


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def vector_fields(prefix, vector, axes):
    return {f"{prefix}_{axis}": None if vector is None else float(vector[i]) for i, axis in enumerate(axes)}


def guidance_signature(guidance):
    values = []
    for agent in range(4):
        try:
            item = guidance.assignment_for(agent)
        except KeyError:
            continue
        values.append((agent, str(item.assignment_id), str(item.assignment_kind),
                       tuple(item.final_waypoint), getattr(item, "planned_path", None), bool(item.reachable), bool(item.hold_state)))
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


class SearcherResidualTrace:
    def __init__(self, scenario, mode, transition_type):
        self.base = dict(scenario_id=str(scenario["scenario_id"]), scenario_seed=int(scenario["scenario_seed"]),
                         transition_type=transition_type, mode=mode)
        self.action_rows, self.mission_rows, self.diagnostic_rows = [], [], []
        self.previous = dict(found=False, contact=False, success=False)

    def before_step(self, *, state, env, bridge, recovery, guidance):
        runtime = env.unwrapped
        snapshot = None if recovery is None else recovery.snapshot()
        self.before = dict(step=int(state.step), positions=array(runtime._agent_pos),
                           targets=array(runtime._nav_targets) if hasattr(runtime, "_nav_targets") else None,
                           cursors={a: bridge.path_tracker.snapshot(a).next_index for a in range(3)},
                           active=() if snapshot is None else snapshot.active_agent_ids,
                           modes={} if snapshot is None else dict(snapshot.mode_by_agent),
                           allocation=guidance_signature(guidance))

    def after_step(self, *, state, env, task, metadata, result, guidance, recovery_snapshot):
        runtime, before = env.unwrapped, self.before
        prior = array(runtime._last_prior_acc) if hasattr(runtime, "_last_prior_acc") else None
        final = array(runtime._agent_acc) if hasattr(runtime, "_agent_acc") else None
        for diagnostic in self.diagnostic_rows:
            row = dict(diagnostic)
            agent = row["agent_id"]
            raw, applied = row.pop("raw_residual"), row.pop("applied_residual")
            row.update(self.base, step=before["step"], stage=int(metadata.stage_before),
                       waypoint_cursor=before["cursors"][agent], c2_active=agent in before["active"],
                       c2_state_or_tier=before["modes"].get(agent),
                       prior_norm=None if prior is None else float(np.linalg.norm(prior[agent])))
            for prefix, values, axes in (("position", before["positions"][agent], "xyz"),
                                         ("navigation_target", None if before["targets"] is None else before["targets"][agent], "xyz"),
                                         ("prior_action", None if prior is None else prior[agent], "012"),
                                         ("raw_residual", raw, "012"), ("applied_residual", applied, "012"),
                                         ("final_action", None if final is None else final[agent], "012")):
                row.update(vector_fields(prefix, values, axes))
            self.action_rows.append(row)
        self.diagnostic_rows.clear()
        flags = dict(found=bool(task.target_found), contact=bool(runtime.capture_contact_step_count), success=bool(task.mission_complete))
        executor = array(runtime._agent_pos)[3]
        target = array(env.get_target_state().position)
        ratio = getattr(runtime, "last_residual_contribution_ratio_search", None)
        ratio = None if ratio is None or not math.isfinite(float(ratio)) else float(ratio)
        row = dict(self.base, step=before["step"], state_step=int(state.step), mission_stage=int(metadata.stage_after), **flags,
                   **{f"{key}_event": value and not self.previous[key] for key, value in flags.items()},
                   known_map_fraction=float(np.asarray(state.occupancy.known_mask).mean()),
                   belief_entropy=float(state.target_belief.entropy), belief_peak=float(state.target_belief.peak_probability),
                   executor_target_distance=float(np.linalg.norm(executor-target)),
                   bser_replan_event=bool(result.replanned), allocation_change_event=before["allocation"] != guidance_signature(guidance),
                   search_recovery_active_agents=[] if recovery_snapshot is None else list(recovery_snapshot.active_agent_ids),
                   team_collision_event=bool(array(runtime.collision_flags).any()),
                   team_residual_contribution_ratio=ratio if int(metadata.stage_before) == 0 else None,
                   **vector_fields("executor", executor, "xyz"))
        self.mission_rows.append(row)
        self.previous = flags
