"""Public-only feature comparisons and candidate representation diagnostics."""

from dataclasses import asdict
import numpy as np

from core.env.observation_contract import FIELDS
from chapter3_bser.controllers.path_tracker import PathTracker
from chapter3_bser.objective import marginal_gain
from . import FEATURE_SCHEMA, TOLERANCE
from .provenance import digest


def feature_catalog():
    fields = []
    for field in FIELDS:
        for index in range(field.start, field.end):
            fields.append(dict(index=index, name=field.name, source="core/env/uav_env.py:UAVEnv._get_obs",
                               scale=field.normalization, update="observation refresh after guidance installation",
                               semantics=field.semantics))
    fields[18].update(source="core/env/uav_env.py:UAVEnv._get_obs -> UAVEnv._nearest_obstacle_distance override",
                      scale="online_unknown: clip(min(nearest AABB distance, obstacle_sensor_range)/10,0,1)",
                      semantics="existing range-capped obstacle sensor observation; audit copies it unchanged, never queries hidden obstacle geometry")
    auxiliary = (
        ("normalized_step", "public_state.step / max_steps", "every transition"),
        ("known_area_ratio", "mean(public occupancy.known_mask)", "last provider full refresh"),
        ("recent_collision_ratio", "last 20 transition collisions / 20", "every transition"),
        ("distance_since_information", "distance since known ratio/map revision increased / map diagonal", "every transition; reset on public information"),
        ("belief_entropy", "entropy / log(max(2, number of cells))", "last provider full refresh"),
        ("belief_peak", "public belief peak probability", "last provider full refresh"),
    )
    fields.extend(dict(index=28+i, name=name, source="SearchStateFeatureExtractor", scale=scale+"; clipped [0,1]", update=update)
                  for i, (name, scale, update) in enumerate(auxiliary))
    return dict(schema=FEATURE_SCHEMA, fields=fields, tolerance=TOLERANCE,
                checked_observation_detail="_get_obs clamps navigation norm and speed to eps before distance normalization; candidate_feature uses unclamped distance",
                guidance_rebuilt_fields=[6, 7, 8, 9, 10, 11, 15, 17],
                training_reference_semantic_review="since 38aee62, observe_transition delegates information-reset bookkeeping to synchronize_state; current scorer also synchronizes on same-step observation refresh. Audit preserves this ordering and records refresh timestamps; no feature correction",
                other_task_fields="waypoint_progress, finished, hold counters, known target and phase are copied from current observation; no hypothetical resets")


def exact_feature_id(feature):
    return digest(np.asarray(feature, dtype=np.float32).tolist())


def tolerance_groups(vectors, *, tolerance=TOLERANCE):
    """Stable first-representative max-absolute-distance clustering, atol only."""
    representatives, labels = [], []
    for value in vectors:
        value = np.asarray(value, dtype=float)
        match = next((i for i, prior in enumerate(representatives) if prior.shape == value.shape and np.max(np.abs(value-prior), initial=0) <= tolerance), None)
        if match is None:
            match = len(representatives)
            representatives.append(value)
        labels.append(match)
    return labels


def candidate_geometry(candidate):
    return dict(agent_id=int(candidate.agent_id), waypoint=list(candidate.waypoint),
                path=np.asarray(candidate.path_points).tolist())


def assignment_geometry(allocation):
    if allocation is None:
        return {}
    return {str(item.agent_id): dict(waypoint=list(item.waypoint), path=[list(p) for p in item.path])
            for item in allocation.search_assignments}


def guidance_geometry(guidance):
    if guidance is None:
        return {}
    return {str(item.agent_id): dict(final=list(item.final_waypoint), path=[list(p) for p in item.planned_path],
                                    tracking=list(item.tracking_waypoint), hold=bool(item.hold_state),
                                    hold_position=list(item.hold_position) if item.hold_state else None, reachable=bool(item.reachable))
            for item in guidance.agent_assignments if item.agent_id in (0, 1, 2)}


def compare_feature(actual, rebuilt, *, observation_step, public_step, observation_position, public_position,
                    semantic, public_tracking, effective_tracking, overlay_active=None):
    same_time = observation_step == public_step and np.allclose(observation_position, public_position, rtol=0, atol=TOLERANCE)
    overlay = (not np.allclose(public_tracking, effective_tracking, rtol=0, atol=TOLERANCE)
               if overlay_active is None else bool(overlay_active))
    if not same_time:
        return dict(comparable=False, reason="different_step_or_physical_state", delta=None, max_abs_difference=None,
                    observation_step=observation_step, public_state_step=public_step,
                    observation_position=list(observation_position), public_position=list(public_position),
                    stratum="C2_overlay" if overlay else "no_C2_overlay")
    difference = np.asarray(actual, dtype=float) - np.asarray(rebuilt, dtype=float)
    return dict(comparable=True, reason=None, delta=difference.tolist(), max_abs_difference=float(np.max(np.abs(difference))),
                observation_step=observation_step, public_state_step=public_step,
                stratum="C2_overlay" if overlay else "no_C2_overlay", semantic_waypoint=list(semantic),
                public_tracking_waypoint=list(public_tracking), effective_tracking_waypoint=list(effective_tracking))


def pool_audit(candidates, context, baseline, shadow, *, identity, decision_index):
    """Receives only public PlanningStateView and original pool; never an env."""
    ordered = sorted(candidates, key=lambda c: c.key)
    features = {c.key: shadow.candidate_feature(c, context.state) for c in ordered}
    values = {c.key: shadow.estimate_candidate_value(features[c.key]) for c in ordered}
    pool_hash = digest([dict(candidate_id=c.candidate_id, **candidate_geometry(c)) for c in ordered])
    scores, rounds = [], []
    if baseline.standby is not None:
        for algorithm in ("A", "B"):
            selected, used = [], set()
            while True:
                feasible = [c for c in ordered if c.agent_id not in used]
                gains = {c.key: marginal_gain(selected, c, baseline.standby, context) for c in feasible}
                eligible = [c for c in feasible if gains[c.key] > 1e-15]
                if not eligible:
                    break
                ranked = sorted(eligible, key=lambda c: (-(gains[c.key] + (.1*values[c.key] if algorithm == "B" else 0)), c.key))
                original_order = sorted(eligible, key=lambda c: (-gains[c.key], c.key))
                rank_changed = [c.key for c in original_order] != [c.key for c in ranked]
                winner = ranked[0]
                context_hash = digest(dict(pool=pool_hash, belief=context.belief, detection=context.detection_by_id,
                                           response=context.response_weight_by_id, standby=baseline.standby.key,
                                           selected_prefix=[c.key for c in selected]))
                round_rows = []
                for c in feasible:
                    row = dict(**identity, decision_index=decision_index, step=int(context.state.step), algorithm=algorithm,
                               greedy_round=len(selected), agent_id=c.agent_id, candidate_id=c.candidate_id,
                               candidate_key=c.key, candidate_pool_hash=pool_hash, context_hash=context_hash,
                               candidate_source=c.source,
                               selected_prefix=[v.candidate_id for v in selected], bser_marginal_gain=gains[c.key],
                               prediction=values[c.key], auxiliary_term=.1*values[c.key],
                               auxiliary_applied=algorithm == "B", final_score=gains[c.key]+(.1*values[c.key] if algorithm == "B" else 0),
                               ranking_changed_at_this_prefix=rank_changed,
                               eligible=gains[c.key] > 1e-15, selected=c.key == winner.key,
                               feature_id=exact_feature_id(features[c.key]), waypoint=c.waypoint,
                               path_hash=digest(np.asarray(c.path_points).tolist()))
                    scores.append(row)
                    round_rows.append(row)
                rounds.append(round_rows)
                selected.append(winner)
                used.add(winner.agent_id)
    representation = []
    for agent_id in sorted({c.agent_id for c in ordered}):
        pool = [c for c in ordered if c.agent_id == agent_id]
        vectors = [features[c.key] for c in pool]
        labels = tolerance_groups(vectors)
        predictions = np.array([values[c.key] for c in pool])
        positions = next(a.position for a in context.state.agents if a.agent_id == agent_id)
        previews = [PathTracker().tracking_target(agent_id, positions, c.path_points, c.waypoint) for c in pool]
        collisions = sum(len({tuple(c.waypoint) for c, label in zip(pool, labels) if label == group}) > 1 for group in set(labels))
        gains = np.array([marginal_gain((), c, baseline.standby, context) for c in pool]) if baseline.standby else np.zeros(len(pool))
        comparisons = []
        for left in range(len(pool)):
            for right in range(left+1, len(pool)):
                gap, auxiliary_gap = float(abs(gains[left]-gains[right])), float(.1*abs(predictions[left]-predictions[right]))
                comparisons.append(dict(left=pool[left].candidate_id, right=pool[right].candidate_id,
                                        bser_gap=gap, auxiliary_gap=auxiliary_gap, ratio=auxiliary_gap/gap if gap else None,
                                        ratio_reason=None if gap else "zero_original_gain_gap"))
        representation.append(dict(**identity, step=context.state.step, decision_index=decision_index, agent_id=agent_id,
                                   candidate_pool_hash=pool_hash, candidate_count=len(pool),
                                   endpoint_count=len({tuple(c.waypoint) for c in pool}),
                                   path_count=len({digest(np.asarray(c.path_points).tolist()) for c in pool}),
                                   initial_tracking_point_count=len(set(previews)),
                                   exact_feature_count=len({exact_feature_id(v) for v in vectors}),
                                   tolerance_feature_count=len(set(labels)), tolerance=TOLERANCE, tolerance_labels=labels,
                                   same_feature_different_endpoint_groups=collisions,
                                   exact_feature_ids=[exact_feature_id(v) for v in vectors],
                                   prediction_min=float(predictions.min()), prediction_max=float(predictions.max()), prediction_std=float(predictions.std()),
                                   bser_gain_min=float(gains.min()), bser_gain_max=float(gains.max()),
                                   gain_context="empty prefix, original baseline standby", candidate_gain_comparisons=comparisons))
    return representation, scores, pool_hash
