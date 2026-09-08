"""Offline categorical branch sensitivity analysis; no simulator/model imports."""

import argparse
import json
from pathlib import Path

if __package__:
    from . import analyze_searcher_residual_trace as trace
else:
    import analyze_searcher_residual_trace as trace

SCHEMA = "bser.searcher_residual_branch_sensitivity.v1"
ALPHAS = (0., .25, .5, .75, 1.)
HORIZON = 10
TOLERANCE = 1e-6
BRANCH_TOLERANCE = 1e-8
LIMITATIONS = [
    "Post-hoc selected 5 help and 10 hurt scenarios; mechanism diagnosis only, no independent significance test or failure-causation claim.",
    "Alpha=0 suppresses only the selected branch agents from the full_prrac anchor for ten steps; it is NOT historical searcher_residual_off.",
    "Same-state intervention is established at the anchor only; later states may diverge. Hash equality covers the explicit stable runtime inventory, not unknown excluded state.",
    "0.1m/0.5m are diagnostic distances, not safety thresholds. A local branch change is not proof of a final Success effect.",
    "Waypoint/navigation outcomes are categorical. Re-entering a category across alpha is non_monotonic; more than two unordered categories without re-entry is unresolved, not an assumed numeric ordering.",
    "Switch intervals refer to the first divergent state on the sampled grid, not an exact threshold. Alpha1 reproduction or same-state failure prohibits scenario interpretation.",
    "Navigation-target changes can include geometric tracking changes; classification alone does not establish that an internal discrete PathTracker predicate caused them.",
]


def nonempty_output(output, sources):
    output = Path(output).resolve()
    for source in sources:
        source = Path(source).resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("output overlaps protected historical/trace source")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("nonempty output; overwrite refused")
    return output


def select_anchor(divergence, left_rows, right_rows):
    indices = [trace.action_index(rows) for rows in (left_rows, right_rows)]
    first = dict(navigation_target=None, waypoint_cursor=None)
    for key in sorted(set(indices[0]) & set(indices[1])):
        left, right = (index[key] for index in indices)
        nav = trace.distance(left, right, "navigation_target")
        if nav is not None and nav > BRANCH_TOLERANCE and first["navigation_target"] is None:
            first["navigation_target"] = key[0]
        if left.get("waypoint_cursor") is not None and right.get("waypoint_cursor") is not None and left["waypoint_cursor"] != right["waypoint_cursor"] and first["waypoint_cursor"] is None:
            first["waypoint_cursor"] = key[0]
    for name, actual in first.items():
        expected = divergence.get(f"first_{name}_difference_step")
        if expected != actual:
            raise ValueError(f"paired divergence disagrees with source traces: {name}")
    valid = [v for v in first.values() if v is not None]
    if not valid:
        raise ValueError("missing discrete branch step; scenario not analyzable")
    step = min(valid)
    agents = []
    for agent in range(3):
        left, right = (index.get((step, agent)) for index in indices)
        if left is None or right is None:
            raise ValueError("missing branch step/agent")
        delta = trace.distance(left, right, "navigation_target")
        if delta is None or left.get("waypoint_cursor") is None or right.get("waypoint_cursor") is None:
            raise ValueError("missing branch identification fields")
        if delta > BRANCH_TOLERANCE or left["waypoint_cursor"] != right["waypoint_cursor"]:
            agents.append(agent)
    if not agents:
        raise ValueError("no branch agents")
    return dict(scenario_id=str(divergence["scenario_id"]), transition_type=divergence["transition_type"],
                branch_step=step, anchor_step=max(0, step-1), branch_agents=agents)


def rows_index(rows):
    return trace.action_index(rows)


def control_reproduction(control, reference, missions, tolerance=TOLERANCE):
    mismatches = []
    ref = rows_index(reference)
    for row in control["rows"]:
        key = int(row["step"]), int(row["agent_id"])
        original = ref.get(key)
        if original is None:
            mismatches.append(dict(key=key, field="missing_historical_action_state"))
            continue
        for prefix in ("position", "navigation_target"):
            delta = trace.distance(row, original, prefix)
            if delta is None or delta > tolerance:
                mismatches.append(dict(key=key, field=prefix, difference=delta))
        fields = ["waypoint_cursor", "c2_active", "c2_state_or_tier"]
        if row["sample_kind"] == "transition":
            fields.append("collision_event")
        for field in fields:
            if field not in row or field not in original or row[field] != original[field]:
                mismatches.append(dict(key=key, field=field))
    mission_index = {r["step"]: r for r in missions}
    for row in control["missions"]:
        if row["step"] not in mission_index or row.get("found_event") != mission_index[row["step"]].get("found_event"):
            mismatches.append(dict(step=row["step"], field="found_event"))
    if not control["complete"]:
        mismatches.append(dict(field="incomplete_control_window"))
    return mismatches


def branch_equal(left, right):
    """Equality of joint Searcher branch state; categorical, tolerance explicit."""
    for a, b in zip(left, right):
        delta = trace.distance(a, b, "navigation_target")
        if delta is None or a.get("waypoint_cursor") is None or b.get("waypoint_cursor") is None:
            return None
        if a["waypoint_cursor"] != b["waypoint_cursor"] or delta > BRANCH_TOLERANCE:
            return False
    return True


def classify(branches, anchor, horizon=HORIZON):
    alphas = sorted(branches)
    first, interval = None, None
    unordered, reentered = False, False
    for step in range(anchor+1, anchor+horizon+1):
        representatives, labels = [], []
        for alpha in alphas:
            current = [branches[alpha].get((step, a)) for a in range(3)]
            if any(r is None for r in current):
                return dict(classification="unresolved", alpha_switch_interval=None, first_discrete_branch_divergence=None)
            label = None
            for i, representative in enumerate(representatives):
                same = branch_equal(current, representative)
                if same is None:
                    return dict(classification="unresolved", alpha_switch_interval=None, first_discrete_branch_divergence=None)
                if same:
                    label = i
                    break
            if label is None:
                label = len(representatives)
                representatives.append(current)
            labels.append(label)
        runs = [v for i, v in enumerate(labels) if not i or labels[i-1] != v]
        reentered |= len(set(runs)) != len(runs)
        unordered |= len(set(labels)) > 2
        if len(set(labels)) > 1 and first is None:
            first = step
            if len(runs) == 2:
                index = next(i for i in range(1, len(labels)) if labels[i] != labels[i-1])
                interval = [alphas[index-1], alphas[index]]
    kind = "non_monotonic" if reentered else "unresolved" if unordered else "stable" if first is None else "immediate_threshold" if first == anchor+1 else "delayed_threshold"
    return dict(classification=kind, alpha_switch_interval=interval if kind in ("immediate_threshold", "delayed_threshold") else None,
                first_discrete_branch_divergence=first)


def summarize_scenario(results, reference, missions, expected_alphas=ALPHAS, horizon=HORIZON, tolerance=TOLERANCE):
    alphas = [r["alpha"] for r in results]
    if len(alphas) != len(set(alphas)) or set(alphas) != set(expected_alphas) or 1. not in alphas:
        raise ValueError("missing/duplicate alpha branches")
    by_alpha = {r["alpha"]: r for r in results}
    selection = results[0]["selection"]
    if any(r["selection"] != selection for r in results):
        raise ValueError("branch selection differs across alpha")
    hashes = {r["anchor_state_hash"] for r in results}
    control_errors = control_reproduction(by_alpha[1.], reference, missions, tolerance)
    complete = all(r["complete"] for r in results)
    same_state = len(hashes) == 1 and None not in hashes
    valid = complete and same_state and not control_errors
    anchor = selection["anchor_step"]
    indices = {alpha: rows_index(r["rows"]) for alpha, r in by_alpha.items()}
    expected_keys = {(t, a) for t in range(anchor, anchor+horizon+1) for a in range(3)}
    valid &= all(set(index) == expected_keys for index in indices.values())
    classification = classify(indices, anchor, horizon) if valid else dict(classification="unresolved", alpha_switch_interval=None, first_discrete_branch_divergence=None)
    scenario_rows, step_rows = [], []
    for alpha in sorted(by_alpha):
        result, index, control = by_alpha[alpha], indices[alpha], indices[1.]
        first_waypoint, first_nav, first_p1, first_p5 = None, None, None, None
        endpoint_distances, c2_activation = [], False
        for key, row in sorted(index.items()):
            baseline = control.get(key)
            delta = None if baseline is None else trace.distance(row, baseline, "position")
            step_rows.append(dict(row, position_separation_from_alpha1=delta))
            if not valid:
                continue
            step, agent = key
            nav_delta = trace.distance(row, baseline, "navigation_target")
            if step > anchor and row["waypoint_cursor"] != baseline["waypoint_cursor"] and first_waypoint is None:
                first_waypoint = step
            if step > anchor and nav_delta is not None and nav_delta > BRANCH_TOLERANCE and first_nav is None:
                first_nav = step
            if delta is not None and delta >= .1 and first_p1 is None:
                first_p1 = step
            if delta is not None and delta >= .5 and first_p5 is None:
                first_p5 = step
            if step == anchor+horizon:
                endpoint_distances.append(delta)
            previous = index.get((step-1, agent))
            c2_activation |= previous is not None and not previous["c2_active"] and row["c2_active"]
        scenario_rows.append(dict(selection, alpha=alpha, anchor_state_hash=result["anchor_state_hash"],
            same_state_pass=same_state, control_reproduction_pass=not control_errors, interpretation_allowed=bool(valid),
            **classification,
            immediate_waypoint_difference_vs_alpha1=(first_waypoint == anchor+1) if valid else None,
            immediate_navigation_target_difference_vs_alpha1=(first_nav == anchor+1) if valid else None,
            first_waypoint_difference_step_vs_alpha1=first_waypoint,
            first_navigation_target_difference_step_vs_alpha1=first_nav,
            first_position_separation_0p1_step_vs_alpha1=first_p1, first_position_separation_0p5_step_vs_alpha1=first_p5,
            position_separation_at_horizon=max(endpoint_distances) if endpoint_distances and valid else None,
            collision_within_horizon=any(r.get("collision_event") is True for r in result["rows"]) if valid else None,
            c2_activation_within_horizon=bool(c2_activation) if valid else None,
            found_within_horizon=any(r.get("found_event") is True for r in result["missions"]) if valid else None))
    return scenario_rows, step_rows, dict(scenario_id=selection["scenario_id"], interpretation_allowed=bool(valid),
        same_state_pass=same_state, control_reproduction_mismatches=control_errors, complete=complete)


def analyze(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    # An analysis child is allowed, but source files and existing outputs are not.
    if output == source or output in source.parents:
        raise ValueError("analysis output overlaps source")
    nonempty_output(output, [])
    manifest_path = source/"branch_sensitivity_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "completed":
        raise ValueError("branch manifest failed/incomplete; interpretation prohibited")
    if manifest.get("training_update") is not False or manifest.get("explore") is not False:
        raise ValueError("invalid diagnostic protocol")
    nonempty_output(output, [manifest[k] for k in ("source_historical_root", "source_trace_root") if k in manifest])
    for filename in ("branch_sensitivity_step_trace.csv", "branch_sensitivity_scenario.csv", "selected_anchors.csv"):
        if manifest["output_sha256"].get(filename) != trace.file_sha(source/filename):
            raise ValueError("branch output hash mismatch")
    rows = trace.read_csv(source/"branch_sensitivity_scenario.csv")
    selected = {str(r["scenario_id"]): r for r in manifest["selected"]}
    if len(selected) != len(manifest["selected"]) or manifest["horizon"] != HORIZON:
        raise ValueError("duplicate selected scenarios or incorrect horizon")
    expected = {(sid, a) for sid in selected for a in manifest["alphas"]}
    keys = [(str(r["scenario_id"]), r["alpha"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("scenario/alpha set mismatch")
    if any(r.get("interpretation_allowed") is not True or r.get("same_state_pass") is not True or r.get("control_reproduction_pass") is not True for r in rows):
        raise ValueError("failed scenario; interpretation prohibited")
    if not manifest["smoke"] and (len(selected) != 15 or manifest["branch_runs_expected"] != 75 or manifest["alphas"] != list(ALPHAS)):
        raise ValueError("formal protocol mismatch")
    if manifest["branch_runs_expected"] != len(expected):
        raise ValueError("branch count mismatch")
    step_rows = trace.read_csv(source/"branch_sensitivity_step_trace.csv")
    step_keys = [(str(r["scenario_id"]), r["alpha"], r["step"], r["agent_id"]) for r in step_rows]
    expected_steps = {(sid, alpha, step, agent) for sid, selection in selected.items() for alpha in manifest["alphas"]
                      for step in range(int(selection["anchor_step"]), int(selection["anchor_step"])+HORIZON+1) for agent in range(3)}
    if len(set(step_keys)) != len(step_keys) or set(step_keys) != expected_steps:
        raise ValueError("missing/duplicate branch step rows; no interpolation")
    for sid, selection in selected.items():
        scenario_rows = [r for r in rows if str(r["scenario_id"]) == sid]
        hashes = {r["anchor_state_hash"] for r in scenario_rows}
        if len(hashes) != 1 or None in hashes:
            raise ValueError("same-state hash mismatch")
        indices = {alpha: rows_index([r for r in step_rows if str(r["scenario_id"]) == sid and r["alpha"] == alpha]) for alpha in manifest["alphas"]}
        calculated = classify(indices, int(selection["anchor_step"]))
        for row in scenario_rows:
            if any(row[k] != calculated[k] for k in calculated) or any(row[k] != selection[k] for k in ("anchor_step", "branch_step", "branch_agents", "transition_type")):
                raise ValueError("scenario summary differs from branch trace/selection")
    summaries, groups = {}, {}
    for name, transition in (("help", trace.HELP), ("hurt", trace.HURT)):
        result = []
        for sid, selection in sorted(selected.items()):
            if selection["transition_type"] != transition:
                continue
            branch_rows = sorted([r for r in rows if str(r["scenario_id"]) == sid], key=lambda r: r["alpha"])
            base = branch_rows[0]
            if any(r["transition_type"] != transition or r["classification"] != base["classification"] or r["alpha_switch_interval"] != base["alpha_switch_interval"] for r in branch_rows):
                raise ValueError("inconsistent scenario classification")
            reductions = [1-r["alpha"] for r in branch_rows if r["alpha"] < 1 and
                          (r["first_waypoint_difference_step_vs_alpha1"] is not None or r["first_navigation_target_difference_step_vs_alpha1"] is not None)]
            result.append(dict(scenario_id=sid, transition_type=transition, classification=base["classification"],
                alpha_switch_interval=base["alpha_switch_interval"], first_discrete_branch_divergence=base["first_discrete_branch_divergence"],
                smallest_sampled_alpha_reduction_with_discrete_divergence=min(reductions) if reductions else None,
                position_separation_at_horizon_by_alpha={str(r["alpha"]): r["position_separation_at_horizon"] for r in branch_rows}))
        counts = {kind: sum(r["classification"] == kind for r in result) for kind in ("stable", "immediate_threshold", "delayed_threshold", "non_monotonic", "unresolved")}
        summaries[name] = result
        groups[name] = dict(scenario_count=len(result), classification_counts=counts,
            descriptive_rates={k: v/len(result) if result else None for k, v in counts.items()},
            alpha_switch_intervals=[r["alpha_switch_interval"] for r in result],
            first_discrete_branch_divergence=trace.mean([r["first_discrete_branch_divergence"] for r in result]),
            smallest_sampled_alpha_reduction_with_discrete_divergence=trace.mean([r["smallest_sampled_alpha_reduction_with_discrete_divergence"] for r in result]),
            mean_position_separation_at_horizon_by_alpha={str(a): trace.mean([r["position_separation_at_horizon"] for r in rows if r["transition_type"] == transition and r["alpha"] == a]) for a in manifest["alphas"]})
    if not manifest["smoke"] and (groups["help"]["scenario_count"] != 5 or groups["hurt"]["scenario_count"] != 10):
        raise ValueError("formal help/hurt count mismatch")
    summary = dict(schema=SCHEMA, scenario_count=len(selected), branch_runs_expected=len(expected),
        groups=groups, limitations=LIMITATIONS, smoke=manifest["smoke"],
        source_manifest_sha256=trace.file_sha(manifest_path), alphas=manifest["alphas"], horizon=manifest["horizon"])
    output.mkdir(parents=True, exist_ok=True)
    for name, result in summaries.items():
        trace.write_csv(output/f"branch_sensitivity_{name}.csv", result)
    (output/"branch_sensitivity_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        analyze(args.input_dir, args.output_dir)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"branch analysis failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
