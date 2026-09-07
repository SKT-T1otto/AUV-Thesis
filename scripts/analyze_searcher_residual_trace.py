"""Strict same-key, no-interpolation offline trace analysis (stdlib only)."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

MODES = ("full_prrac", "searcher_residual_off")
HELP, HURT = "full_success_searcher_off_fail", "full_fail_searcher_off_success"
CONTEXT = ("raw_residual_norm", "applied_residual_norm", "residual_contribution_ratio", "residual_prior_cosine",
           "negative_alignment", "route_active", "collision_event", "c2_active")
KINDS = ("action_difference", "navigation_target_difference", "waypoint_cursor_difference", "collision_state_difference",
         "c2_state_difference", "position_separation_0p5", "position_separation_1p0")
LIMITATIONS = [
    "The 15 discordant scenarios were selected post-hoc from 100 outcomes; this is mechanism diagnosis, not an independent validation set.",
    "After trajectory divergence, matched step/agent rows are not the same counterfactual state; later residual differences are not single-step causal effects.",
    "Inspect pre-divergence state, first-divergence context and repeated help/hurt patterns; trace equality does not prove equality of all hidden runtime state.",
    "Negative alignment is not automatically harmful; correction can support avoidance/recovery. Future-event associations are not necessity labels.",
    "0.5m/1.0m thresholds are diagnostic, not physical safety thresholds. No collision-imminent signal is invented.",
    "Contribution ratio is the original team scalar on mission rows; per-agent unavailable ratios remain NA. Group means weight scenarios equally.",
    "Windows before divergence/found use [max(0,event_step-20),event_step); absent event means unavailable, not an all-episode substitute.",
    "Follow-up events use the same agent at t+1 through t+5; incomplete/missing windows are censored, not counted as no event. Same-transition collision is reported separately.",
]


def decode(value):
    if value is None or value.strip().lower() in ("", "na", "none", "null", "nan"):
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        if value.startswith(("[", "{")):
            return json.loads(value)
        result = float(value)
    except (ValueError, json.JSONDecodeError):
        return value
    if not math.isfinite(result):
        raise ValueError("nonfinite trace value")
    return result


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return [{key: decode(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    fields = sorted({k for row in rows for k in row}) or ["scenario_id"]
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "NA" if row.get(key) is None else json.dumps(row[key], sort_keys=True) if isinstance(row[key], (dict, list)) else row[key] for key in fields})


def numeric(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def distance(left, right, prefix, axes="xyz"):
    values = [(left.get(f"{prefix}_{axis}"), right.get(f"{prefix}_{axis}")) for axis in axes]
    return math.sqrt(sum((a-b)**2 for a, b in values)) if all(numeric(a) and numeric(b) for a, b in values) else None


def changed(left, right, fields):
    valid = [(left.get(k), right.get(k)) for k in fields if left.get(k) is not None and right.get(k) is not None]
    return any(a != b for a, b in valid) if valid else None


def action_index(rows):
    result = {}
    for row in rows:
        key = int(row["step"]), int(row["agent_id"])
        if key in result or row["step"] != key[0] or row["agent_id"] != key[1] or key[0] < 0 or key[1] not in (0, 1, 2):
            raise ValueError("duplicate/invalid trace step-agent key")
        result[key] = row
    return result


def mean(values):
    valid = [v for v in values if numeric(v)]
    return dict(mean=statistics.mean(valid) if valid else None, valid=len(valid), missing=len(values)-len(valid))


def before_window(actions, missions, end):
    if end is None:
        return dict(negative_alignment=None, residual_ratio=None, alignment_valid_samples=0, ratio_valid_steps=0, observed_steps=0)
    selected = [r for r in actions if max(0, end-20) <= r["step"] < end]
    alignment = [r.get("negative_alignment") for r in selected]
    ratios = [r.get("team_residual_contribution_ratio") for r in missions if max(0, end-20) <= r["step"] < end]
    return dict(negative_alignment=mean(alignment)["mean"], residual_ratio=mean(ratios)["mean"],
                alignment_valid_samples=mean(alignment)["valid"], ratio_valid_steps=mean(ratios)["valid"],
                observed_steps=len({r["step"] for r in selected}))


def followups(actions):
    indexed = action_index(actions)
    result = dict(negative_alignment_samples=0, complete_windows=0, censored_windows=0, collision=0,
                  c2_activation=0, waypoint_switch=0, navigation_target_change=0, none=0, same_transition_collision=0)
    for (step, agent), row in sorted(indexed.items()):
        if row.get("negative_alignment") is not True:
            continue
        result["negative_alignment_samples"] += 1
        result["same_transition_collision"] += int(row.get("collision_event") is True)
        flags, previous = [], row
        for future in range(step+1, step+6):
            next_row = indexed.get((future, agent))
            if next_row is None:
                break
            nav_delta = distance(previous, next_row, "navigation_target")
            if nav_delta is None or any(next_row.get(k) is None for k in ("collision_event", "c2_active", "waypoint_switch_event")) or previous.get("c2_active") is None:
                break
            flags.append(dict(collision=bool(next_row["collision_event"]),
                              c2_activation=bool(next_row["c2_active"] and not previous["c2_active"]),
                              waypoint_switch=bool(next_row["waypoint_switch_event"]), navigation_target_change=nav_delta > 1e-8))
            previous = next_row
        if len(flags) != 5:
            result["censored_windows"] += 1
            continue
        result["complete_windows"] += 1
        events = {key: any(f[key] for f in flags) for key in flags[0]}
        for key, value in events.items():
            result[key] += int(value)
        result["none"] += int(not any(events.values()))
    return result


def analyze_scenario(sid, group, action_pair, mission_pair):
    indices = [action_index(rows) for rows in action_pair]
    shared = sorted(set(indices[0]) & set(indices[1]))
    first, coverage = {}, {kind: 0 for kind in KINDS}
    for key in shared:
        left, right = (index[key] for index in indices)
        action_delta = distance(left, right, "final_action", "012")
        nav_delta = distance(left, right, "navigation_target")
        separation = distance(left, right, "position")
        flags = dict(action_difference=None if action_delta is None else action_delta > 1e-8,
                     navigation_target_difference=None if nav_delta is None else nav_delta > 1e-8,
                     waypoint_cursor_difference=changed(left, right, ("waypoint_cursor",)),
                     collision_state_difference=changed(left, right, ("collision_event", "collision_streak")),
                     c2_state_difference=changed(left, right, ("c2_active", "c2_state_or_tier")),
                     position_separation_0p5=None if separation is None else separation >= .5,
                     position_separation_1p0=None if separation is None else separation >= 1.)
        for kind, flag in flags.items():
            coverage[kind] += int(flag is not None)
            if flag and kind not in first:
                first[kind] = key
    divergence = dict(scenario_id=sid, transition_type=group, same_state_counterfactual=False,
                      paired_action_rows=len(shared), r0_unpaired_rows=len(indices[0])-len(shared),
                      r1_unpaired_rows=len(indices[1])-len(shared), comparison_coverage=coverage)
    for kind in KINDS:
        key = first.get(kind)
        divergence[f"first_{kind}_step"] = None if key is None else key[0]
        divergence[f"first_{kind}_agent_id"] = None if key is None else key[1]
        context = {}
        if key is not None:
            for label, index, mission in zip(("R0", "R1"), indices, mission_pair):
                by_step = {r["step"]: r for r in mission}
                context[label] = {}
                for phase, step in (("before", key[0]-1), ("at", key[0])):
                    row = index.get((step, key[1]), {})
                    context[label][phase] = {name: row.get(name) for name in CONTEXT}
                    context[label][phase]["team_residual_contribution_ratio"] = by_step.get(step, {}).get("team_residual_contribution_ratio")
        divergence[f"first_{kind}_context"] = context or None
    summary = dict(scenario_id=sid, transition_type=group,
                   first_position_separation_0p5_step=divergence["first_position_separation_0p5_step"],
                   first_position_separation_1p0_step=divergence["first_position_separation_1p0_step"])
    for short, kind in (("nav_target", "navigation_target"), ("waypoint", "waypoint_cursor"), ("collision", "collision_state"), ("c2", "c2_state")):
        summary[f"first_{short}_difference_step"] = divergence[f"first_{kind}_difference_step"]
    events = {}
    for label, actions, mission in zip(("R0", "R1"), action_pair, mission_pair):
        found_rows = [r for r in mission if r.get("found_event") is True]
        found_step = found_rows[0]["state_step"] if found_rows else None
        summary[f"{label}_found"] = mission[-1]["found"] if mission else None
        summary[f"{label}_found_step"] = found_step
        summary[f"{label}_collision_count_pre_found"] = sum(bool(r["collision_event"]) for r in actions) if actions and all(r.get("collision_event") is not None for r in actions) else None
        summary[f"{label}_{'raw_' if label == 'R1' else ''}negative_alignment_rate_pre_found"] = mean([r.get("negative_alignment") for r in actions])["mean"]
        events[label] = followups(actions)
    summary["R0_residual_contribution_ratio_pre_found"] = mean([r.get("team_residual_contribution_ratio") for r in mission_pair[0]])["mean"]
    for suffix, end in (("before_first_position_divergence", summary["first_position_separation_0p5_step"]),
                        ("20steps_before_found", summary["R0_found_step"])):
        window = before_window(action_pair[0], mission_pair[0], end)
        summary[f"R0_negative_alignment_{suffix}"] = window["negative_alignment"]
        summary[f"R0_residual_ratio_{suffix}"] = window["residual_ratio"]
        for key in ("alignment_valid_samples", "ratio_valid_steps", "observed_steps"):
            summary[f"{suffix}_{key}"] = window[key]
    return divergence, summary, events


def analyze(full_trace, searcher_off_trace, output_dir):
    folders = [Path(p).resolve() for p in (full_trace, searcher_off_trace)]
    if folders[0].parent != folders[1].parent:
        raise ValueError("trace folders must share one trace_manifest.json parent")
    root = folders[0].parent
    manifest = json.loads((root/"trace_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "bser.searcher_residual_trace.v1" or manifest.get("status") != "completed":
        raise ValueError("trace manifest incomplete, incompatible or historical reproduction mismatched")
    selected = {str(row["scenario_id"]): row for row in manifest["selected"]}
    if len(selected) != manifest["scenario_count"] or manifest["episode_runs_expected"] != 2*len(selected):
        raise ValueError("selected manifest count mismatch")
    if not manifest.get("smoke") and (len(selected) != 15 or sum(r["transition_type"] == HELP for r in selected.values()) != 5 or sum(r["transition_type"] == HURT for r in selected.values()) != 10):
        raise ValueError("formal traces require 5 help + 10 hurt")
    data = []
    for folder, mode in zip(folders, MODES):
        bundle = {}
        for kind, filename in (("action", "searcher_action_trace.csv"), ("mission", "mission_step_trace.csv"), ("episode", "episode_evaluation.csv")):
            path = folder/filename
            if manifest["trace_files"].get(path.relative_to(root).as_posix()) != file_sha(path):
                raise ValueError("trace file hash mismatch")
            rows = read_csv(path)
            grouped = {sid: [] for sid in selected}
            for row in rows:
                sid = str(row["scenario_id"])
                if sid not in selected or row.get("mode", row.get("evaluation_mode")) != mode or row["scenario_seed"] != selected[sid]["scenario_seed"]:
                    raise ValueError("trace identity mismatch")
                if kind != "episode" and row["transition_type"] != selected[sid]["transition_type"]:
                    raise ValueError("trace transition mismatch")
                grouped[sid].append(row)
            for sid, items in grouped.items():
                if not items:
                    raise ValueError(f"missing {kind} rows for {sid}")
                if kind == "action":
                    indexed = action_index(items)
                    steps = sorted({key[0] for key in indexed})
                    if steps != list(range(steps[-1]+1)) or any((step, agent) not in indexed for step in steps for agent in range(3)):
                        raise ValueError("missing PRE_FOUND step/agent rows; no interpolation")
                    if any(row["stage"] != 0 for row in items):
                        raise ValueError("action trace contains post-found rows")
                elif kind == "mission":
                    items.sort(key=lambda r: r["step"])
                    if [r["step"] for r in items] != list(range(len(items))) or any(r["state_step"] != r["step"]+1 for r in items):
                        raise ValueError("mission step duplicate/gap/off-by-one")
                elif len(items) != 1:
                    raise ValueError("duplicate episode result")
            bundle[kind] = grouped
        for sid in selected:
            mission, episode = bundle["mission"][sid], bundle["episode"][sid][0]
            for key, field in (("found", "found"), ("contact", "contact_episode"), ("success", "success")):
                if mission[-1][key] != episode[field]:
                    raise ValueError("mission endpoint differs from episode outcome")
            if len(mission) != episode["episode_length"]:
                raise ValueError("truncated mission trace")
            found_events = [row for row in mission if row.get("found_event") is True]
            if len(found_events) != int(bool(episode["found"])):
                raise ValueError("missing/duplicate found_event")
            search_steps = found_events[0]["state_step"] if found_events else len(mission)
            if len(bundle["action"][sid]) != 3*search_steps:
                raise ValueError("truncated or extra PRE_FOUND action rows")
        data.append(bundle)
    output = Path(output_dir).resolve()
    if any(output == folder or folder in output.parents or output in folder.parents for folder in folders):
        raise ValueError("analysis output overlaps input trace folders")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("nonempty output; overwrite refused")
    divergences, summaries, event_records = [], [], {}
    for sid, selection in sorted(selected.items()):
        row, summary, events = analyze_scenario(sid, selection["transition_type"], [d["action"][sid] for d in data], [d["mission"][sid] for d in data])
        divergences.append(row)
        summaries.append(summary)
        event_records[sid] = events
    groups = {}
    for label, transition in (("help", HELP), ("hurt", HURT)):
        rows = [r for r in summaries if r["transition_type"] == transition]
        group = dict(scenario_count=len(rows))
        for field in ("first_position_separation_0p5_step", "first_position_separation_1p0_step", "R0_negative_alignment_before_first_position_divergence", "R0_residual_ratio_before_first_position_divergence"):
            group[field] = mean([r[field] for r in rows])
        group["followup_events"] = {}
        for mode in ("R0", "R1"):
            counts = [event_records[r["scenario_id"]][mode] for r in rows]
            group["followup_events"][mode] = dict(per_scenario={r["scenario_id"]: event_records[r["scenario_id"]][mode] for r in rows},
                scenario_equal_event_rates={key: mean([c[key]/c["complete_windows"] if c["complete_windows"] else None for c in counts]) for key in ("collision", "c2_activation", "waypoint_switch", "navigation_target_change", "none")},
                pooled_counts_descriptive_only={key: sum(c[key] for c in counts) for key in counts[0]} if counts else {})
        groups[label] = group
    summary = dict(scenario_count=len(selected), help_count=groups["help"]["scenario_count"], hurt_count=groups["hurt"]["scenario_count"],
                   episode_runs_expected=2*len(selected), smoke=bool(manifest.get("smoke")), groups=groups, limitations=LIMITATIONS,
                   trace_manifest_sha256=file_sha(root/"trace_manifest.json"), difference_tolerance=1e-8,
                   first_divergence_tie_break="earliest exact step, then smallest agent_id; no interpolation", same_state_counterfactual=False)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/"paired_first_divergence.csv", divergences)
    write_csv(output/"residual_help_trace_summary.csv", [r for r in summaries if r["transition_type"] == HELP])
    write_csv(output/"residual_hurt_trace_summary.csv", [r for r in summaries if r["transition_type"] == HURT])
    with (output/"searcher_residual_trace_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-trace", type=Path, required=True)
    parser.add_argument("--searcher-off-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        analyze(args.full_trace, args.searcher_off_trace, args.output_dir)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"trace analysis failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
