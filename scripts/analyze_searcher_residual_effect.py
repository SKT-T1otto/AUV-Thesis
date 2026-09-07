"""Read-only paired residual-ablation analysis; Python standard library only.

Differences are R1 (searcher_residual_off) minus R0 (full_prrac). CSV missing
values are NA; JSON missing values are null. No runtime/checkpoint imports.
Native evaluation manifests omit checkpoint/max_steps: explicit episode fields
are required in that case, with provenance recorded, never inferred from names.
Found-state fields require found-time/pre-found semantics, not final-state data.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys


VARIANTS = ("full_prrac", "searcher_residual_off")
TRANSITIONS = {
    (True, False): "full_success_searcher_off_fail",
    (False, True): "full_fail_searcher_off_success",
    (True, True): "both_success",
    (False, False): "both_fail",
}
OPTIONAL = ("search_continuity_episode.csv", "search_collision_recovery_episode.csv",
            "search_collision_recovery_summary.csv")
MISSING = {"", "na", "n/a", "null", "none", "nan"}
FIELDS = {
    "outcome_groups.csv": "scenario_id full_found searcher_off_found full_success searcher_off_success transition_type",
    "success_transition_cases.csv": "scenario_id transition_type full_found_step searcher_off_found_step found_step_difference full_success searcher_off_success full_final_distance searcher_off_final_distance",
    "found_state_comparison.csv": "scenario_id variant found_step remaining_steps searcher_target_distance executor_target_distance executor_wait_distance searcher_distance_travelled known_map_fraction_gain",
    "collision_comparison.csv": "scenario_id variant collision_episode collision_count max_collision_streak recovery_entry_count route_refresh_attempt_count egress_attempt_count",
    "recovery_comparison.csv": "scenario_id variant search_recovery_entry_count route_refresh_success_count egress_success_count mean_recovery_duration",
    "searcher_trajectory_comparison.csv": "scenario_id variant agent_id distance_travelled waypoint_switch_count guidance_change_count",
    "found_execution_transition_comparison.csv": "scenario_id variant found contact success found_step remaining_steps executor_distance_at_found executor_distance_at_handoff handoff_delay found_to_first_contact_steps found_to_success_steps final_executor_target_distance",
    "searcher_residual_diagnostics.csv": "scenario_id variant transition_type raw_residual_norm_mean_pre_found applied_residual_norm_mean_pre_found residual_contribution_ratio_mean_pre_found negative_alignment_rate_pre_found waypoint_switch_count_pre_found route_active_rate_pre_found",
}
ALIASES = {
    "found_step": ("found_step",),
    "final_distance": ("executor_final_distance_to_target", "final_distance"),
    "searcher_target_distance": ("searcher_target_distance_at_found", "searcher_distance_to_target_at_found"),
    "executor_target_distance": ("executor_target_distance_at_found", "executor_distance_to_target_at_found"),
    "executor_wait_distance": ("executor_wait_distance_at_found", "executor_distance_to_wait_at_found"),
    "searcher_distance_travelled": ("searcher_distance_travelled_at_found", "searcher_distance_travelled_pre_found"),
    "known_map_fraction_gain": ("known_map_fraction_gain_at_found", "map_known_fraction_gain_pre_found"),
    "collision_episode": ("searcher_collision_episode_pre_found", "collision_episode"),
    "collision_count": ("searcher_collision_count_pre_found", "searcher_collision_count_pre_found_total", "collision_count"),
    "max_collision_streak": ("searcher_collision_max_streak_pre_found", "max_collision_streak"),
    "recovery_entry_count": ("search_recovery_entry_count", "recovery_entry_count"),
    "search_recovery_entry_count": ("search_recovery_entry_count", "recovery_entry_count"),
    "route_refresh_attempt_count": ("route_refresh_attempt_count",),
    "egress_attempt_count": ("egress_attempt_count",),
    "route_refresh_success_count": ("route_refresh_success_count",),
    "egress_success_count": ("egress_success_count",),
    "mean_recovery_duration": ("recovery_duration_mean", "mean_recovery_duration"),
    # contact_episode is the existing evaluator's explicit episode contact flag.
    "contact": ("contact", "reached_contact", "contact_reached", "has_contact", "contact_episode"),
    "executor_distance_at_found": ("executor_distance_to_target_at_found", "executor_target_distance_at_found", "executor_distance_at_found"),
    "executor_distance_at_handoff": ("executor_distance_at_handoff", "executor_distance_to_target_at_handoff"),
    "handoff_delay": ("handoff_delay",),
    "found_to_first_contact_steps": ("found_to_first_contact_steps",),
    "found_to_success_steps": ("found_to_success_steps",),
    "raw_residual_norm_mean_pre_found": ("searcher_raw_residual_norm_mean_pre_found",),
    "applied_residual_norm_mean_pre_found": ("searcher_applied_residual_norm_mean_pre_found",),
    "residual_contribution_ratio_mean_pre_found": ("searcher_residual_contribution_ratio_mean_pre_found",),
    "negative_alignment_rate_pre_found": ("searcher_residual_negative_alignment_rate_pre_found",),
    "waypoint_switch_count_pre_found": ("searcher_waypoint_switch_count_pre_found",),
    "route_active_rate_pre_found": ("searcher_route_active_rate_pre_found",),
}
EXECUTION_METRICS = ("executor_distance_at_found", "executor_distance_at_handoff", "handoff_delay",
                     "found_to_first_contact_steps", "found_to_success_steps")
RESIDUAL_METRICS = tuple(FIELDS["searcher_residual_diagnostics.csv"].split()[3:])
CHAIN_METRICS = ("found_step", "remaining_steps", *EXECUTION_METRICS, "collision_count", "recovery_entry_count")
CORE_FIELDS = ("found_step", *EXECUTION_METRICS, *RESIDUAL_METRICS, "contact")


def present(value):
    return value is not None and str(value).strip().lower() not in MISSING


def number(value):
    if not present(value):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return result


def integer(value):
    result = number(value)
    if result is None or result != int(result):
        raise ValueError(f"required integer missing/invalid: {value!r}")
    return int(result)


def boolean(value):
    token = str(value).strip().lower()
    if token in ("true", "1", "1.0"):
        return True
    if token in ("false", "0", "0.0"):
        return False
    raise ValueError(f"required boolean missing/invalid: {value!r}")


def read_csv(path):
    if not path.is_file() or not path.stat().st_size:
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"duplicate CSV headers: {path}")
        rows = []
        for row in reader:
            if None in row:
                raise ValueError(f"malformed CSV row: {path}")
            if any(present(v) for v in row.values()):
                rows.append(row)
        return rows


def unique_rows(rows, label):
    result = {}
    for row in rows:
        key = str(row.get("scenario_id", "")).strip()
        if not key or key in result:
            raise ValueError(f"missing/duplicate scenario_id in {label}: {key!r}")
        result[key] = row
    return result


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_input(path, variant):
    path = Path(path).resolve(strict=True)
    manifest = json.loads((path / "evaluation_manifest.json").read_text(encoding="utf-8-sig"))
    summary = json.loads((path / "evaluation_summary.json").read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or not isinstance(summary, dict):
        raise ValueError("manifest and summary must be JSON objects")
    episodes = unique_rows(read_csv(path / "episode_evaluation.csv"), path)
    if not episodes:
        raise ValueError(f"empty/missing episode_evaluation.csv: {path}")
    optional = {name: read_csv(path / name) for name in OPTIONAL}
    teams, agents = {}, {}
    for name, rows in optional.items():
        if name.endswith("summary.csv"):
            continue  # Aggregate summaries must never be broadcast to scenarios.
        team_rows, agent_rows = [], {}
        for row in rows:
            sid = str(row.get("scenario_id", "")).strip()
            if sid not in episodes:
                raise ValueError(f"unknown scenario_id in {name}: {sid!r}")
            if present(row.get("agent_id")):
                agent = integer(row["agent_id"])
                if agent not in (0, 1, 2) or (sid, agent) in agent_rows:
                    raise ValueError(f"invalid/duplicate Searcher key in {name}: {(sid, agent)}")
                agent_rows[sid, agent] = row
            else:
                team_rows.append(row)
        teams[name] = unique_rows(team_rows, name)
        agents[name] = agent_rows
    identity, sources = {}, {}
    for key, converter in (("scenario_seed", integer), ("manifest_sha256", str),
                           ("checkpoint", str), ("max_steps", integer)):
        values = []
        if present(manifest.get(key)):
            values.append(("evaluation_manifest.json", converter(manifest[key])))
        elif key in ("scenario_seed", "manifest_sha256"):
            raise ValueError(f"evaluation_manifest.json missing {key}")
        if key in ("checkpoint", "max_steps"):
            for sid, row in episodes.items():
                if present(row.get(key)):
                    values.append((f"episode_evaluation.csv:{sid}", converter(row[key])))
                elif not present(manifest.get(key)):
                    raise ValueError(f"missing {key} in manifest and episode {sid}")
        if not values or len({v for _, v in values}) != 1:
            raise ValueError(f"missing/conflicting {key} in {path}")
        identity[key] = values[0][1]
        sources[key] = sorted({source.split(":", 1)[0] for source, _ in values})
    if identity["max_steps"] <= 0:
        raise ValueError("max_steps must be positive")
    scenario_rows = unique_rows(manifest.get("scenarios", []), "manifest scenarios")
    if "scenarios" in manifest and set(scenario_rows) != set(episodes):
        raise ValueError("scenario_id set mismatch between manifest and episodes")
    seeds = {}
    for sid, row in episodes.items():
        seed_values = [integer(r["scenario_seed"]) for r in (row, scenario_rows.get(sid, {}))
                       if present(r.get("scenario_seed"))]
        if not seed_values or len(set(seed_values)) != 1:
            raise ValueError(f"missing/conflicting scenario_seed for {sid}")
        seeds[sid] = seed_values[0]
        found, success = boolean(row.get("found")), boolean(row.get("success"))
        if success and not found:
            raise ValueError(f"success without found: {sid}")
    # Reject mixed modes, runtime/C2/checkpoint identities, and stale sidecars.
    provenance_keys = ("checkpoint", "max_steps", "manifest_sha256", "evaluation_mode",
                       "execution_variant", "search_recovery_variant", "runtime_integration_mode",
                       "checkpoint_config_hash", "checkpoint_runtime_revision", "evaluation_runtime_revision")
    invariants = dict(identity, evaluation_mode=variant)
    for key in provenance_keys:
        values = {str(row[key]).strip() for row in episodes.values() if present(row.get(key))}
        if key not in invariants and values:
            if len(values) != 1:
                raise ValueError(f"mixed {key} in episode file")
            invariants[key] = next(iter(values))
    all_rows = [*episodes.values(), *(r for rows in optional.values() for r in rows), summary,
                *summary.get("summary", [])]
    for row in all_rows:
        for key in provenance_keys:
            if key in invariants and present(row.get(key)):
                value = integer(row[key]) if key == "max_steps" else str(row[key]).strip()
                if value != invariants[key]:
                    raise ValueError(f"conflicting {key} in input rows: {path}")
        sid = str(row.get("scenario_id", "")).strip()
        if sid and present(row.get("scenario_seed")) and integer(row["scenario_seed"]) != seeds.get(sid):
            raise ValueError(f"sidecar scenario_seed mismatch: {sid}")
    files = ("episode_evaluation.csv", "evaluation_manifest.json", "evaluation_summary.json", *OPTIONAL)
    return dict(path=path, variant=variant, episodes=episodes, teams=teams, agents=agents,
                identity=identity, identity_sources=sources, seeds=seeds, invariants=invariants,
                files={name: file_hash(path/name) if (path/name).is_file() else None for name in files},
                optional_row_counts={name: len(rows) for name, rows in optional.items()})


def merged_row(data, sid):
    row = dict(data["episodes"][sid])
    for rows in data["teams"].values():
        for key, value in rows.get(sid, {}).items():
            if present(value):
                if present(row.get(key)) and row[key] != value:
                    try:
                        equal = number(row[key]) == number(value)
                    except ValueError:
                        equal = str(row[key]).lower() == str(value).lower()
                    if not equal:
                        raise ValueError(f"conflicting duplicate field {key} for {sid}")
                row[key] = value
    return row


def metric(row, name):
    for key in ALIASES[name]:
        if present(row.get(key)):
            value = boolean(row[key]) if name in ("collision_episode", "contact") else number(row[key])
            if name not in ("known_map_fraction_gain", "collision_episode", "contact") and value < 0:
                raise ValueError(f"negative {key}: {value}")
            if (name.endswith("count") or name in ("max_collision_streak", "waypoint_switch_count_pre_found")) and value != int(value):
                raise ValueError(f"noninteger count {key}: {value}")
            if name in ("negative_alignment_rate_pre_found", "route_active_rate_pre_found") and value > 1:
                raise ValueError(f"rate outside [0, 1]: {key}={value}")
            if name == "contact" and any(boolean(row[alias]) != value for alias in ALIASES[name] if present(row.get(alias))):
                raise ValueError("conflicting explicit contact aliases")
            return value
    return None


def difference(full, off):
    return None if full is None or off is None else off - full


def paired_stats(values):
    valid = [v for v in values if v is not None]
    return dict(mean=statistics.mean(valid) if valid else None, valid_pairs=len(valid),
                missing_pairs=len(values)-len(valid), ties=sum(v == 0 for v in valid))


def descriptive_stats(values):
    valid = [v for v in values if v is not None]
    return dict(mean=statistics.mean(valid) if valid else None, valid=len(valid), missing=len(values)-len(valid))


def exact_mcnemar(b, c):
    """Two-sided exact binomial McNemar test on discordant scenario pairs."""
    b, c = integer(b), integer(c)
    if min(b, c) < 0:
        raise ValueError("discordant counts must be nonnegative")
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c)+1)) / (1 << n))


def paired_binary_stats(pairs):
    """Missing flags are unknown, never negative. Counts use complete pairs."""
    valid = [(r0, r1) for r0, r1 in pairs if r0 is not None and r1 is not None]
    n = len(valid)
    counts = {"both_positive": sum(r0 and r1 for r0, r1 in valid),
              "r0_only": sum(r0 and not r1 for r0, r1 in valid),
              "r1_only": sum(not r0 and r1 for r0, r1 in valid),
              "both_negative": sum(not r0 and not r1 for r0, r1 in valid)}
    b, c = counts["r0_only"], counts["r1_only"]
    return dict(available=bool(n), unavailable_reason=None if n else "no_complete_observed_pairs",
                total_scenarios=len(pairs), valid_pairs=n, missing_pairs=len(pairs)-n,
                r0_valid_count=sum(r0 is not None for r0, _ in pairs),
                r1_valid_count=sum(r1 is not None for _, r1 in pairs),
                counts_scope="complete scenario pairs only; percentage-point denominator is valid_pairs",
                r0_positive=sum(r0 for r0, _ in valid) if n else None,
                r1_positive=sum(r1 for _, r1 in valid) if n else None,
                **{key: value if n else None for key, value in counts.items()},
                absolute_count_change=c-b if n else None, percentage_point_change=100*(c-b)/n if n else None,
                discordant_total=b+c if n else None, exact_mcnemar_p_two_sided=exact_mcnemar(b, c) if n else None)


def trajectory_rows(data, sid, row):
    long = data["agents"]["search_continuity_episode.csv"]
    explicit = [(agent, values) for (scene, agent), values in sorted(long.items()) if scene == sid]
    if explicit:
        return [dict(scenario_id=sid, variant=data["variant"], agent_id=agent,
                     **{name: number(values.get(name)) for name in
                        ("distance_travelled", "waypoint_switch_count", "guidance_change_count")})
                for agent, values in explicit]
    result = []
    for agent in range(3):
        values = {}
        for name in ("distance_travelled", "waypoint_switch_count", "guidance_change_count"):
            keys = (f"searcher_{name}_pre_found_agent_{agent}", f"{name}_agent_{agent}")
            values[name] = next((number(row[k]) for k in keys if present(row.get(k))), None)
        if any(v is not None for v in values.values()):
            result.append(dict(scenario_id=sid, variant=data["variant"], agent_id=agent, **values))
    return result or [dict(scenario_id=sid, variant=data["variant"], agent_id=None)]


def analyze(full_output, searcher_off_output, output_dir="searcher_residual_analysis"):
    data = [load_input(path, variant) for path, variant in zip((full_output, searcher_off_output), VARIANTS)]
    full, off = data
    if set(full["episodes"]) != set(off["episodes"]):
        raise ValueError("scenario_id sets differ between experiments")
    if full["identity"] != off["identity"] or full["seeds"] != off["seeds"]:
        raise ValueError("manifest identity mismatch (scenario_seed/manifest_sha256/checkpoint/max_steps)")
    for key in set(full["invariants"]) | set(off["invariants"]):
        if key != "evaluation_mode" and full["invariants"].get(key) != off["invariants"].get(key):
            raise ValueError(f"paired experiment provenance mismatch: {key}")
    output = Path(output_dir).resolve()
    if any(output == d["path"] or d["path"] in output.parents or output in d["path"].parents for d in data):
        raise ValueError("output must be separate from the read-only input directories")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing nonempty/existing output: {output}")
    tables = {name: [] for name in FIELDS}
    counts = {name: 0 for name in TRANSITIONS.values()}
    differences = {name: [] for name in ("found_step_difference", "executor_distance_difference", "collision_difference", "recovery_difference",
                                        *(f"{name}_difference" for name in EXECUTION_METRICS))}
    by_transition = {group: {name: [] for name in differences} for group in counts}
    sources = {variant: {} for variant in VARIANTS}
    core_values = {variant: {} for variant in VARIANTS}
    chains = []
    binary_pairs = {name: [] for name in ("found", "contact", "success")}
    for sid in sorted(full["episodes"]):
        pair = [merged_row(d, sid) for d in data]
        found = [boolean(row["found"]) for row in pair]
        success = [boolean(row["success"]) for row in pair]
        transition = TRANSITIONS[tuple(success)]
        counts[transition] += 1
        tables["outcome_groups.csv"].append(dict(scenario_id=sid, full_found=found[0], searcher_off_found=found[1],
            full_success=success[0], searcher_off_success=success[1], transition_type=transition))
        steps, finals, collisions, recoveries = [], [], [], []
        for index, (d, row) in enumerate(zip(data, pair)):
            base = dict(scenario_id=sid, variant=d["variant"])
            step = metric(row, "found_step") if found[index] else None
            if step is not None and (step != int(step) or not 0 <= step <= d["identity"]["max_steps"]):
                raise ValueError(f"invalid found_step: {sid}")
            steps.append(step)
            finals.append(metric(row, "final_distance"))
            remaining = None if step is None else d["identity"]["max_steps"]-step
            if remaining is not None and present(row.get("remaining_steps_after_found")) and number(row["remaining_steps_after_found"]) != remaining:
                raise ValueError(f"remaining_steps_after_found mismatch: {sid}")
            state = dict(base, found_step=step, remaining_steps=remaining)
            for name in FIELDS["found_state_comparison.csv"].split()[4:]:
                state[name] = metric(row, name) if found[index] else None
            tables["found_state_comparison.csv"].append(state)
            collision = dict(base, **{name: metric(row, name) for name in FIELDS["collision_comparison.csv"].split()[2:]})
            # When pre-found counts exist, do not combine them with all-agent whole-episode flags.
            if any(present(row.get(k)) for k in ALIASES["collision_count"][:2]):
                collision["collision_episode"] = boolean(row["searcher_collision_episode_pre_found"]) if present(row.get("searcher_collision_episode_pre_found")) else bool(collision["collision_count"])
            tables["collision_comparison.csv"].append(collision)
            collisions.append(collision["collision_count"])
            recovery = dict(base, **{name: metric(row, name) for name in FIELDS["recovery_comparison.csv"].split()[2:]})
            tables["recovery_comparison.csv"].append(recovery)
            recoveries.append(recovery["search_recovery_entry_count"])
            contact = metric(row, "contact")
            chain = dict(base, found=found[index], contact=contact, success=success[index], found_step=step,
                         remaining_steps=remaining, final_executor_target_distance=finals[-1])
            for name in EXECUTION_METRICS:
                eligible = found[index]
                if name == "found_to_success_steps":
                    eligible = eligible and success[index]
                elif name == "found_to_first_contact_steps":
                    eligible = eligible and contact is not False
                chain[name] = metric(row, name) if eligible else None
            tables["found_execution_transition_comparison.csv"].append(chain)
            residual = dict(base, transition_type=transition, **{name: metric(row, name) for name in RESIDUAL_METRICS})
            tables["searcher_residual_diagnostics.csv"].append(residual)
            chains.append(dict(chain, transition_type=transition, collision_count=collision["collision_count"],
                               recovery_entry_count=recovery["search_recovery_entry_count"],
                               collision_scope="pre_found" if any(present(row.get(k)) for k in ALIASES["collision_count"][:2]) else "unspecified"))
            core_values[d["variant"]][sid] = {**chain, **residual}
            tables["searcher_trajectory_comparison.csv"].extend(trajectory_rows(d, sid, row))
            sources[d["variant"]][sid] = {name: next((k for k in keys if present(row.get(k))), None) for name, keys in ALIASES.items()}
        deltas = (difference(*steps), difference(*finals), difference(*collisions), difference(*recoveries))
        collision_scopes = [any(present(row.get(k)) for k in ALIASES["collision_count"][:2]) for row in pair]
        if collision_scopes[0] != collision_scopes[1]:
            deltas = (*deltas[:2], None, deltas[3])
        core_pair = [core_values[variant][sid] for variant in VARIANTS]
        deltas = (*deltas, *(difference(*(record[name] for record in core_pair)) for name in EXECUTION_METRICS))
        for name in binary_pairs:
            binary_pairs[name].append(tuple(record[name] for record in core_pair))
        for name, delta in zip(differences, deltas):
            differences[name].append(delta)
            by_transition[transition][name].append(delta)
        if success[0] != success[1]:
            tables["success_transition_cases.csv"].append(dict(scenario_id=sid, transition_type=transition,
                full_found_step=steps[0], searcher_off_found_step=steps[1], found_step_difference=deltas[0],
                full_success=success[0], searcher_off_success=success[1], full_final_distance=finals[0], searcher_off_final_distance=finals[1]))
    outcomes = tables["outcome_groups.csv"]
    n = len(outcomes)
    success_counts = [sum(row[key] for row in outcomes) for key in ("full_success", "searcher_off_success")]
    found_counts = [sum(row[key] for row in outcomes) for key in ("full_found", "searcher_off_found")]
    def effect(values):
        change = values[1]-values[0]
        return dict(absolute_count_change=change, rate_change=change/n, percentage_point_change=100*change/n,
                    relative_change=None if values[0] == 0 else change/values[0],
                    relative_change_reason="baseline_count_zero" if values[0] == 0 else None)
    stats = {key: paired_stats(values) for key, values in differences.items()}
    residual_groups, execution_groups = {}, {}
    for group in counts:
        r0_rows = [r for r in tables["searcher_residual_diagnostics.csv"] if r["variant"] == VARIANTS[0] and r["transition_type"] == group]
        residual_groups[group] = dict(count=len(r0_rows), **{name: descriptive_stats([r[name] for r in r0_rows]) for name in RESIDUAL_METRICS})
        execution_groups[group] = dict(count=counts[group])
        for variant in VARIANTS:
            group_rows = [r for r in chains if r["variant"] == variant and r["transition_type"] == group]
            values = {name: descriptive_stats([r[name] for r in group_rows]) for name in CHAIN_METRICS}
            scopes = sorted({r["collision_scope"] for r in group_rows if r["collision_count"] is not None})
            values["collision_count_by_scope"] = {scope: descriptive_stats([r["collision_count"] for r in group_rows if r["collision_scope"] == scope]) for scope in scopes}
            if len(scopes) > 1:
                values["collision_count"].update(mean=None, reason="mixed_collision_scopes; see collision_count_by_scope")
            execution_groups[group][variant] = values
    significance = {name: paired_binary_stats(pairs) for name, pairs in binary_pairs.items()}
    significance["definitions"] = dict(R0=VARIANTS[0], R1=VARIANTS[1], difference="R1 - R0",
        method="two-sided exact McNemar/binomial; math.comb; unadjusted diagnostic tests",
        interpretation="Diagnostic paired associations only; not proof of a causal effect. Missing contact is unknown.")
    coverage = {name: {**{variant: sum(values[name] is not None for values in core_values[variant].values()) for variant in VARIANTS},
                      "paired_valid": sum(all(core_values[variant][sid][name] is not None for variant in VARIANTS) for sid in sorted(full["episodes"]))}
                for name in CORE_FIELDS}
    summary = dict(total_scenarios=n, full_success_count=success_counts[0], searcher_off_success_count=success_counts[1],
        full_found_count=found_counts[0], searcher_off_found_count=found_counts[1],
        full_success_rate=success_counts[0]/n, searcher_off_success_rate=success_counts[1]/n,
        success_gain=success_counts[1]-success_counts[0], found_gain=found_counts[1]-found_counts[0],
        searcher_residual_effect=effect(success_counts), found_effect=effect(found_counts), transition_counts=counts,
        paired_transitions=dict(R0_success_R1_fail=counts[TRANSITIONS[True, False]], R0_fail_R1_success=counts[TRANSITIONS[False, True]]),
        mean_metrics={key: value["mean"] for key, value in stats.items()}, paired_metric_coverage=stats,
        full_prrac_residual_by_transition=residual_groups, execution_chain_by_transition=execution_groups,
        paired_metrics_by_transition={group: {name: paired_stats(values) for name, values in metrics.items()}
                                      for group, metrics in by_transition.items()},
        definitions=dict(R0=VARIANTS[0], R1=VARIANTS[1], difference="R1 - R0", success_gain="success count difference",
                         found_gain="found count difference", executor_distance_difference="deprecated / final-state only: final executor-target distance, not found-time distance",
                         execution_chain_by_transition="scenario-level descriptive means separately for R0 and R1; missing/unreached stages excluded",
                         execution_paired_metrics="R1 - R0; both values required; only explicit at_found/at_handoff distances; timings require found, success timing also requires success",
                         contact="explicit flags only (including evaluator contact_episode); not inferred from success or timing",
                         recovery_difference="search recovery entry count", means="complete pairs only; found step requires both found"),
        limitations=["Descriptive paired associations, not proof of a causal failure mechanism.",
                     "Found-state distances require explicitly at_found fields; final/minimum distances are never substituted.",
                     "Pre-found travel/map gain describe the found state only when found=True; otherwise NA.",
                     "Aggregate recovery summaries are not assigned to individual scenarios; missing values are not zeros.",
                     "Collision/recovery counts depend on search exposure duration; an association alone does not explain causation.",
                     "Collision scope follows source columns; mixed pre-found/unspecified scopes are excluded from paired means.",
                     "A Success count change (such as 32 to 37) is descriptive, not a significant improvement without evidence from an exact paired test; diagnostic p-values are unadjusted.",
                     "Residual metrics are episode-level/pre-found aggregates, not proof that any particular action produced an outcome.",
                     "Outcome-transition stratification is post-hoc mechanism diagnosis, not an independent confirmatory test.",
                     "Small A/B transition groups support descriptive means only; consult valid/missing counts."],
        missing_values={name: {key: sum(row.get(key) is None for row in rows) for key in FIELDS[name].split()}
                        for name, rows in tables.items()})
    try:
        repo = Path(__file__).resolve().parents[1]
        # Trust only this script's repository for this read, without editing Git config.
        commit = subprocess.run(["git", "-c", f"safe.directory={repo.as_posix()}", "rev-parse", "HEAD"], cwd=repo,
                                capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        git_error = None
    except (OSError, subprocess.SubprocessError) as exc:
        commit, git_error = None, str(exc)
    manifest = dict(schema="searcher_residual_analysis.v1", timestamp=datetime.now(timezone.utc).isoformat(),
                    python_version=platform.python_version(), git_commit=commit, git_error=git_error,
                    script_sha256=file_hash(Path(__file__)), input_paths={d["variant"]: str(d["path"]) for d in data},
                    verified_identity=full["identity"], scenario_seeds=full["seeds"], field_aliases=ALIASES, field_sources=sources,
                    field_coverage=coverage, field_coverage_definition="valid eligible scenario values, not agents; non-found stages excluded even if a source column exists; paired_valid requires both variants",
                    provenance={d["variant"]: {key: d[key] for key in ("identity_sources", "files", "optional_row_counts")} for d in data})
    for d in data:
        for name, expected in d["files"].items():
            path = d["path"]/name
            if (file_hash(path) if path.is_file() else None) != expected:
                raise ValueError(f"input changed during analysis: {path}")
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        with (output/name).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS[name].split())
            writer.writeheader()
            writer.writerows({key: "NA" if row.get(key) is None else row[key] for key in FIELDS[name].split()} for row in rows)
    for name, value in (("searcher_residual_effect_summary.json", summary), ("analysis_manifest.json", manifest),
                        ("paired_binary_significance.json", significance)):
        with (output/name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-output", type=Path, required=True)
    parser.add_argument("--searcher-off-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("searcher_residual_analysis"))
    args = parser.parse_args(argv)
    try:
        result = analyze(args.full_output, args.searcher_off_output, args.output_dir)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in ("total_scenarios", "paired_transitions", "searcher_residual_effect")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
