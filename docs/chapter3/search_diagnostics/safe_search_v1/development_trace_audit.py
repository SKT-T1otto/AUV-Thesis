"""Read-only window audit after all 100 fixed-development arms have completed.

Run from the repository root with ``python -B -m
docs.chapter3.search_diagnostics.safe_search_v1.development_trace_audit``.
Observation-footprint renewal is a diagnostic proxy, never detection probability.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from chapter3_bser.experiments.safe_search_v1.analyze_development import (
    PLAN, ROOT, outcome, validate_rows,
)
from chapter3_bser.experiments.safe_search_v1.run_development import collect_completed
from chapter3_bser.experiments.safe_search_v1.run_paired import VARIANTS, file_hash, read_json

COVERAGE_FIELDS = (
    "new_cell_observations", "aged_revisit_cell_observations",
    "effective_observation_steps", "observation_calls", "target_update_call_steps",
    "observation_update_steps", "repeated_cell_observations",
)
MOTION_FIELDS = ("searcher_motion_distance", "searcher_stall_agent_steps",
                 "searcher_assignment_id_switches", "searcher_semantic_waypoint_switches")


def windows():
    return [dict(first_step=1 + 100 * i, last_step=100 + 100 * i,
                 pre_found_exposure_steps=0, coverage_available=True,
                 coverage_unavailable_steps=0,
                 **{name: 0 for name in COVERAGE_FIELDS + MOTION_FIELDS}) for i in range(4)]


def trailing_zero_steps(rows, fields):
    """Exact trailing physical search steps; unknown data never becomes a zero."""
    count = 0
    for row in reversed(rows):
        coverage = row.get("search_coverage") or {}
        if coverage.get("available") is not True or any(coverage.get(f) is None for f in fields):
            return None
        if any(coverage[f] > 0 for f in fields):
            return count
        count += 1
    return count


def audit_trace(rows, terminal):
    """Extract four fixed physical-step windows; Found transition is included."""
    result = windows()
    search_rows = []
    n = 0
    for row in rows:
        n += 1
        if row["step_before"] != n - 1 or row["step_after"] != n or n > 400:
            raise ValueError("trace must have contiguous physical steps in 1..400")
        if row["before"]["step"] != n - 1 or row["after"]["step"] != n:
            raise ValueError("snapshot and trace timestamps disagree")
        is_search = not row["before"]["found"]
        if row.get("search_transition") is not is_search:
            raise ValueError("search transition must describe the pre-step Found state")
        expected_search = terminal["found_step"] is None or n <= terminal["found_step"]
        if is_search != expected_search:
            raise ValueError("trace search exposure disagrees with terminal Found step")
        if not is_search:
            continue
        search_rows.append(row)
        window = result[(n - 1) // 100]
        window["pre_found_exposure_steps"] += 1
        coverage = row.get("search_coverage") or {}
        if coverage and (coverage.get("step") != n or coverage.get("pre_found_exposure_steps") != 1):
            raise ValueError("coverage and physical search timestamps disagree")
        measured = coverage.get("available") is True
        if measured:
            for name in COVERAGE_FIELDS:
                value = coverage.get(name)
                if type(value) is not int or value < 0:
                    raise ValueError("invalid available coverage field: " + name)
                window[name] += value
        else:
            window["coverage_available"] = False
            window["coverage_unavailable_steps"] += 1
        before = {agent["agent_id"]: agent for agent in row["before"]["agents"]}
        after = {agent["agent_id"]: agent for agent in row["after"]["agents"]}
        metrics = {agent["agent_id"]: agent for agent in row["agent_metrics"]}
        if not {0, 1, 2} <= before.keys() & after.keys() & metrics.keys():
            raise ValueError("all three searcher records are required")
        for agent in (0, 1, 2):
            start, end, measurement = before[agent], after[agent], metrics[agent]
            displacement = math.dist(start["position"], end["position"])
            if not math.isfinite(displacement):
                raise ValueError("nonfinite searcher displacement")
            if not math.isclose(displacement, measurement["displacement"], abs_tol=1e-10, rel_tol=1e-10):
                raise ValueError("recorded displacement differs from public positions")
            if type(measurement["motion_stall_proxy"]) is not bool:
                raise ValueError("motion-stall proxy must be boolean")
            window["searcher_motion_distance"] += displacement
            window["searcher_stall_agent_steps"] += measurement["motion_stall_proxy"]
            window["searcher_assignment_id_switches"] += start["assignment_id"] != end["assignment_id"]
            window["searcher_semantic_waypoint_switches"] += start["semantic_waypoint"] != end["semantic_waypoint"]
    if n != terminal["physical_steps"]:
        raise ValueError("trace length differs from terminal summary")
    if len(search_rows) != terminal["pre_found_exposure_steps"]:
        raise ValueError("trace exposure differs from terminal summary")
    if sum(w["searcher_stall_agent_steps"] for w in result) != terminal["searcher_motion_stall_proxy_agent_steps"]:
        raise ValueError("trace stall count differs from terminal summary")
    terminal_coverage = terminal.get("search_coverage") or {}
    if terminal_coverage.get("available") is True:
        if not all(w["coverage_available"] for w in result):
            raise ValueError("terminal coverage availability contradicts trace")
        for name in COVERAGE_FIELDS:
            if sum(w[name] for w in result) != terminal_coverage[name]:
                raise ValueError("trace coverage differs from terminal summary: " + name)
    for window in result:
        if not window["coverage_available"]:
            window.update({name: None for name in COVERAGE_FIELDS})
        exposure = window["pre_found_exposure_steps"]
        window["searcher_stall_agent_fraction"] = (
            window["searcher_stall_agent_steps"] / (3 * exposure) if exposure else None)
        window["effective_observation_fraction"] = (
            window["effective_observation_steps"] / exposure
            if exposure and window["coverage_available"] else None)
    label = outcome(terminal)
    tail = result[-1]
    no_new_aged_final100 = (tail["new_cell_observations"] == 0
                           and tail["aged_revisit_cell_observations"] == 0)
    return dict(original_episode_index=terminal["original_episode_index"], variant=terminal["variant"],
        scenario_id=terminal["scenario_id"], outcome=label, found_step=terminal["found_step"],
        physical_steps=n, pre_found_exposure_steps=len(search_rows), windows=result,
        trailing_search_steps_without_target_update_call=trailing_zero_steps(search_rows, ["target_update_call_steps"]),
        trailing_search_steps_without_nonempty_footprint_update=trailing_zero_steps(search_rows, ["observation_update_steps"]),
        trailing_search_steps_without_new_or_aged_evidence=trailing_zero_steps(search_rows,
            ["new_cell_observations", "aged_revisit_cell_observations"]),
        timeout_final100_without_new_or_aged_evidence=(
            no_new_aged_final100 if label == "no_found_timeout" and tail["coverage_available"]
            and tail["pre_found_exposure_steps"] == 100 else None))


def aggregate(episodes):
    pooled = []
    for index in range(4):
        source = [episode["windows"][index] for episode in episodes]
        exposure = sum(w["pre_found_exposure_steps"] for w in source)
        available = bool(source) and all(w["coverage_available"] for w in source)
        combined = dict(first_step=1 + 100 * index, last_step=100 + 100 * index,
                        episode_count=len(episodes), episodes_with_search_exposure=sum(
                            w["pre_found_exposure_steps"] > 0 for w in source),
                        pre_found_exposure_steps=exposure, coverage_available=available,
                        coverage_unavailable_steps=sum(w["coverage_unavailable_steps"] for w in source))
        combined.update({name: sum(w[name] for w in source) for name in MOTION_FIELDS})
        combined.update({name: sum(w[name] for w in source) if available else None
                         for name in COVERAGE_FIELDS})
        combined["searcher_stall_agent_fraction"] = (
            combined["searcher_stall_agent_steps"] / (3 * exposure) if exposure else None)
        combined["effective_observation_fraction"] = (
            combined["effective_observation_steps"] / exposure if available and exposure else None)
        pooled.append(combined)
    tails = [episode["timeout_final100_without_new_or_aged_evidence"] for episode in episodes
             if episode["outcome"] == "no_found_timeout"]
    return dict(episode_count=len(episodes), original_episode_indices=[e["original_episode_index"] for e in episodes],
                windows=pooled, no_found_timeout_count=len(tails),
                timeout_final100_coverage_available_count=sum(value is not None for value in tails),
                timeout_final100_without_new_or_aged_evidence_count=sum(value is True for value in tails),
                trailing_steps_by_episode=[dict(original_episode_index=e["original_episode_index"],
                    **{name: e[name] for name in (
                        "trailing_search_steps_without_target_update_call",
                        "trailing_search_steps_without_nonempty_footprint_update",
                        "trailing_search_steps_without_new_or_aged_evidence")}) for e in episodes])


def summarize_audits(episodes):
    variants = {}
    for variant in VARIANTS:
        arm = [e for e in episodes if e["variant"] == variant]
        groups = defaultdict(list)
        for episode in arm:
            groups[episode["outcome"]].append(episode)
        variants[variant] = dict(all_outcomes=aggregate(arm),
            by_outcome={label: aggregate(group) for label, group in sorted(groups.items())})
    indexed = {(e["original_episode_index"], e["variant"]): e for e in episodes}
    same_timeout = sorted({i for i, v in indexed if v == "V0"
        and indexed[i, "V0"]["outcome"] == "no_found_timeout"
        and indexed[i, "V4"]["outcome"] == "no_found_timeout"})
    paired = {v: aggregate([indexed[i, v] for i in same_timeout]) for v in ("V0", "V4")}
    deltas = []
    for zero, four in zip(paired["V0"]["windows"], paired["V4"]["windows"]):
        deltas.append(dict(first_step=zero["first_step"], last_step=zero["last_step"],
            **{name: four[name] - zero[name] if four[name] is not None and zero[name] is not None else None
               for name in ("pre_found_exposure_steps",) + COVERAGE_FIELDS + MOTION_FIELDS}))
    return dict(schema="ch3.safe_search.development_window_audit.v1", episodes=episodes, variants=variants,
        paired_v0_v4_both_no_found_timeout=dict(n_pairs=len(same_timeout),
            original_episode_indices=same_timeout, variants=paired, v4_minus_v0_window_totals=deltas),
        definitions=dict(windows="Fixed physical transitions 1–100, 101–200, 201–300, 301–400; only pre-Found transitions, including the Found transition.",
            new="Actual newly stamped target-evidence cells, excluding initialization.",
            aged="Actual repeat observations after configured target-belief revisit half-life.",
            effective="Step containing new or aged evidence; proxy, not detection probability.",
            actual_updates="observation_calls counts original target update calls; target_update_call_steps counts call-bearing steps; observation_update_steps counts steps with nonempty actual stamp footprints.",
            movement="Sum of public start-to-end distances over searchers 0,1,2; not net exploration or swept sensor volume.",
            switches="Per-searcher assignment-ID and exact semantic-waypoint changes, including the Found transition; an ID change is not necessarily a goal change.",
            tail="Consecutive physical search transitions ending at the last pre-Found transition; unknown coverage produces null, not zero.",
            timeout_final100="Only no-Found 400-step timeouts with all 100 final-window coverage records available are evaluated."),
        caveats=["Variants with earlier Found have less late-window exposure; use outcome strata and inspect exposure denominators.",
            "Outcome-stratified subsets differ between variants and do not establish causal effects.",
            "The shared V0/V4 no-Found-timeout subset is paired but selected using both outcomes; it diagnoses persistent failures and does not estimate overall Found improvement.",
            "Zero new/aged evidence is not zero observations; inspect repeated evidence, update cadence and movement separately.",
            "Evidence renewal, motion and survival proxies are not direct target detection probabilities."])


def reject_constant(value):
    raise ValueError("nonfinite JSON number: " + value)


def trace_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line, parse_constant=reject_constant)


def check_output(path, run_dir):
    path, run_dir = Path(path).resolve(), Path(run_dir).resolve()
    if path.exists():
        raise FileExistsError("output JSON must be new")
    if path == run_dir or run_dir in path.parents:
        raise ValueError("audit output must be outside the original run directory")
    for name in ("outputs", "3090结果", ".git", "core", "chapter3_bser", "configs", "tests", "tools", "scripts"):
        protected = ROOT / name
        if path == protected or protected in path.parents:
            raise ValueError("audit cannot write retained inputs or production source")
    return path


def analyze(run_dir):
    run_dir = Path(run_dir).resolve()
    identity, terminals = collect_completed(run_dir)
    plan = read_json(PLAN)
    validate_rows(terminals, plan)
    if file_hash(PLAN) != identity["sources_before"]["inventory"]["files"][PLAN.relative_to(ROOT).as_posix()]:
        raise ValueError("current plan bytes differ from reviewed experiment plan")
    hashes = {str(run_dir / name): file_hash(run_dir / name) for name in ("identity.json", "episodes.json")}
    episodes = []
    for terminal in terminals:
        index, variant = terminal["original_episode_index"], terminal["variant"]
        path = run_dir / f"scene_{index:04d}" / f"episode_{index:04d}" / variant / "step_trace.jsonl"
        hashes[str(path)] = file_hash(path)
        episodes.append(audit_trace(trace_rows(path), terminal))
    if any(file_hash(path) != sha for path, sha in hashes.items()):
        raise RuntimeError("audit inputs changed while reading")
    result = summarize_audits(episodes)
    result["sources"] = dict(run_directory=str(run_dir), input_sha256=hashes,
        source_inventory_sha256=identity["sources_before"]["inventory"]["sha256"],
        audit_script_sha256=file_hash(__file__))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)
    output = check_output(args.output_json, args.run_dir)
    result = analyze(args.run_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
