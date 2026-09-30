"""Export one two-panel static research diagnostic from completed D2 analysis.

Run as a namespace module from the repository root. Inputs stay read-only;
outputs require a new directory. ``--self-check`` renders visibly synthetic
examples, including an empty paired persistent-failure subset, without a simulator.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path

from chapter3_bser.experiments.safe_search_v1.run_development import INDICES
from chapter3_bser.experiments.safe_search_v1.run_paired import ROOT, VARIANTS, file_hash, read_json

CATEGORIES = (
    ("pre_found_collision_searcher", "Pre-Found collision:\nsearcher", "#BF713B", "///"),
    ("pre_found_collision_executor", "Pre-Found collision:\nexecutor", "#85492D", "..."),
    ("pre_found_collision_mixed", "Pre-Found collision:\nmixed", "#E1AC78", "xx"),
    ("no_found_timeout", "No-Found timeout", "#D7DDE3", ""),
    ("post_found_failure", "Post-Found failure", "#A2A2C9", "\\\\"),
    ("success", "Mission success", "#39798A", ""),
)
OUTCOMES = {key for key, *_ in CATEGORIES if key != "post_found_failure"} | {
    "post_found_timeout", "post_found_obstacle_collision"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def nonnegative_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid nonnegative numeric value: " + label)
    return value


def prepare(analysis, audit):
    """Reconcile both complete artifacts before computing any plotted values."""
    if (analysis.get("schema") != "ch3.safe_search.development_analysis.v1"
            or analysis.get("complete_stage") is not True
            or analysis.get("scenario_count") != 20 or analysis.get("episode_count") != 100
            or analysis.get("horizon") != 400 or analysis.get("baseline") != "B0_search_prior"
            or analysis.get("training") is not False or analysis.get("checkpoint_loaded") is not False
            or analysis.get("formal_thesis_evaluation") is not False):
        raise ValueError("figure requires a complete fixed-20 D2 analysis")
    if audit.get("schema") != "ch3.safe_search.development_window_audit.v1":
        raise ValueError("unsupported window-audit schema")
    expected = {(index, variant) for index in INDICES for variant in VARIANTS}
    def index_records(records):
        if not isinstance(records, list) or len(records) != 100:
            raise ValueError("figure requires all 100 unique episode records")
        result = {}
        for row in records:
            key = (row["original_episode_index"], row["variant"])
            if key not in expected or key in result or row.get("outcome") not in OUTCOMES:
                raise ValueError("unexpected/duplicate episode identity or outcome")
            result[key] = row
        if result.keys() != expected:
            raise ValueError("incomplete figure population")
        return result
    first, second = index_records(analysis.get("episode_diagnostics")), index_records(audit.get("episodes"))
    for key in expected:
        for name in ("scenario_id", "outcome", "found_step"):
            if first[key].get(name) != second[key].get(name):
                raise ValueError("analysis/window-audit episode mismatch: " + str((key, name)))
    source = analysis.get("sources", {}).get("source_inventory_sha256")
    if not source or source != audit.get("sources", {}).get("source_inventory_sha256"):
        raise ValueError("analysis and audit must identify the same source inventory")
    if bool(analysis.get("synthetic_test")) != bool(audit.get("synthetic_test")):
        raise ValueError("cannot mix real and synthetic artifacts")
    counts, found = {}, {}
    for variant in VARIANTS:
        arm = [first[index, variant] for index in INDICES]
        partition = dict(Counter(row["outcome"] for row in arm))
        summary = analysis["variants"][variant]
        supplied = {name: count for name, count in summary["outcome_partition"].items() if count}
        if supplied != partition or summary.get("n") != 20:
            raise ValueError("outcome partition does not reconcile to 20 episodes")
        found[variant] = sum(row["found_step"] is not None for row in arm)
        if found[variant] != summary["found_count"]:
            raise ValueError("Found count disagrees with episode records")
        counts[variant] = {key: partition.get(key, 0) for key, *_ in CATEGORIES}
        counts[variant]["post_found_failure"] = sum(partition.get(key, 0)
            for key in ("post_found_timeout", "post_found_obstacle_collision"))
        if sum(counts[variant].values()) != 20:
            raise ValueError("plotted outcome classes must be mutually exclusive and complete")
    paired_indices = [index for index in INDICES if all(
        second[index, variant]["outcome"] == "no_found_timeout" for variant in ("V0", "V4"))]
    paired = audit["paired_v0_v4_both_no_found_timeout"]
    n = len(paired_indices)
    if paired["n_pairs"] != n or paired["original_episode_indices"] != paired_indices:
        raise ValueError("paired persistent-failure subset identity mismatch")
    means = {variant: [] for variant in ("V0", "V4")}
    for variant in means:
        group = paired["variants"][variant]
        if group["episode_count"] != n or group["original_episode_indices"] != paired_indices:
            raise ValueError("paired-window aggregation population mismatch")
        if len(group["windows"]) != 4:
            raise ValueError("four 100-step windows are required")
        for index, window in enumerate(group["windows"]):
            if (window["first_step"], window["last_step"]) != (1 + index * 100, 100 + index * 100):
                raise ValueError("unexpected physical-step window")
            if window["pre_found_exposure_steps"] != n * 100:
                raise ValueError("shared no-Found timeouts require full paired window exposure")
            rows = [second[scene, variant]["windows"][index] for scene in paired_indices]
            available = bool(rows) and all(row["coverage_available"] for row in rows)
            if window["coverage_available"] is not available:
                raise ValueError("window coverage availability mismatch")
            if not available:
                means[variant].append(None)
                continue
            total = 0
            for name in ("new_cell_observations", "aged_revisit_cell_observations"):
                value = sum(nonnegative_number(row[name], name) for row in rows)
                if value != window[name]:
                    raise ValueError("paired window cell counts disagree with individual episodes")
                total += value
            means[variant].append(total / n)
    return dict(outcome_counts=counts, found_counts=found,
                paired_persistent_failure_count=n, paired_original_indices=paired_indices,
                mean_new_plus_aged_cell_observations=means,
                synthetic_test=bool(analysis.get("synthetic_test")))


def render(prepared, output):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#555D66", "text.color": "#252B32", "axes.labelcolor": "#252B32"})
    fig = plt.figure(figsize=(14.4, 7.3), facecolor="white")
    left = fig.add_axes([0.070, 0.34, 0.415, 0.46])
    right = fig.add_axes([0.595, 0.34, 0.375, 0.46])
    title = ("SYNTHETIC CHECK - NOT EXPERIMENT RESULTS" if prepared["synthetic_test"]
             else "Safe-search development diagnostics")
    fig.text(0.070, 0.955, title, size=17, weight="bold", ha="left")
    fig.text(0.070, 0.907, "B0 | 20 fixed scenarios per variant | 400-step task horizon | Development evidence",
             size=11, color="#58616B", ha="left")
    x = list(range(5))
    base = [0] * 5
    for category, label, color, hatch in CATEGORIES:
        values = [prepared["outcome_counts"][variant][category] for variant in VARIANTS]
        if not any(values):
            continue
        left.bar(x, values, bottom=base, width=0.66, color=color, edgecolor="white",
                 linewidth=0.6, hatch=hatch, label=label)
        for point, value, bottom in zip(x, values, base):
            if value:
                left.text(point, bottom + value / 2, str(value), ha="center", va="center", size=10,
                          color="white" if category in ("pre_found_collision_executor", "success") else "#252B32")
        base = [old + new for old, new in zip(base, values)]
    for point, variant in zip(x, VARIANTS):
        left.text(point, 20.65, f"Found {prepared['found_counts'][variant]}/20", ha="center", size=9)
    left.set(xticks=x, xticklabels=VARIANTS, ylim=(0, 23), yticks=[0, 5, 10, 15, 20],
             ylabel="Episodes (count)", xlabel="Variant")
    left.set_title("A  Mutually exclusive terminal outcomes", loc="left", pad=16, weight="bold")
    left.set_axisbelow(True)
    left.grid(axis="y", color="#E7EAEE", linewidth=0.7)
    left.legend(loc="upper left", bbox_to_anchor=(-0.01, -0.24), ncol=3, frameon=False,
                fontsize=8.5, columnspacing=1.0, handlelength=1.7, handletextpad=0.6,
                labelspacing=1.15, borderaxespad=0)
    right.set_title("B  Evidence renewal in persistent failures", loc="left", pad=16, weight="bold")
    n = prepared["paired_persistent_failure_count"]
    if not n:
        right.set_axis_off()
        right.text(0.5, 0.53, "No shared no-Found timeouts\nPaired V0 / V4 subset: n = 0",
                   ha="center", va="center", transform=right.transAxes, size=13, color="#58616B")
    else:
        for variant, offset, color, hatch in (("V0", -0.18, "#7B8792", ""), ("V4", 0.18, "#35739B", "//")):
            values = prepared["mean_new_plus_aged_cell_observations"][variant]
            positions = [index + offset for index in range(4)]
            bars = right.bar(positions, [value if value is not None else math.nan for value in values],
                             width=0.34, label=variant, color=color, edgecolor="white", linewidth=0.5, hatch=hatch)
            for position, value, bar in zip(positions, values, bars):
                if value is None:
                    right.text(position, 0, "NA", ha="center", va="bottom", size=9)
                else:
                    right.annotate(f"{value:.1f}", (position, value), xytext=(0, 4), textcoords="offset points",
                                   ha="center", va="bottom", size=9)
        observed = [value for series in prepared["mean_new_plus_aged_cell_observations"].values()
                    for value in series if value is not None]
        right.set_ylim(0, max(observed + [1]) * 1.20)
        right.set(xticks=list(range(4)), xticklabels=["1-100", "101-200", "201-300", "301-400"],
                  xlabel="Physical-step window", ylabel="New + aged cell observations\n(mean count per paired episode)")
        right.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        right.set_axisbelow(True)
        right.grid(axis="y", color="#E7EAEE", linewidth=0.7)
        right.legend(loc="upper right", ncol=2, frameon=False, fontsize=10)
    right.text(0, -0.245, f"Paired persistent-failure subset: n = {n}\nBoth V0 and V4 ended at step 400 without Found.",
               transform=right.transAxes, va="top", size=10, color="#58616B")
    fig.text(0.070, 0.058,
        "B is selected by both outcomes and does not estimate an overall causal effect. Descriptive means; no inferential error bars.\n"
        "New + aged target-evidence cell observations are a footprint-renewal proxy, not target detection probability.",
        ha="left", va="bottom", size=9, color="#58616B", linespacing=1.5)
    fig.savefig(output / "development_diagnostics.png", dpi=220, facecolor="white")
    fig.savefig(output / "development_diagnostics.pdf", facecolor="white",
                metadata={"Title": title, "Subject": "Fixed-20 development diagnosis, not independent thesis evaluation"})
    plt.close(fig)
    return matplotlib.__version__


def validate_output(output, inputs=()):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("figure output requires a new directory")
    if output == ROOT or any(output == p.resolve() or output in p.resolve().parents for p in inputs):
        raise ValueError("figure directory cannot contain source inputs")
    for name in ("outputs", "3090结果", ".git", "core", "chapter3_bser", "configs", "tests", "tools", "scripts"):
        protected = ROOT / name
        if output == protected or protected in output.parents:
            raise ValueError("figure output cannot replace retained data or production source")
    return output


def save(analysis, audit, output, sources):
    prepared = prepare(analysis, audit)
    output = validate_output(output)
    output.mkdir(parents=True)
    version = render(prepared, output)
    sidecar = dict(schema="ch3.safe_search.development_figure_sources.v1", sources=sources,
        script_sha256=file_hash(__file__), matplotlib_version=version, backend="Agg",
        analysis_content_sha256=digest(analysis), audit_content_sha256=digest(audit),
        plotted_data=prepared, output_sha256={name: file_hash(output / name)
            for name in ("development_diagnostics.png", "development_diagnostics.pdf")})
    with (output / "sources.json").open("x", encoding="utf-8") as handle:
        json.dump(sidecar, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return sidecar


def synthetic_example(empty=False):
    """Explicit renderer fixtures, separate from any experiment evidence."""
    from .development_trace_audit import summarize_audits
    episodes = []
    variants = {}
    for variant in VARIANTS:
        outcomes = []
        for ordinal, index in enumerate(INDICES):
            timeout = ordinal in ([7, 8, 9, 10] if variant == "V4" and not empty
                                  else [] if variant == "V4" else [6, 7, 8, 9])
            label = ("no_found_timeout" if timeout else "pre_found_collision_searcher" if ordinal < 3
                     else "pre_found_collision_executor" if ordinal == 3 else "pre_found_collision_mixed" if ordinal == 4
                     else "post_found_timeout" if ordinal == 19 else "success")
            outcomes.append(label)
            windows = []
            for w in range(4):
                new = max(0, (90 - ordinal) if w == 0 else (10 + ordinal) if w == 1 else 0)
                aged = 3 if w == 2 and variant == "V4" else 0
                windows.append(dict(first_step=1 + w * 100, last_step=100 + w * 100,
                    pre_found_exposure_steps=100, coverage_available=True, coverage_unavailable_steps=0,
                    new_cell_observations=new, aged_revisit_cell_observations=aged,
                    effective_observation_steps=2, observation_calls=20, target_update_call_steps=20,
                    observation_update_steps=20, repeated_cell_observations=100,
                    searcher_motion_distance=30.0, searcher_stall_agent_steps=0,
                    searcher_assignment_id_switches=0, searcher_semantic_waypoint_switches=0))
            episodes.append(dict(original_episode_index=index, variant=variant, scenario_id=f"synthetic_{index}",
                outcome=label, found_step=100 if label in ("success", "post_found_timeout") else None,
                windows=windows, timeout_final100_without_new_or_aged_evidence=True if timeout else None,
                trailing_search_steps_without_target_update_call=0,
                trailing_search_steps_without_nonempty_footprint_update=0,
                trailing_search_steps_without_new_or_aged_evidence=100))
        variants[variant] = dict(n=20, outcome_partition=dict(Counter(outcomes)),
                                found_count=sum(label in ("success", "post_found_timeout") for label in outcomes))
    audit = summarize_audits(episodes)
    audit.update(synthetic_test=True, sources=dict(source_inventory_sha256="synthetic-renderer-fixture"))
    analysis = dict(schema="ch3.safe_search.development_analysis.v1", complete_stage=True,
        scenario_count=20, episode_count=100, horizon=400, baseline="B0_search_prior", training=False,
        checkpoint_loaded=False, formal_thesis_evaluation=False, variants=variants,
        episode_diagnostics=copy.deepcopy(episodes), synthetic_test=True,
        sources=dict(source_inventory_sha256="synthetic-renderer-fixture"))
    return analysis, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--window-audit", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        if args.analysis or args.window_audit:
            parser.error("--self-check cannot use experiment inputs")
        output = validate_output(args.output_dir)
        output.mkdir(parents=True)
        for label, empty in (("paired_subset", False), ("empty_subset", True)):
            analysis, audit = synthetic_example(empty=empty)
            plotted = prepare(analysis, audit)
            assert plotted["paired_persistent_failure_count"] == (0 if empty else 3)
            save(analysis, audit, output / label, dict(synthetic_test=True, fixture=label))
        print("Synthetic PNG/PDF renderer checks passed; no simulator or experiment data used.")
        return
    if args.analysis is None or args.window_audit is None:
        parser.error("both --analysis and --window-audit are required")
    output = validate_output(args.output_dir, (args.analysis, args.window_audit))
    hashes = {str(path.resolve()): file_hash(path) for path in (args.analysis, args.window_audit)}
    analysis, audit = read_json(args.analysis), read_json(args.window_audit)
    if any(file_hash(path) != sha for path, sha in hashes.items()):
        raise RuntimeError("figure input changed during reading")
    save(analysis, audit, output, dict(input_sha256=hashes))


if __name__ == "__main__":
    main()
