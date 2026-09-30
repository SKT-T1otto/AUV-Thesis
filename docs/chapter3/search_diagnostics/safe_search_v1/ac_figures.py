"""Static figures from the completed A+C analysis, without simulator access.

Run from the repository root using ``python -B -m
docs.chapter3.search_diagnostics.safe_search_v1.ac_figures``. The input must be
the final analyze_ac_development analysis.json. ``--self-check`` validates
synthetic schemas in memory only; it never exports synthetic figures.
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
from chapter3_bser.experiments.safe_search_v1.run_paired import ROOT, file_hash

VARIANTS = ("V0", "V1", "V2", "V3", "V4", "V5")
OUTCOMES = {"pre_found_collision_searcher", "pre_found_collision_executor",
    "pre_found_collision_mixed", "no_found_timeout", "post_found_timeout",
    "post_found_obstacle_collision", "success"}
CATEGORIES = (
    ("pre_found_collision", "Pre-Found collision", "#BF713B", "///"),
    ("no_found_timeout", "No-Found timeout", "#D7DDE3", ""),
    ("post_found_failure", "Post-Found failure", "#A2A2C9", "\\\\"),
    ("success", "Mission success", "#39798A", ""),
)
FIELD_PATHS = [
    "variants.<variant>.n", "variants.<variant>.outcome_partition",
    "variants.<variant>.found_count", "variants.<variant>.success_count",
    "variants.<variant>.pre_found_collision_count",
    "variants.<variant>.pre_found_exposure_steps",
    "variants.<variant>.searcher_motion_stall_proxy_agent_steps",
    "variants.<variant>.searcher_hold_agent_steps",
    "variants.<variant>.searcher_motion_stall_proxy_fraction",
    "variants.<variant>.searcher_hold_fraction",
    "paired_v5_minus_reference.<reference>.scenario_pairs",
    "source_comparability.passed", "sources.all_inputs_unchanged",
    "sources.new_source_inventory_sha256", "sources.reference_source_inventory_sha256",
]


def integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or maximum is not None and value > maximum:
        raise ValueError("invalid integer: " + name)
    return value


def sha256(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("missing or invalid source SHA-256")
    return value


def check_fraction(actual, expected, name):
    if expected is None:
        if actual is not None:
            raise ValueError("zero exposure must have a null rate: " + name)
    elif (isinstance(actual, bool) or not isinstance(actual, (int, float))
            or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError("rate does not match pooled numerator/denominator: " + name)


def prepare(analysis, *, allow_synthetic=False):
    """Reconcile the six summaries to the 20 prespecified paired scene records."""
    synthetic = analysis.get("synthetic_test", False)
    if synthetic and not allow_synthetic:
        raise ValueError("synthetic inputs cannot be exported as A+C results")
    required = dict(schema="ch3.safe_search.ac_development_analysis.v1", complete_stage=True,
        new_variant="V5", new_episode_count=20, reference_episode_count=100, independent_scene_count=20,
        primary_reference="V3", secondary_references=["V0", "V4"],
        additional_descriptive_references=["V1", "V2"], training=False,
        checkpoint_loaded=False, formal_thesis_evaluation=False, performance_passed=None)
    if any(name not in analysis or analysis[name] != value for name, value in required.items()):
        raise ValueError("figure requires a complete fixed-20 A+C analysis")
    if (analysis["complete_stage"] is not True or analysis["training"] is not False
            or analysis["checkpoint_loaded"] is not False or analysis["formal_thesis_evaluation"] is not False
            or analysis.get("source_comparability", {}).get("passed") is not True
            or analysis.get("sources", {}).get("all_inputs_unchanged") is not True
            or analysis.get("original_registered_selection", {}).get("selected_variant") != "V3"):
        raise ValueError("analysis completion/source comparison/original selection is unverified")
    sources = analysis["sources"]
    source_identity = {name: sha256(sources.get(name)) for name in
        ("new_source_inventory_sha256", "reference_source_inventory_sha256")}
    if not isinstance(sources.get("all_input_sha256"), dict) or not sources["all_input_sha256"]:
        raise ValueError("analysis has no input fingerprint record")
    for value in sources["all_input_sha256"].values():
        sha256(value)
    summaries, pairs = analysis.get("variants", {}), analysis.get("paired_v5_minus_reference", {})
    if set(summaries) != set(VARIANTS) or set(pairs) != set(VARIANTS[:-1]):
        raise ValueError("six variant summaries and all five paired references are required")
    episodes, scene_ids = {}, {}
    for reference in VARIANTS[:-1]:
        records = pairs[reference].get("scenario_pairs")
        if not isinstance(records, list) or len(records) != 20:
            raise ValueError("each comparison must contain 20 scene pairs")
        seen = set()
        for pair in records:
            index = pair.get("original_episode_index")
            if type(index) is not int or index not in INDICES or index in seen:
                raise ValueError("duplicate/unexpected paired scene")
            seen.add(index)
            scene_id = pair.get("scenario_id")
            if not isinstance(scene_id, str) or not scene_id or scene_ids.setdefault(index, scene_id) != scene_id:
                raise ValueError("paired scene IDs differ across references")
            for variant, prefix in ((reference, "reference"), ("V5", "v5")):
                outcome, found = pair.get(prefix + "_outcome"), pair.get(prefix + "_found_step")
                if outcome not in OUTCOMES:
                    raise ValueError("unknown terminal outcome")
                if found is not None:
                    integer(found, "Found step", maximum=400)
                if ((outcome == "no_found_timeout" and found is not None)
                        or (outcome == "success" or outcome.startswith("post_found_")) and found is None):
                    raise ValueError("paired outcome and Found flag disagree")
                identity = (scene_id, outcome, found)
                if episodes.setdefault((index, variant), identity) != identity:
                    raise ValueError("V5 paired record differs across references")
        if seen != set(INDICES):
            raise ValueError("paired population differs from frozen development scenes")
    if len(episodes) != 120 or len(set(scene_ids.values())) != 20:
        raise ValueError("figure population must be 20 unique scenes by six variants")
    outcome_counts, found_counts, motion = {}, {}, {}
    for variant in VARIANTS:
        summary = summaries[variant]
        if type(summary.get("n")) is not int or summary["n"] != 20:
            raise ValueError("every variant must have 20 completed episodes")
        rows = [episodes[index, variant] for index in INDICES]
        counts = Counter(row[1] for row in rows)
        supplied = summary.get("outcome_partition", {})
        if not isinstance(supplied, dict) or set(supplied) - OUTCOMES:
            raise ValueError("unknown outcome partition")
        for value in supplied.values():
            integer(value, "outcome count", maximum=20)
        if {name: value for name, value in supplied.items() if value} != dict(counts):
            raise ValueError("summary outcome partition differs from paired records")
        found = sum(row[2] is not None for row in rows)
        pre_collision = sum(value for name, value in counts.items() if name.startswith("pre_found_collision_"))
        for name, expected in (("found_count", found), ("success_count", counts["success"]),
                ("pre_found_collision_count", pre_collision)):
            if integer(summary.get(name), name, maximum=20) != expected:
                raise ValueError("summary count differs from paired outcome/Found records: " + name)
        found_counts[variant] = found
        outcome_counts[variant] = dict(pre_found_collision=pre_collision,
            no_found_timeout=counts["no_found_timeout"], success=counts["success"],
            post_found_failure=counts["post_found_timeout"] + counts["post_found_obstacle_collision"])
        if sum(outcome_counts[variant].values()) != 20:
            raise ValueError("plotted outcomes are not a complete mutually exclusive partition")
        exposure = integer(summary.get("pre_found_exposure_steps"), "search exposure", maximum=8000)
        denominator = 3 * exposure
        stall = integer(summary.get("searcher_motion_stall_proxy_agent_steps"), "stall", maximum=denominator)
        hold = integer(summary.get("searcher_hold_agent_steps"), "Hold", maximum=denominator)
        if stall + hold > denominator:
            raise ValueError("disjoint stall/Hold total exceeds available search agent-steps")
        # Exposure is also bounded by the paired Found and no-Found timeout records.
        fixed_exposure = sum(row[2] if row[2] is not None else 400
                             for row in rows if row[2] is not None or row[1] == "no_found_timeout")
        early_collisions = sum(row[2] is None and row[1].startswith("pre_found_collision_") for row in rows)
        if not fixed_exposure + early_collisions <= exposure <= fixed_exposure + 400 * early_collisions:
            raise ValueError("summary exposure is incompatible with paired outcomes and Found times")
        stall_rate, hold_rate = (stall / denominator, hold / denominator) if denominator else (None, None)
        check_fraction(summary.get("searcher_motion_stall_proxy_fraction"), stall_rate, "stall")
        check_fraction(summary.get("searcher_hold_fraction"), hold_rate, "Hold")
        motion[variant] = dict(pre_found_exposure_steps=exposure, denominator_search_agent_steps=denominator,
            stall_agent_steps=stall, hold_agent_steps=hold, stall_fraction=stall_rate, hold_fraction=hold_rate,
            combined_fraction=(stall + hold) / denominator if denominator else None)
    return dict(variants=list(VARIANTS), scenario_count=20, episode_count=120,
        outcome_counts=outcome_counts, found_counts=found_counts, motion=motion,
        source_identity=source_identity, synthetic_test=bool(synthetic))


def render(prepared, output):
    if prepared.get("synthetic_test"):
        raise ValueError("synthetic figure export is disabled")
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#555D66", "text.color": "#252B32", "axes.labelcolor": "#252B32"})
    fig = plt.figure(figsize=(14.8, 7.6), facecolor="white")
    left = fig.add_axes([0.064, 0.36, 0.414, 0.44])
    right = fig.add_axes([0.582, 0.36, 0.385, 0.44])
    fig.text(0.064, 0.953, "Safe-search A+C development comparison", size=17, weight="bold")
    fig.text(0.064, 0.904,
        "B0 | Same 20 fixed scenarios per variant | 400-step task horizon | V5 = A+C",
        size=11, color="#58616B")
    positions, base = list(range(6)), [0] * 6
    for axis in (left, right):
        for point in (3, 5):
            axis.axvspan(point - 0.45, point + 0.45, color="#E9EDF0", alpha=0.55, zorder=0)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#E7EAEE", linewidth=0.7)
    for category, label, color, hatch in CATEGORIES:
        values = [prepared["outcome_counts"][variant][category] for variant in VARIANTS]
        if not any(values):
            continue
        left.bar(positions, values, bottom=base, width=0.66, color=color, edgecolor="white",
                 linewidth=0.6, hatch=hatch, label=label)
        for point, value, bottom in zip(positions, values, base):
            if value:
                left.text(point, bottom + value / 2, str(value), ha="center", va="center", size=10,
                          color="white" if category == "success" else "#252B32")
        base = [a + b for a, b in zip(base, values)]
    for point, variant in enumerate(VARIANTS):
        left.text(point, 20.7, f"{prepared['found_counts'][variant]}/20", ha="center", size=10)
    left.text(0, 1.013, "Found:", transform=left.transAxes, size=9, color="#58616B")
    left.set(xticks=positions, xticklabels=VARIANTS, ylim=(0, 23), yticks=[0, 5, 10, 15, 20],
             ylabel="Episodes (count)", xlabel="Variant")
    left.set_title("A  Mutually exclusive terminal outcomes", loc="left", pad=20, weight="bold")
    left.legend(loc="upper left", bbox_to_anchor=(-0.01, -0.23), ncol=2, frameon=False,
                fontsize=9, columnspacing=1.7, handlelength=1.8, labelspacing=1.0, borderaxespad=0)
    stalls = [100 * prepared["motion"][v]["stall_fraction"]
              if prepared["motion"][v]["stall_fraction"] is not None else math.nan for v in VARIANTS]
    holds = [100 * prepared["motion"][v]["hold_fraction"]
             if prepared["motion"][v]["hold_fraction"] is not None else math.nan for v in VARIANTS]
    right.bar(positions, stalls, width=0.66, color="#A2A2C9", edgecolor="white", linewidth=0.6,
              hatch="//", label="Stall proxy (excludes Hold)")
    right.bar(positions, holds, bottom=stalls, width=0.66, color="#C49B45", edgecolor="white",
              linewidth=0.6, label="Hold")
    tick_labels = []
    for point, variant in enumerate(VARIANTS):
        item = prepared["motion"][variant]
        tick_labels.append(f"{variant}\nD={item['denominator_search_agent_steps']:,}")
        value = item["combined_fraction"]
        right.text(point, 2 if value is None else value * 100 + 2,
                   "NA" if value is None else f"{value:.1%}", ha="center", va="bottom", size=9)
    right.set(xticks=positions, xticklabels=tick_labels, ylim=(0, 110), yticks=[0, 25, 50, 75, 100],
              ylabel="Share of pre-Found search-agent steps", xlabel="Variant; D = pooled denominator")
    right.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    right.set_title("B  Stall and Hold exposure", loc="left", pad=20, weight="bold")
    right.legend(loc="upper left", bbox_to_anchor=(-0.01, -0.27), ncol=1,
                 frameon=False, fontsize=9, labelspacing=1.0, borderaxespad=0)
    for axis in (left, right):
        for point, tick in enumerate(axis.get_xticklabels()):
            if point in (3, 5):
                tick.set_weight("bold")
    fig.text(0.064, 0.097,
        "B: pooled agent-step counts / (3 searchers x pooled pre-Found steps); the Found transition is included. Stall and Hold are disjoint.\n"
        "Stall is a 10-step stable-assignment motion proxy with low progress and displacement; it excludes Hold and unreachable assignments.",
        ha="left", va="bottom", size=9, color="#58616B", linespacing=1.5)
    fig.text(0.064, 0.032,
        "V3 and V5 are highlighted for comparison only. V5 is a post-hoc development addition; these bars do not establish statistical superiority.\n"
        "Exposure depends on discovery and collision times. Collision at the Found step has pre-Found precedence; Found is distinct from mission success.",
        ha="left", va="bottom", size=9, color="#58616B", linespacing=1.5)
    fig.savefig(output / "ac_development_diagnostics.png", dpi=220, facecolor="white")
    fig.savefig(output / "ac_development_diagnostics.pdf", facecolor="white",
        metadata={"Title": "Safe-search A+C development comparison",
                  "Subject": "20 paired development scenarios; no independent thesis performance claim"})
    plt.close(fig)
    return matplotlib.__version__


def validate_output(path, input_path):
    output, source = Path(path).resolve(), Path(input_path).resolve()
    if output.exists():
        raise FileExistsError("figure output must be a new directory")
    if output == ROOT or output == source or output in source.parents:
        raise ValueError("figure output cannot contain or replace its input")
    for name in ("outputs", "3090结果", ".git", "core", "chapter3_bser", "configs", "tests", "tools", "scripts"):
        protected = ROOT / name
        if output == protected or protected in output.parents:
            raise ValueError("figure output cannot modify production source or retained inputs")
    return output


def save(analysis_path, output_path):
    analysis_path = Path(analysis_path).resolve()
    output = validate_output(output_path, analysis_path)
    fingerprint = file_hash(analysis_path)
    def reject(value):
        raise ValueError("nonfinite analysis value: " + value)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"), parse_constant=reject)
    prepared = prepare(analysis)
    if file_hash(analysis_path) != fingerprint:
        raise RuntimeError("figure input changed during validation")
    output.mkdir(parents=True)
    version = render(prepared, output)
    if file_hash(analysis_path) != fingerprint:
        raise RuntimeError("figure input changed during rendering")
    sidecar = dict(schema="ch3.safe_search.ac_figure_sources.v1",
        input_sha256={str(analysis_path): fingerprint}, script_sha256=file_hash(__file__),
        analysis_content_sha256=hashlib.sha256(json.dumps(analysis, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        matplotlib_version=version, backend="Agg", plotted_field_paths=FIELD_PATHS, plotted_data=prepared,
        definitions=dict(outcomes="Mutually exclusive terminal partition; collision at Found takes pre-Found precedence.",
            found="Count of episodes with any non-null Found step; distinct from success.",
            motion_denominator="Three searchers times pooled pre-Found exposure, including the Found transition.",
            stall="Existing 10-step stable-assignment low-motion proxy; excludes Hold and unreachable assignments.",
            hold="Pre-Found search-agent steps whose before-step guidance is Hold; disjoint from the stall proxy.",
            zero_exposure="Undefined rates are null and rendered NA, never zero.",
            aggregation="Ratio of pooled counts, not a mean of per-episode rates.",
            interpretation="Descriptive development counts/rates; no statistical superiority or independent test claim."),
        validation=dict(complete_analysis_schema=True, fixed20_by6_pair_partition_reconciled=True,
            found_counts_reconciled=True, disjoint_motion_bounds_and_pooled_rates_reconciled=True,
            raw_traces_reaudited_by_figure=False, raw_trace_authority="analyze_ac_development input verification"),
        output_sha256={name: file_hash(output / name) for name in
                      ("ac_development_diagnostics.png", "ac_development_diagnostics.pdf")})
    with (output / "sources.json").open("x", encoding="utf-8") as handle:
        json.dump(sidecar, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return sidecar


def self_check():
    """Small schema fixtures only: no files, real outcomes or simulator are used."""
    fixture = dict(schema="ch3.safe_search.ac_development_analysis.v1", complete_stage=True,
        new_variant="V5", new_episode_count=20, reference_episode_count=100, independent_scene_count=20,
        primary_reference="V3", secondary_references=["V0", "V4"], additional_descriptive_references=["V1", "V2"],
        original_registered_selection=dict(selected_variant="V3"), source_comparability=dict(passed=True),
        training=False, checkpoint_loaded=False, formal_thesis_evaluation=False, performance_passed=None,
        synthetic_test=True, sources=dict(new_source_inventory_sha256="a" * 64,
            reference_source_inventory_sha256="b" * 64, all_inputs_unchanged=True,
            all_input_sha256={"synthetic": "c" * 64}), variants={}, paired_v5_minus_reference={})
    for variant in VARIANTS:
        fixture["variants"][variant] = dict(n=20, outcome_partition=dict(no_found_timeout=20),
            found_count=0, success_count=0, pre_found_collision_count=0, pre_found_exposure_steps=8000,
            searcher_motion_stall_proxy_agent_steps=2400, searcher_hold_agent_steps=1200,
            searcher_motion_stall_proxy_fraction=0.1, searcher_hold_fraction=0.05)
    for variant in VARIANTS[:-1]:
        fixture["paired_v5_minus_reference"][variant] = dict(scenario_pairs=[dict(original_episode_index=index,
            scenario_id=f"synthetic_{index}", reference_outcome="no_found_timeout", v5_outcome="no_found_timeout",
            reference_found_step=None, v5_found_step=None) for index in INDICES])
    result = prepare(fixture, allow_synthetic=True)
    assert result["motion"]["V5"]["combined_fraction"] == 0.15
    edits = [lambda a: a.update(complete_stage=False),
        lambda a: a["variants"]["V5"].update(n=19),
        lambda a: a["variants"]["V5"].update(found_count=1),
        lambda a: a["variants"]["V5"].update(searcher_hold_agent_steps=24000),
        lambda a: a["variants"]["V5"].update(searcher_hold_fraction=0.051),
        lambda a: a["paired_v5_minus_reference"]["V0"]["scenario_pairs"].pop(),
        lambda a: a["source_comparability"].update(passed=False),
        lambda a: a["sources"].update(new_source_inventory_sha256="missing")]
    for edit in edits:
        broken = copy.deepcopy(fixture)
        edit(broken)
        try:
            prepare(broken, allow_synthetic=True)
        except ValueError:
            continue
        raise AssertionError("invalid fixture was accepted")
    try:
        prepare(fixture)
    except ValueError:
        pass
    else:
        raise AssertionError("synthetic export input was accepted")
    # Valid null-rate boundary: every episode discovers at reset and succeeds.
    zero = copy.deepcopy(fixture)
    for summary in zero["variants"].values():
        summary.update(outcome_partition=dict(success=20), found_count=20, success_count=20,
            pre_found_exposure_steps=0, searcher_motion_stall_proxy_agent_steps=0,
            searcher_hold_agent_steps=0, searcher_motion_stall_proxy_fraction=None, searcher_hold_fraction=None)
    for paired in zero["paired_v5_minus_reference"].values():
        for pair in paired["scenario_pairs"]:
            pair.update(reference_outcome="success", v5_outcome="success", reference_found_step=0, v5_found_step=0)
    assert prepare(zero, allow_synthetic=True)["motion"]["V5"]["combined_fraction"] is None
    print("Synthetic schema checks passed (2 valid, 9 rejected); no figures or experiment files written.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        if args.analysis or args.output_dir:
            parser.error("--self-check uses no experiment inputs or output directory")
        self_check()
        return
    if args.analysis is None or args.output_dir is None:
        parser.error("--analysis and --output-dir are required")
    save(args.analysis, args.output_dir)


if __name__ == "__main__":
    main()
