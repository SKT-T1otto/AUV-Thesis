"""Read-only, paired analysis of the frozen 20-scenario development stage.

No simulator, model, or checkpoint is loaded. Incomplete stages fail closed;
program errors never become physical failures or disappear from denominators.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import statistics

from .run_paired import ROOT, VARIANTS, file_hash, output_directory, read_json, write_json

PLAN = ROOT / "docs/chapter3/search_diagnostics/3090_20260928/experiment_plan.json"
HORIZON = 400


def validate_rows(rows, plan):
    """Require exactly the registered 100 identities and actual task terminals."""
    scenes = plan["splits"]["development"]
    if len(scenes) != 20 or len({s["source_episode_index"] for s in scenes}) != 20:
        raise ValueError("development plan must contain 20 unique original indices")
    expected = {(s["source_episode_index"], v): s for s in scenes for v in VARIANTS}
    actual = {}
    for row in rows:
        key = (row["original_episode_index"], row["variant"])
        if key not in expected or key in actual:
            raise ValueError("unexpected or duplicate scheduled identity: " + str(key))
        scene = expected[key]
        for field in ("scenario_id", "scenario_seed", "environment_innovation_seed"):
            if row[field] != scene[field]:
                raise ValueError("scenario/seed mismatch: " + str(key))
        if row.get("baseline") != "B0_search_prior":
            raise ValueError("D2 is the registered B0 development stage")
        if not (row.get("terminal") is True and row.get("full_episode_completed") is True
                and row.get("full_episode_requested") is True):
            raise ValueError("incomplete task episode: " + str(key))
        step, found_step = row["physical_steps"], row["found_step"]
        if type(step) is not int or not 1 <= step <= HORIZON:
            raise ValueError("invalid physical length")
        if type(row.get("found_within_budget")) is not bool or row["found_within_budget"] != (found_step is not None):
            raise ValueError("Found flag/step mismatch")
        if found_step is not None and (type(found_step) is not int or not 0 <= found_step <= step):
            raise ValueError("invalid Found step")
        result = row["episode_result"]
        reason = row["stop_reason"]
        if reason not in ("success", "timeout", "obstacle_collision") or reason != result["termination_reason"]:
            raise ValueError("unknown or inconsistent physical outcome")
        if result.get("task_protocol") != "collision_terminal_v1" or result.get("terminal_step") != step:
            raise ValueError("task contract mismatch")
        if reason == "timeout" and step != HORIZON:
            raise ValueError("short prefix cannot be classified as task timeout")
        if bool(result.get("success")) != (reason == "success") or reason == "success" and found_step is None:
            raise ValueError("mission success is inconsistent with Found")
        collision = result.get("first_collision_step")
        if (reason == "obstacle_collision") != (collision is not None):
            raise ValueError("collision metadata mismatch")
        pre_collision = collision is not None and (found_step is None or collision <= found_step)
        if row.get("pre_found_collision") is not pre_collision:
            raise ValueError("pre-Found collision flag mismatch")
        expected_exposure = found_step if found_step is not None else step
        if row.get("pre_found_exposure_steps") != expected_exposure:
            raise ValueError("pre-Found exposure/Found step mismatch")
        for name in ("searcher_motion_stall_proxy_agent_steps", "searcher_hold_agent_steps"):
            value = row[name]
            if type(value) is not int or not 0 <= value <= 3 * expected_exposure:
                raise ValueError("invalid search agent-step count")
        actual[key] = row
    if actual.keys() != expected.keys():
        raise ValueError("incomplete development stage: expected 100 arms, got " + str(len(actual)))
    return actual


def collision_role(row):
    ids = set(row["episode_result"].get("first_collision_agent_ids", []))
    if not ids or not ids <= {0, 1, 2, 3}:
        raise ValueError("collision must identify searchers 0..2 and/or executor 3")
    return "mixed" if 3 in ids and len(ids) > 1 else "executor" if ids == {3} else "searcher"


def outcome(row):
    if row["pre_found_collision"]:
        return "pre_found_collision_" + collision_role(row)
    if row["found_step"] is None:
        if row["stop_reason"] != "timeout":
            raise ValueError("unclassified no-Found terminal")
        return "no_found_timeout"
    return "success" if row["stop_reason"] == "success" else "post_found_" + row["stop_reason"]


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_bootstrap(differences, replicates=10000, seed=20260928):
    """Scenario-paired percentile interval, explicitly descriptive on this dev set."""
    if not differences or replicates < 1:
        raise ValueError("nonempty differences and positive bootstrap size required")
    rng = random.Random(seed)
    count = len(differences)
    samples = [sum(differences[rng.randrange(count)] for _ in range(count)) / count
               for _ in range(replicates)]
    return dict(mean_delta=statistics.mean(differences),
                bootstrap_95_percentile_ci=[quantile(samples, .025), quantile(samples, .975)],
                n_pairs=count, bootstrap_replicates=replicates, bootstrap_seed=seed)


def trace_summary(path):
    """Summarize only recorded public trajectories and actual path queries."""
    queries, callers, switches, movement = Counter(), Counter(), Counter(), Counter()
    semantic_switches = Counter()
    collision_snapshots = []
    n = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        n += 1
        if row["step_before"] != n - 1 or row["step_after"] != n:
            raise ValueError("trace is not a contiguous physical trajectory")
        if not row["search_transition"]:
            continue
        for q in row["queries"]["counts"]:
            queries[q["reason"]] += q["count"]
            callers[q["caller"] + ":" + q["reason"]] += q["count"]
        before = {a["agent_id"]: a for a in row["before"]["agents"]}
        after = {a["agent_id"]: a for a in row["after"]["agents"]}
        for agent in (0, 1, 2, 3):
            a, b = before[agent], after[agent]
            switches[str(agent)] += int(a["assignment_id"] != b["assignment_id"])
            if "semantic_waypoint" in a and "semantic_waypoint" in b:
                semantic_switches[str(agent)] += int(a["semantic_waypoint"] != b["semantic_waypoint"])
            movement[str(agent)] += math.dist(a["position"], b["position"])
        for hit in row.get("collision_records", []):
            agent = hit["agent_id"]
            a = before[agent]
            collision_snapshots.append(dict(agent_id=agent, step=n,
                obstacle_id=hit.get("obstacle_id"), speed_before=a["speed"],
                cross_track_error_before=a["active_segment_cross_track_error"],
                next_turn_angle_deg=a["next_turn_angle_deg"],
                velocity_to_tracking_angle_deg=a["velocity_to_tracking_angle_deg"],
                hold_before=a["hold"], reachable_before=a["reachable"],
                planned_path=a["planned_path"], position_before=a["position"],
                first_hit_point=hit.get("first_hit_point")))
    return dict(physical_steps=n, pre_found_query_reasons=dict(queries),
                pre_found_query_callers=dict(callers),
                assignment_switches_by_agent=dict(switches),
                semantic_waypoint_switches_by_agent=dict(semantic_switches),
                pre_found_path_length_by_agent=dict(movement),
                pre_found_collision_snapshots=collision_snapshots)


def aggregate(arm, traces):
    counts = Counter(outcome(row) for row in arm)
    exposure = sum(r["pre_found_exposure_steps"] for r in arm)
    stalls = sum(r["searcher_motion_stall_proxy_agent_steps"] for r in arm)
    holds = sum(r["searcher_hold_agent_steps"] for r in arm)
    found = sum(r["found_step"] is not None for r in arm)
    precoll = sum(r["pre_found_collision"] for r in arm)
    if sum(counts.values()) != 20:
        raise ValueError("outcome partition does not reconcile")
    query_reasons, allocation, safe_counts = Counter(), Counter(), Counter()
    for row in arm:
        controller = row["controller"]
        allocation.update(controller.get("allocation_counts", {}))
        safe_counts.update(controller.get("safe_search", {}).get("counts", {}))
        trace = traces.get((row["original_episode_index"], row["variant"]))
        if trace:
            query_reasons.update(trace["pre_found_query_reasons"])
    coverage = [r.get("search_coverage") for r in arm]
    measured = all(c is not None and c.get("available") is True for c in coverage)
    coverage_summary = None
    if measured:
        coverage_summary = {name: sum(c[name] for c in coverage) for name in
            ("effective_observation_steps", "observation_update_steps", "new_cell_observations",
             "repeated_cell_observations", "aged_revisit_cell_observations")}
        coverage_summary["mean_unique_observed_grid_fraction"] = statistics.mean(
            c["unique_observed_grid_fraction"] for c in coverage)
        coverage_summary["effective_observation_fraction"] = (
            coverage_summary["effective_observation_steps"] / exposure if exposure else None)
        overlap = [c.get("inter_agent_overlap") for c in coverage]
        if all(o is not None for o in overlap):
            total = sum(o["total_agent_cell_observations"] for o in overlap)
            redundant = sum(o["redundant_agent_cell_observations"] for o in overlap)
            coverage_summary["inter_agent_overlap"] = dict(total_agent_cell_observations=total,
                redundant_agent_cell_observations=redundant,
                redundant_agent_cell_fraction=redundant / total if total else None)
        else:
            coverage_summary["inter_agent_overlap"] = None
    return dict(n=20, found_count=found, found_rate=found / 20,
        success_count=counts["success"], success_rate=counts["success"] / 20,
        success_if_found=counts["success"] / found if found else None,
        pre_found_collision_count=precoll, pre_found_collision_rate=precoll / 20,
        outcome_partition=dict(counts), pre_found_exposure_steps=exposure,
        mean_pre_found_exposure_steps=exposure / 20,
        found_time_restricted_mean_400=statistics.mean(
            r["found_step"] if r["found_step"] is not None else HORIZON for r in arm),
        mean_found_step_conditional=statistics.mean([r["found_step"] for r in arm
            if r["found_step"] is not None]) if found else None,
        searcher_motion_stall_proxy_agent_steps=stalls, searcher_hold_agent_steps=holds,
        searcher_motion_stall_proxy_fraction=stalls / (3 * exposure) if exposure else None,
        searcher_hold_fraction=holds / (3 * exposure) if exposure else None,
        pre_found_query_reasons=dict(query_reasons) if len(traces) else None,
        allocation_counts_all_phases=dict(allocation), safe_search_counts=dict(safe_counts),
        actual_belief_footprint=coverage_summary,
        total_physical_steps=sum(r["physical_steps"] for r in arm),
        total_wall_seconds=sum(r["wall_seconds"] for r in arm))


def choose_variant(variants, *, d1_passed, wall_time_comparable):
    base = variants["V0"]
    gates = {}
    for name in VARIANTS[1:]:
        arm = variants[name]
        checks = dict(found_not_worse=arm["found_count"] >= base["found_count"],
            pre_collision_not_worse=arm["pre_found_collision_count"] <= base["pre_found_collision_count"],
            stall_not_worse=(arm["searcher_motion_stall_proxy_fraction"] is not None and
                base["searcher_motion_stall_proxy_fraction"] is not None and
                arm["searcher_motion_stall_proxy_fraction"] <= base["searcher_motion_stall_proxy_fraction"]),
            strict_primary_improvement=arm["found_count"] > base["found_count"] or
                arm["pre_found_collision_count"] < base["pre_found_collision_count"] or
                (arm["searcher_motion_stall_proxy_fraction"] is not None and
                 base["searcher_motion_stall_proxy_fraction"] is not None and
                 arm["searcher_motion_stall_proxy_fraction"] < base["searcher_motion_stall_proxy_fraction"]),
            d1_mechanisms_passed=bool(d1_passed))
        gates[name] = dict(checks, eligible=all(checks.values()))
    eligible = [v for v in VARIANTS[1:] if gates[v]["eligible"]]
    if not eligible:
        return dict(gates=gates, selected_variant=None, reason="no_variant_passes_registered_gate")
    key = lambda v: (variants[v]["pre_found_collision_count"], -variants[v]["found_count"],
                     variants[v]["searcher_motion_stall_proxy_fraction"])
    best_key = min(key(v) for v in eligible)
    tied = [v for v in eligible if key(v) == best_key]
    if len(tied) > 1 and not wall_time_comparable:
        return dict(gates=gates, selected_variant=None, tied_variants=tied,
            reason="registered_wall_time_tiebreak_requires_comparable_serial_timing")
    changes = {"V1": 1, "V2": 2, "V3": 1, "V4": 3}
    selected = min(tied, key=lambda v: (variants[v]["total_wall_seconds"], changes[v]))
    return dict(gates=gates, selected_variant=selected,
        reason="registered_development_engineering_gate_only_not_statistical_proof")


def analyze_rows(rows, plan, *, traces=None, d1_passed=False, wall_time_comparable=False):
    indexed = validate_rows(rows, plan)
    traces = traces or {}
    if traces and set(traces) != set(indexed):
        raise ValueError("trace evidence must cover all 100 arms")
    variants = {v: aggregate([r for r in rows if r["variant"] == v], traces) for v in VARIANTS}
    paired = {}
    parameters = plan["statistics"]
    indices = [s["source_episode_index"] for s in plan["splits"]["development"]]
    for variant in VARIANTS[1:]:
        pairs = [(indexed[i, "V0"], indexed[i, variant]) for i in indices]
        comparison = {}
        for metric, getter in (
            ("found", lambda r: int(r["found_step"] is not None)),
            ("pre_found_collision", lambda r: int(r["pre_found_collision"])),
            ("success", lambda r: int(r["episode_result"]["success"])),
            ("restricted_found_time_400", lambda r: r["found_step"] if r["found_step"] is not None else HORIZON),
            ("pre_found_exposure_steps", lambda r: r["pre_found_exposure_steps"]),
        ):
            differences = [getter(b) - getter(a) for a, b in pairs]
            comparison[metric] = paired_bootstrap(differences,
                parameters["bootstrap_replicates"], parameters["bootstrap_seed"])
            if metric in ("found", "pre_found_collision", "success"):
                comparison[metric]["discordant_0_to_1"] = sum(getter(a) == 0 and getter(b) == 1 for a, b in pairs)
                comparison[metric]["discordant_1_to_0"] = sum(getter(a) == 1 and getter(b) == 0 for a, b in pairs)
        comparison["scenario_pairs"] = [dict(original_episode_index=a["original_episode_index"],
            scenario_id=a["scenario_id"], v0_outcome=outcome(a), variant_outcome=outcome(b),
            found_delta=int(b["found_step"] is not None) - int(a["found_step"] is not None),
            pre_found_collision_delta=int(b["pre_found_collision"]) - int(a["pre_found_collision"]))
            for a, b in pairs]
        paired[variant] = comparison
    diagnostics = [dict(original_episode_index=r["original_episode_index"], scenario_id=r["scenario_id"],
        variant=r["variant"], outcome=outcome(r), found_step=r["found_step"],
        pre_found_exposure_steps=r["pre_found_exposure_steps"],
        stall_agent_steps=r["searcher_motion_stall_proxy_agent_steps"],
        hold_agent_steps=r["searcher_hold_agent_steps"],
        collision_agent_ids=r["episode_result"].get("first_collision_agent_ids", []),
        allocation_counts=r["controller"].get("allocation_counts", {}),
        safe_search=r["controller"].get("safe_search"), search_coverage=r.get("search_coverage"),
        trace=traces.get((r["original_episode_index"], r["variant"]))) for r in rows]
    return dict(schema="ch3.safe_search.development_analysis.v1", complete_stage=True,
        stage="D2", baseline="B0_search_prior", scenario_count=20, episode_count=100,
        horizon=HORIZON, variants=variants, paired_vs_v0=paired, episode_diagnostics=diagnostics,
        selection=choose_variant(variants, d1_passed=d1_passed, wall_time_comparable=wall_time_comparable),
        training=False, checkpoint_loaded=False, formal_thesis_evaluation=False,
        performance_passed=None,
        metric_definitions=dict(found="At least one target discovery before task terminal.",
            success="Completed collision-terminal mission, distinct from target discovery.",
            restricted_found_time_400="Mean Found step with every non-Found episode assigned 400, including early collision. Lower is better; no censoring assumption.",
            pre_found_exposure="Actual physical search transitions; survival time is not coverage.",
            stall="Original 10-step stable-assignment motion proxy, pooled over 3 searchers × search exposure.",
            footprint="Cells whose timestamps were written by the actual negative-belief update; not obstacle coverage, target probability, or continuous swept coverage."),
        caveats=["Development set was partly chosen for known failure cases; these estimates do not establish generalization.",
            "Percentile paired bootstrap uses 20 scenario pairs, 10000 resamples, seed 20260928; four comparisons are exploratory and not multiplicity-adjusted.",
            "Differences in exposure can change aggregate hold, stall, query, and coverage totals; inspect rates and paired cases.",
            "Collision role identifies the colliding agent, not a causal controller mechanism.",
            "Historical validation labels do not make the remaining 80 scenes a new independent formal test set."])


def markdown(result):
    lines = ["# 固定 20 场景开发评估", "",
        "B0，V0–V4，同一组原始索引和创新种子，400 步任务上限；100 个回合均已正常达到任务终止。未训练、未加载 checkpoint。", "",
        "| 变体 | Found | 任务成功 | 发现前碰撞 | 未发现超时 | 停滞占比 | Hold 占比 | 截断发现时间均值 |", 
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in VARIANTS:
        a = result["variants"][name]
        lines.append(f"| {name} | {a['found_count']}/20 | {a['success_count']}/20 | {a['pre_found_collision_count']}/20 | "
            f"{a['outcome_partition'].get('no_found_timeout', 0)}/20 | {a['searcher_motion_stall_proxy_fraction']:.2%} | "
            f"{a['searcher_hold_fraction']:.2%} | {a['found_time_restricted_mean_400']:.1f} |")
    lines += ["", "未发现回合（包括提前碰撞）在截断发现时间中统一记为 400；该指标越低越好。Found 是发现目标，任务成功还需要后续交接和执行完成。", "",
        "| 变体 − V0 | Found 差值（百分点）及配对 95% 区间 | 新增发现 / 丢失发现 | 碰撞差值（百分点）及配对 95% 区间 |", "|---|---:|---:|---:|"]
    for name, p in result["paired_vs_v0"].items():
        f, c = p["found"], p["pre_found_collision"]
        fi, ci = f["bootstrap_95_percentile_ci"], c["bootstrap_95_percentile_ci"]
        lines.append(f"| {name} | {100*f['mean_delta']:+.1f} [{100*fi[0]:+.1f}, {100*fi[1]:+.1f}] | "
            f"{f['discordant_0_to_1']} / {f['discordant_1_to_0']} | {100*c['mean_delta']:+.1f} [{100*ci[0]:+.1f}, {100*ci[1]:+.1f}] |")
    lines += ["", "工程筛选：`" + str(result["selection"]["selected_variant"]) + "`；" + result["selection"]["reason"] + "。", "",
        "配对区间为固定 20 场景的 10,000 次 percentile bootstrap（种子 20260928）。这是已查看开发集上的探索比较，未进行多重比较校正，不能当作论文独立测试结论。", "",
        "详细结果、失败角色分解、实际目标信念观察足迹和每场景配对见同目录 JSON。运动停滞和存活时长均不能替代有效覆盖；碰撞角色也不能单独证明碰撞原因。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--d1-mechanisms-passed", action="store_true")
    parser.add_argument("--serial-wall-time-comparable", action="store_true")
    args = parser.parse_args(argv)
    # The development runner owns source/input/child identity reconciliation.
    from .run_development import collect_completed
    plan = read_json(args.plan)
    identity, rows = collect_completed(args.run_dir, plan=plan)
    reviewed_plan_sha = identity["sources_before"]["inventory"]["files"][PLAN.relative_to(ROOT).as_posix()]
    if file_hash(args.plan) != reviewed_plan_sha:
        raise ValueError("analysis plan bytes differ from the source-frozen experiment plan")
    traces = {}
    for row in rows:
        index, variant = row["original_episode_index"], row["variant"]
        path = args.run_dir / f"scene_{index:04d}" / f"episode_{index:04d}" / variant / "step_trace.jsonl"
        traces[index, variant] = trace_summary(path)
        if traces[index, variant]["physical_steps"] != row["physical_steps"]:
            raise ValueError("trace and terminal step count disagree")
    if args.serial_wall_time_comparable and identity["workers"] != 1:
        raise ValueError("parallel worker timing cannot resolve registered serial wall-time tie-break")
    result = analyze_rows(rows, plan, traces=traces, d1_passed=args.d1_mechanisms_passed,
                          wall_time_comparable=args.serial_wall_time_comparable)
    result["sources"] = dict(run_directory=str(args.run_dir.resolve()),
        plan=str(args.plan.resolve()), plan_sha256=file_hash(args.plan),
        identity_sha256=file_hash(args.run_dir / "identity.json"),
        episodes_sha256=file_hash(args.run_dir / "episodes.json"),
        source_inventory_sha256=identity["sources_before"]["inventory"]["sha256"])
    output_directory(args.output_dir, (args.plan, args.run_dir / "episodes.json"))
    if args.output_dir.exists():
        raise FileExistsError("analysis requires a new output directory")
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "analysis.json", result)
    (args.output_dir / "analysis.md").write_text(markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
