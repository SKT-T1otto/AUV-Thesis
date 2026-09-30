"""Read-only V5=A+C development analysis against separately verified V0--V4.

V5 is a post-hoc development addition. This report cannot replace the original
registered V3 selection or establish performance on independent test scenes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chapter3_bser.experiments.safe_search_v1 import analyze_development as original
from chapter3_bser.experiments.safe_search_v1.run_development import INDICES, collect_completed
from chapter3_bser.experiments.safe_search_v1.run_paired import ROOT, VARIANTS, file_hash, read_json
from docs.chapter3.search_diagnostics.safe_search_v1 import development_trace_audit as windows
from docs.chapter3.search_diagnostics.safe_search_v1.development_baseline_audit import Inputs, key_outcome

AC_PLAN = ROOT / "docs/chapter3/search_diagnostics/safe_search_v1/ac_experiment_plan.json"
NEW_VARIANT = "V5"
BINARY_METRICS = {"found", "pre_found_collision", "success"}


def validate_new_rows(rows, reference_rows):
    """Independent terminal/metric consistency validation for exactly 20 V5 rows."""
    reference = {r["original_episode_index"]: r for r in reference_rows if r["variant"] == "V0"}
    if set(reference) != set(INDICES):
        raise ValueError("reference must cover the frozen 20 original indices")
    result = {}
    for row in rows:
        index = row["original_episode_index"]
        if index not in reference or index in result or row["variant"] != NEW_VARIANT:
            raise ValueError("missing, duplicate or unexpected V5 identity")
        old = reference[index]
        if any(row[name] != old[name] for name in
               ("baseline", "scenario_id", "scenario_seed", "environment_innovation_seed")):
            raise ValueError("V5 scenario, baseline or innovation seed differs from reference")
        if row.get("baseline") != "B0_search_prior" or not all(row.get(k) is True for k in
                ("terminal", "full_episode_completed", "full_episode_requested")):
            raise ValueError("V5 requires full terminal B0 episodes")
        step, found = row["physical_steps"], row["found_step"]
        if type(step) is not int or not 1 <= step <= 400:
            raise ValueError("invalid V5 terminal step")
        if found is not None and (type(found) is not int or not 0 <= found <= step):
            raise ValueError("invalid V5 Found step")
        if row.get("found_within_budget") is not (found is not None):
            raise ValueError("inconsistent V5 Found flag")
        episode, reason = row["episode_result"], row["stop_reason"]
        if (reason not in ("success", "timeout", "obstacle_collision")
                or episode.get("termination_reason") != reason
                or episode.get("terminal_step") != step
                or episode.get("task_protocol") != "collision_terminal_v1"):
            raise ValueError("V5 terminal task contract mismatch")
        if reason == "timeout" and step != 400:
            raise ValueError("a short prefix is not a V5 timeout")
        if episode.get("success") is not (reason == "success") or reason == "success" and found is None:
            raise ValueError("V5 success differs from terminal reason or Found")
        collision = episode.get("first_collision_step")
        if (reason == "obstacle_collision") != (collision is not None):
            raise ValueError("V5 collision metadata mismatch")
        if collision is not None:
            if type(collision) is not int or collision != step:
                raise ValueError("collision terminal step mismatch")
            original.collision_role(row)
        pre_collision = collision is not None and (found is None or collision <= found)
        if row.get("pre_found_collision") is not pre_collision:
            raise ValueError("V5 pre-Found collision precedence mismatch")
        exposure = found if found is not None else step
        if row.get("pre_found_exposure_steps") != exposure:
            raise ValueError("V5 search exposure differs from Found step")
        for name in ("searcher_motion_stall_proxy_agent_steps", "searcher_hold_agent_steps"):
            count = row[name]
            if type(count) is not int or not 0 <= count <= 3 * exposure:
                raise ValueError("V5 motion count outside exposure")
        if row["searcher_motion_stall_proxy_agent_steps"] + row["searcher_hold_agent_steps"] > 3 * exposure:
            raise ValueError("V5 disjoint stall/hold counts exceed exposure")
        coverage = row.get("search_coverage") or {}
        if coverage.get("available") is not True:
            raise ValueError("V5 requires measured actual belief-footprint coverage")
        result[index] = row
    if set(result) != set(reference):
        raise ValueError("V5 analysis requires all 20 complete scene pairs")
    return result


def episode_metrics(row, trace):
    """Per-scene rates, with undefined zero-exposure rates kept null."""
    exposure = row["pre_found_exposure_steps"]
    coverage = row["search_coverage"]
    queries = trace["pre_found_query_reasons"]
    rate = lambda value, scale=1: value / (scale * exposure) if exposure else None
    values = dict(found=int(row["found_step"] is not None), pre_found_collision=int(row["pre_found_collision"]),
        success=int(row["episode_result"]["success"]),
        restricted_found_time_400=row["found_step"] if row["found_step"] is not None else 400,
        pre_found_exposure_steps=exposure,
        stall_fraction=rate(row["searcher_motion_stall_proxy_agent_steps"], 3),
        hold_fraction=rate(row["searcher_hold_agent_steps"], 3),
        stall_plus_hold_fraction=rate(row["searcher_motion_stall_proxy_agent_steps"] + row["searcher_hold_agent_steps"], 3),
        effective_observation_steps=coverage["effective_observation_steps"],
        effective_observation_fraction=rate(coverage["effective_observation_steps"]),
        unique_observed_grid_fraction=coverage["unique_observed_grid_fraction"],
        new_cell_observations=coverage["new_cell_observations"],
        aged_revisit_cell_observations=coverage["aged_revisit_cell_observations"],
        repeated_cell_observations=coverage["repeated_cell_observations"],
        total_pre_found_queries=sum(queries.values()),
        queries_per_search_step=rate(sum(queries.values())))
    for reason in ("reachable", "no_start_connector", "invalid_start", "no_goal_connector", "invalid_goal", "disconnected_endpoint_components"):
        values["query_" + reason + "_per_search_step"] = rate(queries.get(reason, 0))
    return values


def paired_metrics(pairs, traces, *, replicates=10000, seed=20260928):
    comparisons, scenario_pairs = {}, []
    all_values = []
    for old, new in pairs:
        old_values = episode_metrics(old, traces[old["original_episode_index"], old["variant"]])
        new_values = episode_metrics(new, traces[new["original_episode_index"], new["variant"]])
        all_values.append((old["original_episode_index"], old_values, new_values))
        scenario_pairs.append(dict(original_episode_index=old["original_episode_index"], scenario_id=old["scenario_id"],
            reference_outcome=original.outcome(old), v5_outcome=original.outcome(new),
            reference_found_step=old["found_step"], v5_found_step=new["found_step"],
            found_delta=new_values["found"] - old_values["found"],
            pre_found_collision_delta=new_values["pre_found_collision"] - old_values["pre_found_collision"]))
    for metric in all_values[0][1]:
        complete = [(index, a[metric], b[metric]) for index, a, b in all_values
                    if a[metric] is not None and b[metric] is not None]
        excluded = [index for index, a, b in all_values if a[metric] is None or b[metric] is None]
        summary = (original.paired_bootstrap([b - a for _, a, b in complete], replicates, seed)
                   if complete else dict(mean_delta=None, bootstrap_95_percentile_ci=None, n_pairs=0,
                       bootstrap_replicates=replicates, bootstrap_seed=seed))
        summary.update(direction="V5 minus reference", n_planned_pairs=len(pairs),
            excluded_zero_exposure_original_indices=excluded)
        if metric in BINARY_METRICS:
            summary.update(discordant_0_to_1=sum(a == 0 and b == 1 for _, a, b in complete),
                discordant_1_to_0=sum(a == 1 and b == 0 for _, a, b in complete))
        comparisons[metric] = summary
    return dict(metrics=comparisons, scenario_pairs=scenario_pairs)


def validate_reference_analysis(analysis, summaries, reference_identity, reference_rows_path, identity_path):
    if (analysis.get("complete_stage") is not True or analysis.get("episode_count") != 100
            or analysis.get("scenario_count") != 20 or analysis.get("variants") != summaries
            or analysis.get("selection", {}).get("selected_variant") != "V3"):
        raise ValueError("original analysis does not match complete reference rows or registered V3 selection")
    sources = analysis.get("sources", {})
    if (sources.get("identity_sha256") != file_hash(identity_path)
            or sources.get("episodes_sha256") != file_hash(reference_rows_path)
            or sources.get("source_inventory_sha256") != reference_identity["sources_before"]["inventory"]["sha256"]):
        raise ValueError("original analysis provenance differs from the separate reference run")


def read_run_traces(inputs, run_dir, rows):
    """Reconcile actual traces against each terminal record without exporting rows."""
    result, window_summaries = {}, []
    for row in rows:
        index, variant = row["original_episode_index"], row["variant"]
        path = run_dir / f"scene_{index:04d}" / f"episode_{index:04d}" / variant / "step_trace.jsonl"
        inputs.watch(path)
        trace = list(windows.trace_rows(path))
        window_summaries.append(windows.audit_trace(trace, row))
        # Explicitly reconcile Hold, which is separate from the original stall proxy.
        hold = sum(bool(a["hold"]) for r in trace if r["search_transition"]
                   for a in r["before"]["agents"] if a["agent_id"] in (0, 1, 2))
        if hold != row["searcher_hold_agent_steps"]:
            raise ValueError("trace Hold count differs from terminal summary")
        result[index, variant] = original.trace_summary(path)
        if result[index, variant]["physical_steps"] != row["physical_steps"]:
            raise ValueError("trace differs from terminal physical length")
    return result, window_summaries


def watch_run(inputs, directory, variants):
    for name in ("identity.json", "summary.json", "episodes.json"):
        inputs.watch(directory / name)
    for index in INDICES:
        child = directory / f"scene_{index:04d}"
        for name in ("identity.json", "summary.json", "episodes.json", "evaluation_manifest.json", "resolved_config.json"):
            inputs.watch(child / name)
        for variant in variants:
            inputs.watch(child / f"episode_{index:04d}" / variant / "summary.json")


def validate_plan(plan):
    required = dict(schema="ch3.safe_search.ac_plan.v1", variant=NEW_VARIANT,
        options=dict(planning_state_consistency=True, failure_policy=False, path_safety=True),
        development_indices=list(INDICES), seed=12729, max_steps=400,
        primary_reference="V3", secondary_references=["V0", "V4"], additional_descriptive_references=["V1", "V2"],
        training=False, checkpoint_loaded=False, formal_thesis_evaluation=False, heldout_scenarios_executed=0)
    if any(plan.get(k) != value for k, value in required.items()):
        raise ValueError("unexpected A+C experiment plan")
    statistics = plan.get("statistics", {})
    if any(statistics.get(k) != v for k, v in dict(unit="paired_original_scenario", bootstrap_replicates=10000,
            bootstrap_seed=20260928, interval="percentile_95", multiplicity_adjustment=False).items()):
        raise ValueError("A+C statistics differ from the frozen protocol")
    gate = plan.get("development_gate", {})
    expected_gate = dict(comparator="V3", found_count_not_lower=True, pre_found_collision_count_not_higher=True,
        stall_plus_hold_fraction_strictly_lower=True, effective_observation_fraction_not_lower=True)
    if any(gate.get(k) != v for k, v in expected_gate.items()) or set(gate) - set(expected_gate) - {"note"}:
        raise ValueError("unknown or changed A+C development gate")
    control = plan.get("source_comparability", {})
    if (control.get("require_reviewed_exact_source_diff") is not True
            or control.get("require_all_selected_controls_equal") is not True
            or control.get("old_arm_replay_controls") != [dict(original_episode_index=3, variant="V3"),
                                                         dict(original_episode_index=28, variant="V4")]):
        raise ValueError("A+C requires both prespecified full old-arm replay controls")


def gate_result(summaries, *, source_comparability_passed):
    old, new = summaries["V3"], summaries[NEW_VARIANT]
    combined = lambda s: ((s["searcher_motion_stall_proxy_agent_steps"] + s["searcher_hold_agent_steps"])
                          / (3 * s["pre_found_exposure_steps"]) if s["pre_found_exposure_steps"] else None)
    old_combined, new_combined = combined(old), combined(new)
    old_effective = (old.get("actual_belief_footprint") or {}).get("effective_observation_fraction")
    new_effective = (new.get("actual_belief_footprint") or {}).get("effective_observation_fraction")
    checks = dict(source_comparability_passed=bool(source_comparability_passed),
        found_count_not_lower=new["found_count"] >= old["found_count"],
        pre_found_collision_count_not_higher=new["pre_found_collision_count"] <= old["pre_found_collision_count"],
        stall_plus_hold_fraction_strictly_lower=(new_combined is not None and old_combined is not None
            and new_combined < old_combined),
        effective_observation_fraction_not_lower=(new_effective is not None and old_effective is not None
            and new_effective >= old_effective))
    return dict(comparator="V3", variant=NEW_VARIANT, checks=checks, passed=all(checks.values()),
        values=dict(reference_stall_plus_hold_fraction=old_combined, v5_stall_plus_hold_fraction=new_combined,
            reference_effective_observation_fraction=old_effective, v5_effective_observation_fraction=new_effective),
        rate_aggregation="Pooled counts divided by pooled exposure, not the mean of per-episode rates.",
        meaning="Separate post-hoc A+C development hypothesis gate; does not replace the original registered V3 selection or prove statistical superiority.")


def verify_comparability(inputs, path, plan, new_identity, reference_identity, reference_rows):
    evidence = inputs.read(path)
    if (evidence.get("schema") != "ch3.safe_search.ac_source_comparability.v1"
            or evidence.get("passed") is not True or evidence.get("reviewed_exact_source_diff_passed") is not True
            or evidence.get("new_source_inventory_sha256") != new_identity["sources_before"]["inventory"]["sha256"]
            or evidence.get("reference_source_inventory_sha256") != reference_identity["sources_before"]["inventory"]["sha256"]):
        raise ValueError("missing or incompatible reviewed source-comparability evidence")
    required = [(c["original_episode_index"], c["variant"])
                for c in plan["source_comparability"]["old_arm_replay_controls"]]
    controls = evidence.get("controls", [])
    if sorted((c["original_episode_index"], c["variant"]) for c in controls) != sorted(required):
        raise ValueError("source comparison requires exactly the two frozen control identities")
    reference = {(r["original_episode_index"], r["variant"]): r for r in reference_rows}
    verified = []
    for control in controls:
        index, variant = control["original_episode_index"], control["variant"]
        directory = Path(control["replay_directory"]).resolve()
        if (directory / "failure.json").exists():
            raise ValueError("a source control replay has a program failure")
        identity, summary, rows = (inputs.read(directory / name) for name in
                                    ("identity.json", "summary.json", "episodes.json"))
        expected = next(s for s in new_identity["selected"] if s["original_episode_index"] == index)
        required_identity = dict(schema="ch3.safe_search.paired_identity.v1", variants=[variant], baseline="B0_search_prior",
            selected=[expected], full_episodes=True, task_horizon=400, steps=400, seed=12729,
            sources_before=new_identity["sources_before"], sources_after=new_identity["sources_before"],
            source_and_input_verification_passed=True, training=False, checkpoint_loaded=False,
            input_sha256=new_identity["child_input_sha256"])
        if any(identity.get(k) != value for k, value in required_identity.items()):
            raise ValueError("old-arm control identity/source/input mismatch")
        if (summary.get("all_requested_runs_recorded") is not True
                or summary.get("source_and_input_verification_passed") is not True or len(rows) != 1):
            raise ValueError("old-arm control did not complete")
        row = inputs.read(directory / f"episode_{index:04d}" / variant / "summary.json")
        old = reference[index, variant]
        if (rows != [row] or row.get("terminal") is not True or row.get("full_episode_completed") is not True
                or row.get("full_episode_requested") is not True or row.get("original_episode_index") != index
                or row.get("variant") != variant or row.get("signature_sha256") != old["signature_sha256"]
                or key_outcome(row["complete_episode_row"]) != key_outcome(old["complete_episode_row"])):
            raise ValueError("old-arm control full ordered signature or terminal outcome differs")
        verified.append(dict(original_episode_index=index, variant=variant, physical_steps=row["physical_steps"],
            ordered_signature_sha256=row["signature_sha256"], terminal_outcome_equal=True,
            identity_sha256=file_hash(directory / "identity.json"),
            summary_sha256=file_hash(directory / f"episode_{index:04d}" / variant / "summary.json")))
    return dict(passed=True, reviewed_exact_source_diff_passed=True, controls=verified,
        scope="Two prespecified complete controls reproduce old ordered physical/RNG signatures under the new source; this is not a replay of all 100 old arms.")


def analyze(run_dir, reference_dir, plan_path, reference_analysis_path, comparability_path):
    from chapter3_bser.experiments.safe_search_v1.run_ac_development import collect_completed_ac
    run_dir, reference_dir = Path(run_dir).resolve(), Path(reference_dir).resolve()
    plan_path, reference_analysis_path, comparability_path = (Path(p).resolve() for p in
        (plan_path, reference_analysis_path, comparability_path))
    inputs = Inputs()
    for path in (__file__, original.__file__, windows.__file__, original.PLAN,
            Path(__file__).with_name("development_baseline_audit.py"),
            collect_completed.__code__.co_filename, collect_completed_ac.__code__.co_filename,
            read_json.__code__.co_filename):
        inputs.watch(path)
    plan = inputs.read(plan_path)
    validate_plan(plan)
    watch_run(inputs, reference_dir, VARIANTS)
    watch_run(inputs, run_dir, [NEW_VARIANT])
    reference_identity, old_rows = collect_completed(reference_dir)
    original.validate_rows(old_rows, inputs.read(original.PLAN))
    new_identity, new_rows = collect_completed_ac(run_dir)
    new_indexed = validate_new_rows(new_rows, old_rows)
    if inputs.read(reference_dir / "episodes.json") != old_rows or inputs.read(run_dir / "episodes.json") != new_rows:
        raise ValueError("root episodes differ from validated child arms")
    for path, expected in ((reference_dir / "identity.json", plan["reference_identity_sha256"]),
            (reference_dir / "episodes.json", plan["reference_episodes_sha256"]),
            (reference_analysis_path, plan["reference_analysis_sha256"])):
        if inputs.watch(path) != expected:
            raise ValueError("reference artifact differs from the frozen A+C plan")
    if (reference_identity["sources_before"]["inventory"]["sha256"] != plan["reference_source_inventory_sha256"]
            or new_identity["input_sha256"].get(str(plan_path)) != file_hash(plan_path)):
        raise ValueError("reference source or A+C plan input hash mismatch")
    # Source inventories belong to their respective runs and intentionally may differ.
    # Input JSON semantics must remain the same across all paired child scenes.
    for index in INDICES:
        for name in ("resolved_config.json", "evaluation_manifest.json"):
            old = inputs.read(reference_dir / f"scene_{index:04d}" / name)
            new = inputs.read(run_dir / f"scene_{index:04d}" / name)
            if old != new:
                raise ValueError("V5 and reference manifest/config JSON differ")
    comparability = verify_comparability(inputs, comparability_path, plan, new_identity, reference_identity, old_rows)
    traces, old_windows = read_run_traces(inputs, reference_dir, old_rows)
    new_traces, new_windows = read_run_traces(inputs, run_dir, new_rows)
    traces.update(new_traces)
    summaries = {variant: original.aggregate([r for r in old_rows if r["variant"] == variant], traces)
                 for variant in VARIANTS}
    old_analysis = inputs.read(reference_analysis_path)
    validate_reference_analysis(old_analysis, summaries, reference_identity,
                                reference_dir / "episodes.json", reference_dir / "identity.json")
    summaries[NEW_VARIANT] = original.aggregate(new_rows, traces)
    all_rows = old_rows + new_rows
    indexed = {(r["original_episode_index"], r["variant"]): r for r in all_rows}
    comparisons = {variant: paired_metrics([(indexed[index, variant], new_indexed[index]) for index in INDICES], traces)
                   for variant in ("V3", "V0", "V4", "V1", "V2")}
    outcome_windows = {label: windows.aggregate([w for w in new_windows if w["outcome"] == label])
                       for label in sorted({w["outcome"] for w in new_windows})}
    gate = gate_result(summaries, source_comparability_passed=comparability["passed"])
    inputs.verify()
    return dict(schema="ch3.safe_search.ac_development_analysis.v1", complete_stage=True,
        new_variant=NEW_VARIANT, new_episode_count=20, reference_episode_count=100, independent_scene_count=20,
        primary_reference="V3", secondary_references=["V0", "V4"], additional_descriptive_references=["V1", "V2"],
        original_registered_selection=old_analysis["selection"], ac_development_gate=gate,
        variants=summaries, paired_v5_minus_reference=comparisons,
        v5_windows_by_outcome=outcome_windows, source_comparability=comparability,
        training=False, checkpoint_loaded=False, formal_thesis_evaluation=False, performance_passed=None,
        sources=dict(new_run_directory=str(run_dir), reference_run_directory=str(reference_dir),
            new_source_inventory_sha256=new_identity["sources_before"]["inventory"]["sha256"],
            reference_source_inventory_sha256=reference_identity["sources_before"]["inventory"]["sha256"],
            all_input_sha256=inputs.hashes, all_inputs_unchanged=True),
        definitions=dict(found="Any Found before terminal; Found step zero is valid. Success remains a distinct terminal task outcome.",
            collision="Collision at the Found step takes pre-Found precedence for the outcome partition.",
            restricted_found_time_400="Every non-Found episode, including early collisions, receives 400. Not actual survival time.",
            summary_rates="Pooled counts / pooled exposure; stall and Hold denominators are 3 searchers times search exposure.",
            paired_rate_intervals="Bootstrap means of paired per-episode rate differences, distinct from differences of pooled rates used by the gate. Undefined zero-exposure pairs are explicitly listed and omitted for that metric only.",
            footprint="Actual negative-belief timestamp evidence; effective means a step with a new cell or aged revisit. It is not target-detection probability or swept-space coverage.",
            query_units="Actual query calls, including repeated candidate enumeration; sample counts and unique physical failures are different units."),
        limitations=["V5 is a post-hoc development addition after inspection of V0--V4; the separate A+C gate does not replace the original registered V3 selection.",
            "The 20 partly diagnostic scenes are paired across variants. The 120 total evaluated arms are not 120 independent scenarios or a new formal test population.",
            "All paired percentile intervals use 10000 resamples, seed 20260928. The many metrics/comparisons are descriptive and not multiplicity-adjusted.",
            "Search exposure depends on Found and collision times; raw coverage, motion and query totals do not alone establish improved search efficiency.",
            "Outcome-stratified V5 windows diagnose survivors/failures and do not identify causal treatment effects.",
            "Parallel wall-clock durations are recorded in summaries but are not used by the A+C gate."])


def markdown(result):
    lines = ["# V5（A+C）固定 20 场景新增开发评估", "",
        "新 V5 的 20 个回合全部按原 400 步任务上限达到终止；旧 V0–V4 的 100 个参考回合独立验证。未训练、未加载模型。统计单位是同一组 20 个场景。", "",
        "V5 是查看旧结果后的新增开发组合。原注册工程选择仍为 **V3**，以下新门槛与原选择分开。", "",
        "| 变体 | Found | 成功 | 发现前碰撞 | 未发现超时 | 停滞 | Hold | 有效观察／曝光 | 截断发现时间 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in (*VARIANTS, NEW_VARIANT):
        s = result["variants"][variant]
        fraction = lambda value: "不可定义" if value is None else f"{100*value:.2f}%"
        lines.append(f"| {variant} | {s['found_count']}/20 | {s['success_count']}/20 | {s['pre_found_collision_count']}/20 | "
            f"{s['outcome_partition'].get('no_found_timeout', 0)}/20 | {fraction(s['searcher_motion_stall_proxy_fraction'])} | "
            f"{fraction(s['searcher_hold_fraction'])} | {fraction(s['actual_belief_footprint']['effective_observation_fraction'])} | {s['found_time_restricted_mean_400']:.2f} |")
    lines += ["", "停滞排除 Hold；二者以三个搜索者的实际发现前步数为分母。有效观察是新网格或间隔重访的证据更新，不能解释为发现概率。未发现回合的截断发现时间统一记为 400。", "",
        "| V5 减参考 | Found 差值及配对 95% 区间（百分点） | 新增／丢失 Found | 碰撞差值及区间（百分点） |",
        "|---|---:|---:|---:|"]
    for variant, comparison in result["paired_v5_minus_reference"].items():
        f, c = comparison["metrics"]["found"], comparison["metrics"]["pre_found_collision"]
        interval = lambda s: f"{100*s['mean_delta']:+.1f} [{100*s['bootstrap_95_percentile_ci'][0]:+.1f}, {100*s['bootstrap_95_percentile_ci'][1]:+.1f}]"
        lines.append(f"| {variant} | {interval(f)} | {f['discordant_0_to_1']}／{f['discordant_1_to_0']} | {interval(c)} |")
    gate = result["ac_development_gate"]
    lines += ["", "主比较是 V3，V0/V4 为次比较，V1/V2 为附加描述。10,000 次配对 bootstrap、种子 20260928；多项比较未经多重校正，不能作为独立测试或统计优越性证明。", "",
        "A+C 独立开发门槛：**" + ("通过" if gate["passed"] else "未通过") + "**。",
        "要求相对 V3：Found 不下降、发现前碰撞不增加、停滞加 Hold 占比严格下降、有效观察占比不下降，并通过源码对照。", ""]
    labels = dict(source_comparability_passed="新旧源码对照", found_count_not_lower="Found 不下降",
        pre_found_collision_count_not_higher="发现前碰撞不增加", stall_plus_hold_fraction_strictly_lower="停滞加 Hold 占比严格下降",
        effective_observation_fraction_not_lower="有效观察占比不下降")
    lines.extend(f"- {labels[k]}：{'通过' if value else '未通过'}" for k, value in gate["checks"].items())
    lines += ["", "门槛使用汇总计数除以汇总曝光。JSON 中各率的配对区间使用逐场景率差的均值，两种口径不能混写；零曝光导致不可定义的配对会明确列出。", "",
        "两个预设旧臂重放验证了完整有序物理签名和终止结局；不等于将旧 100 回合全部用新源码重跑。全部输入哈希、逐场景新增／丢失结果及其他指标区间见 [analysis.json](analysis.json)。", ""]
    return "\n".join(lines)


def output_directory(path, input_directories):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("V5 analysis requires a new output directory")
    protected = [*(Path(p).resolve() for p in input_directories),
        *(ROOT / name for name in ("outputs", "3090结果", ".git", "core", "chapter3_bser", "configs", "tests", "scripts", "tools"))]
    if any(path == p or p in path.parents for p in protected):
        raise ValueError("analysis output cannot overwrite retained inputs or production source")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "reference-dir", "reference-analysis", "comparability-json", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--plan", type=Path, default=AC_PLAN)
    args = parser.parse_args(argv)
    output = output_directory(args.output_dir, (args.run_dir, args.reference_dir))
    result = analyze(args.run_dir, args.reference_dir, args.plan, args.reference_analysis, args.comparability_json)
    output.mkdir(parents=True)
    with (output / "analysis.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    with (output / "analysis.md").open("x", encoding="utf-8") as handle:
        handle.write(markdown(result))


if __name__ == "__main__":
    main()
