"""Read-only diagnosis of exported CH3 runs; stdlib only, no model loading.

Run with ``python -B -m scripts.analyze_search_failures --help``.
Counts are descriptive: training trajectories are not validation evaluations.
Missing motion traces remain missing; allocation counters cannot measure stalls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


SCHEMA = "ch3.search_failure_diagnosis.v1"
ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "ch3_baseline_search_prior": "B0", "ch3_basic_search_prior_v1": "B0",
    "ch3_baseline_bser_prior": "B1", "ch3_baseline_direct_mc": "B2",
    "ch3_baseline_direct_boundary": "B3", "ch3_hgr": "HGR",
}
BUCKETS = ("not_found_collision", "not_found_timeout", "found_success",
           "found_collision", "found_timeout")
DIAGNOSTIC_COUNTS = ("unreachable_search_queries", "candidate_shortage_events",
                     "partial_allocation_attempts", "partial_search_proposals")
UNAVAILABLE = {
    "stagnant_steps": "No step positions, route progress and sensing footprints in these episode exports.",
    "effective_search_steps": "Survival/exposure is not productive search time.",
    "search_query_failure_reason_breakdown": "B0 generator aggregates unreachable queries without preserving their causes.",
    "turn_inertia_collision_count": "No pre-collision velocity, route tangent and tracking-error sequence.",
    "obstacle_detection_to_braking_delay": "No synchronized sensing and control trace.",
}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"non-finite JSON {x}")))


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name}: expected finite integer")
    if value != int(value) or value < minimum:
        raise ValueError(f"{name}: invalid integer {value}")
    return int(value)


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(n=0, min=None, median=None, mean=None, max=None)
    return dict(n=len(values), min=values[0], median=statistics.median(values),
                mean=statistics.mean(values), max=values[-1])


def select_population(rows, summary):
    if not isinstance(rows, list) or not rows or not all(isinstance(x, dict) for x in rows):
        raise ValueError("episodes must be a non-empty list of objects")
    method = summary.get("method")
    if method not in LABELS:
        raise ValueError(f"unsupported method {method}")
    if "evaluation_complete" in summary:
        if summary["evaluation_complete"] is not True:
            raise ValueError("incomplete evaluation: no final rates may be reported")
        selected, scope = rows, "validation_evaluation"
        expected = integer(summary["n_expected_episodes"], "n_expected_episodes", 1)
    else:
        expected = integer(summary["completed_main_trajectories"], "completed_main_trajectories", 1)
        scope = "training_main" if expected >= 100 else "training_smoke"
        selected = [x for x in rows if x.get("purpose") == "main"] if method == "ch3_hgr" else rows
    if len(selected) != expected:
        raise ValueError(f"selected population {len(selected)} differs from recorded {expected}")
    if any(x.get("trajectory_complete", True) is not True or
           x.get("return_scope", "full_mission") != "full_mission" for x in selected):
        raise ValueError("selected population contains a partial trajectory")
    if any(x.get("method") != method for x in selected):
        raise ValueError("row/summary method mismatch")
    return selected, scope, dict(Counter(x.get("purpose", "unspecified") for x in rows)), len(rows)-len(selected)


def normalize_episode(row, horizon):
    for field in ("found", "success"):
        if type(row.get(field)) is not bool:
            raise ValueError(f"{field} must be a boolean")
    found, success = row["found"], row["success"]
    end = integer(row.get("terminal_step"), "terminal_step", 1)
    if end > horizon:
        raise ValueError("terminal_step exceeds horizon")
    if row.get("episode_terminated") is not True or row.get("episode_truncated") is not False:
        raise ValueError("requires complete strict-protocol episode")
    if row.get("task_protocol") != "collision_terminal_v1":
        raise ValueError("non-strict task protocol")
    reason = row.get("termination_reason")
    if reason not in {"timeout", "success", "obstacle_collision"}:
        raise ValueError(f"unsupported terminal reason {reason}")
    if success != (reason == "success") or (success and not found):
        raise ValueError("inconsistent success outcome")
    found_step = row.get("found_step")
    if found:
        found_step = integer(found_step, "found_step")
        if found_step > end:
            raise ValueError("found after terminal")
    elif found_step is not None:
        raise ValueError("not-found episode has a found_step")
    collision = reason == "obstacle_collision"
    if row.get("collision_episode") is not collision:
        raise ValueError("collision_episode inconsistent with reason")
    phase = row.get("first_collision_phase")
    ids = row.get("first_collision_agent_ids", [])
    if collision:
        if integer(row.get("first_collision_step"), "first_collision_step", 1) != end:
            raise ValueError("strict first collision is not terminal")
        if not ids or any(type(i) is not int or i not in range(4) for i in ids) or len(ids) != len(set(ids)):
            raise ValueError("invalid collision agent ids")
        if phase not in {"Search", "Intercept", "Hold"} or (phase == "Search") == found:
            raise ValueError("collision phase disagrees with discovery")
        if found and found_step >= end:
            raise ValueError("strict collision precedes detection on the same transition")
    elif row.get("first_collision_step") is not None or ids or phase is not None:
        raise ValueError("non-collision episode contains first collision data")
    if reason == "timeout" and end != horizon:
        raise ValueError("timeout before horizon")
    for field in ("actual_length", "episode_length", "episode_steps"):
        if field in row and integer(row[field], field, 1) != end:
            raise ValueError(f"{field} disagrees with terminal_step")
    bucket = ("found_" if found else "not_found_") + ("collision" if collision else reason)
    exposure = found_step if found else end
    identity = row.get("dataset_id") or f"{row['scenario_id']}|{row['scenario_seed']}"
    return dict(episode_key=identity, scenario_id=row["scenario_id"], scenario_seed=row["scenario_seed"],
                environment_innovation_seed=row.get("environment_innovation_seed"),
                bucket=bucket, found=found, success=success, found_step=found_step,
                terminal_step=end, first_collision_phase=phase, first_collision_agent_ids=ids,
                pre_found_exposure_steps=exposure,
                completed_collision_free_search_steps=exposure-int(bucket == "not_found_collision"),
                stagnant_steps=None, effective_search_steps=None)


def summarize_group(rows, diagnostics):
    counts = Counter(x["bucket"] for x in rows)
    n = len(rows)
    selected_diags = [diagnostics[x["scenario_id"]] for x in rows if x["scenario_id"] in diagnostics]
    allocation_counts = {}
    for key in DIAGNOSTIC_COUNTS:
        observed = [d["allocation_counts"][key] for d in selected_diags
                    if key in d.get("allocation_counts", {})]
        allocation_counts[key] = dict(observed_episodes=len(observed),
                                     total=sum(observed) if len(observed) == n and n else None)
    atomic = [d["allocation_reasons"].get("ATOMIC_REJECT_MISSING_SEARCH_ROUTE", 0)
              for d in selected_diags if "allocation_reasons" in d]
    lags = [x["terminal_step"]-max(diagnostics[x["scenario_id"]]["replan_steps"])
            for x in rows if diagnostics.get(x["scenario_id"], {}).get("replan_steps")]
    # Replan ages do NOT identify physical stagnation.
    return dict(n=n, buckets={k:counts[k] for k in BUCKETS},
                allocation_counts_whole_episode=allocation_counts,
                missing_route_atomic_rejections=dict(observed_episodes=len(atomic),
                                                     total=sum(atomic) if len(atomic) == n and n else None),
                steps_since_last_accepted_replan=distribution(lags),
                all_allocation_counts_are_pre_found=bool(n and all(not x["found"] for x in rows)))


def verify_summary(summary, rows):
    if "evaluation_complete" not in summary:
        return {"kind":"training summary: no evaluation-rate claim"}
    counts = Counter(x["bucket"] for x in rows)
    n, found = len(rows), sum(x["found"] for x in rows)
    expected = dict(n_valid_episodes=n, n_success=counts["found_success"],
                    n_collision_failure=counts["not_found_collision"]+counts["found_collision"],
                    n_timeout=counts["not_found_timeout"]+counts["found_timeout"], found_rate=found/n)
    for key, value in expected.items():
        if key not in summary or not math.isclose(summary[key], value, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"summary/episodes disagreement for {key}")
    return dict(kind="recomputed_from_episode_rows", checked_fields=expected)


def analyze_run(directory, source_root):
    directory, source_root = Path(directory), Path(source_root)
    summary = read_json(directory/"summary.json")
    raw = read_json(directory/"episodes.json")
    selected, scope, purposes, excluded = select_population(raw, summary)
    config_name = "resolved_config.json" if (directory/"resolved_config.json").exists() else "config.json"
    config = read_json(directory/config_name)
    conditions = config.get("common_task_conditions", config)
    horizon = integer(conditions["max_steps"], "max_steps", 1)
    rows = [normalize_episode(x, horizon) for x in selected]
    if len({x["episode_key"] for x in rows}) != len(rows):
        raise ValueError("duplicate episode identity in selected population")
    diagnostics = {}
    diag_path = directory/"controller_diagnostics.json"
    if diag_path.exists():
        for d in read_json(diag_path)["episodes"]:
            key = d["scenario_id"]
            if key in diagnostics:
                raise ValueError("duplicate diagnostic scenario_id")
            diagnostics[key] = d
        if set(diagnostics) != {x["scenario_id"] for x in rows}:
            raise ValueError("diagnostic/episode join is not one-to-one and complete")
        for row in rows:
            d = diagnostics[row["scenario_id"]]
            if d.get("found_step") != row["found_step"]:
                raise ValueError("diagnostic found_step mismatch")
            if any(integer(s,"replan step") > row["terminal_step"] for s in d.get("replan_steps", [])):
                raise ValueError("replan after terminal")
            for values in (d.get("allocation_counts", {}), d.get("allocation_reasons", {})):
                for name, value in values.items():
                    integer(value, name)
    manifest_path = directory/"evaluation_manifest.json"
    manifest_identity = None
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        scenarios = manifest["scenarios"]
        if [(x["scenario_id"],x["scenario_seed"]) for x in rows] != [
                (x["scenario_id"],x["scenario_seed"]) for x in scenarios]:
            raise ValueError("episode order/identities disagree with evaluation manifest")
        manifest_identity = dict(selected_content_sha256=manifest["selected_content_sha256"],
                                 actual_scenarios_sha256=digest(scenarios))
    n = len(rows)
    buckets = dict(Counter(x["bucket"] for x in rows))
    pre_collisions = [x for x in rows if x["bucket"] == "not_found_collision"]
    agents = Counter(i for x in pre_collisions for i in x["first_collision_agent_ids"])
    groups = {key:summarize_group([x for x in rows if x["bucket"] == key],diagnostics) for key in BUCKETS}
    source_paths = [directory/"summary.json",directory/"episodes.json",directory/config_name]
    source_paths += [p for p in (diag_path,manifest_path,directory/"identity.json") if p.exists()]
    source_files = {p.relative_to(source_root).as_posix():file_hash(p) for p in source_paths}
    checks = verify_summary(summary,rows)
    temporal = []
    for start in range(0, n, 25):
        part=rows[start:start+25]
        temporal.append(dict(first_selected_index=start,last_selected_index=start+len(part)-1,n=len(part),
                             found=sum(x["found"] for x in part),buckets=dict(Counter(x["bucket"] for x in part))))
    result = dict(run=directory.relative_to(source_root).as_posix(), method=summary["method"],
                  label=LABELS[summary["method"]], population=scope, n=n,
                  raw_episode_rows=len(raw), excluded_non_main_rows=excluded, raw_purposes=purposes,
                  horizon=horizon,buckets={k:buckets.get(k,0) for k in BUCKETS},
                  found_rate=sum(x["found"] for x in rows)/n,
                  pre_found_collision_rate=len(pre_collisions)/n,
                  pre_found_collision_agent_counts={str(k):v for k,v in sorted(agents.items())},
                  pre_found_collision_searcher_episodes=sum(any(i<3 for i in x["first_collision_agent_ids"]) for x in pre_collisions),
                  pre_found_collision_executor_episodes=sum(3 in x["first_collision_agent_ids"] for x in pre_collisions),
                  pre_found_collision_step=distribution([x["terminal_step"] for x in pre_collisions]),
                  pre_found_collision_by_step_bin={f"{lo}-{hi}":sum(lo<=x["terminal_step"]<=hi for x in pre_collisions)
                                                  for lo,hi in ((1,20),(21,50),(51,100),(101,200),(201,horizon))},
                  found_step=distribution([x["found_step"] for x in rows if x["found"]]),
                  pre_found_exposure_steps=sum(x["pre_found_exposure_steps"] for x in rows),
                  completed_collision_free_search_steps=sum(x["completed_collision_free_search_steps"] for x in rows),
                  effective_search_steps=None, stagnant_steps=None, missing_measurements=UNAVAILABLE,
                  groups=groups, descriptive_blocks_of_25=temporal,
                  source_files=source_files, summary_checks=checks, manifest=manifest_identity,
                  comparable_inputs_sha256=config.get("comparable_inputs_sha256"),
                  recorded_production_sha256=summary.get("source_sha256"),
                  rows=rows)
    identity_path = directory/"identity.json"
    if identity_path.exists():
        identity=read_json(identity_path)
        before=identity["sources_before"]["production"]
        if before != identity["sources_after"]["production"]:
            raise ValueError("source identity changed during recorded evaluation")
        result["recorded_production_sha256"]=before["sha256"]
        result["current_source_comparison"]={
            "matching_recorded_files":sum((ROOT/name).exists() and file_hash(ROOT/name)==value for name,value in before["files"].items()),
            "recorded_file_count":len(before["files"]),
            "changed_or_missing_recorded_files":[name for name,value in before["files"].items()
                                                if not (ROOT/name).exists() or file_hash(ROOT/name)!=value],
            "meaning":"Byte comparison only; no historical replay or provenance exemption."}
    return result


def paired_evaluations(runs):
    pairs=[]
    evaluations=[r for r in runs if r["population"]=="validation_evaluation"]
    for i,left in enumerate(evaluations):
        for right in evaluations[i+1:]:
            compatible=bool(left["comparable_inputs_sha256"] and
                            left["comparable_inputs_sha256"]==right["comparable_inputs_sha256"] and
                            left["manifest"] and left["manifest"]==right["manifest"] and
                            left["recorded_production_sha256"]==right["recorded_production_sha256"])
            item=dict(left=left["label"],right=right["label"],comparable_task_scene_inputs=compatible,
                      identical_exogenous_trajectory_events_guaranteed=False)
            if compatible:
                a={r["episode_key"]:r for r in left["rows"]};b={r["episode_key"]:r for r in right["rows"]}
                if a.keys()!=b.keys():raise ValueError("paired evaluation population mismatch")
                if any(a[k]["environment_innovation_seed"]!=b[k]["environment_innovation_seed"] for k in a):
                    raise ValueError("paired environment seed mismatch")
                item["found_transitions"]=dict(Counter(f'{int(a[k]["found"])}->{int(b[k]["found"])}' for k in a))
                item["outcome_transitions"]=dict(Counter(a[k]["bucket"]+" -> "+b[k]["bucket"] for k in a))
            pairs.append(item)
    return pairs


def analyze(source_root):
    source_root=Path(source_root).resolve()
    candidates=[p.parent for p in source_root.rglob("summary.json") if (p.parent/"episodes.json").exists()]
    seen, runs, duplicates={},[],[]
    # Prefer a named run over a loose copy in the export root.
    for directory in sorted(candidates,key=lambda p:(-len(p.relative_to(source_root).parts),p.as_posix())):
        summary=read_json(directory/"summary.json")
        if summary.get("method") not in LABELS:continue
        cfg=directory/("resolved_config.json" if (directory/"resolved_config.json").exists() else "config.json")
        fingerprint=tuple(file_hash(p) for p in (directory/"summary.json",directory/"episodes.json",cfg))
        if fingerprint in seen:
            duplicates.append(dict(duplicate=directory.relative_to(source_root).as_posix(),
                                   canonical=seen[fingerprint],reason="identical summary, episodes and config bytes"))
            continue
        run=analyze_run(directory,source_root);seen[fingerprint]=run["run"];runs.append(run)
    if not runs:raise ValueError("no supported runs found")
    runs.sort(key=lambda x:(x["label"],x["population"],x["run"]))
    return dict(schema=SCHEMA,source_root=str(source_root),runs=runs,duplicates=duplicates,
                paired_evaluations=paired_evaluations(runs),
                interpretation_limits=[
                    "B0/B1 validation evaluations must not be ranked against B2/B3/HGR changing-policy training returns.",
                    "HGR main only; auxiliary and prefix-only trajectories are not additional main episodes.",
                    "Discovery and collision are competing events; more exposure alone is not better performance.",
                    "Whole-episode allocation counters for found episodes include the post-found period.",
                    "Terminal-segment geometry cannot identify inertia or obstacle-discovery timing.",
                    "Only current-state traces and bounded mechanism checks can attribute specific query failures."])


def markdown(report):
    lines=["# 搜索失效诊断（导入结果，非新实验）", "",
           "B0/B1 为验证评价；B2/B3/HGR 为训练主轨迹，不能据此做五方法性能排名。", "",
           "| 方法 | 数据用途 | N | Found | 发现前碰撞 | 未发现超时 | 发现后成功 | 发现后碰撞 | 发现后超时 |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in report["runs"]:
        b=r["buckets"]
        lines.append(f'| {r["label"]} | {r["population"]} | {r["n"]} | {r["found_rate"]:.1%} | '
                     +" | ".join(str(b[k]) for k in BUCKETS)+" |")
    lines += ["", "## 可确认的现象", ""]
    for r in report["runs"]:
        if r["population"]=="training_smoke":continue
        b=r["buckets"];g=r["groups"]["not_found_timeout"]
        lines += [f'- {r["label"]}：未发现 {b["not_found_collision"]+b["not_found_timeout"]} 回合，其中 '
                  f'{b["not_found_collision"]} 回合发现前碰撞、{b["not_found_timeout"]} 回合跑满时限。'
                  f'发现前碰撞步数中位数 {r["pre_found_collision_step"]["median"]}；'
                  f'执行者参与 {r["pre_found_collision_executor_episodes"]} 回合。']
        count=g["allocation_counts_whole_episode"]["unreachable_search_queries"]["total"]
        if count is not None:
            lines += [f'  未发现超时组累计不可达查询 {count:,} 次；缺少路径导致部分分配拒绝 '
                      f'{g["missing_route_atomic_rejections"]["total"]:,} 次。查询次数不是不可达区域数。']
    lines += ["", "## 当前不能由旧日志确定", "",
              "- 停滞步数、有效搜索时间、碰撞前速度/转角/横向误差、首次见障到减速延迟：均缺少逐步证据，保留 null。",
              "- B0 的搜索不可达计数丢失了具体原因；不能把 anchor 路径的 no_start_connector 当作搜索查询原因。",
              "- 长时间未接受重规划不等于智能体不动；运行更久也不等于有效搜索更多。",
              "- 不同源码版本、训练场景和随机流不能冒充同条件历史重放。", "",
              "## 统计与复现", "",
              "- 逐回合重算汇总，检查严格碰撞终止、Found 时序、时限和唯一身份。",
              "- 诊断按 scenario_id 一对一关联；HGR 用 dataset_id 标识主轨迹，不能仅用重复的 scenario_id。",
              "- 暴露步数到 Found/碰撞/超时为止；完整无碰撞搜索步数扣除碰撞转移，均不冒充有效搜索时间。",
              "- duplicates 记录重复导出；每个采用的输入文件保存 SHA-256。",
              "- 本工具不加载模型、不推进环境、不改写来源文件。", ""]
    return "\n".join(lines)


def write_report(report, output):
    output=Path(output).resolve();source=Path(report["source_root"]).resolve()
    if output==source or source in output.parents:
        raise ValueError("diagnostic output must be outside the retained source tree")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new or empty; previous evidence is never overwritten")
    output.mkdir(parents=True,exist_ok=True)
    flat=[]
    for r in report["runs"]:
        flat.extend(dict(run=r["run"],method=r["label"],population=r["population"],**x) for x in r["rows"])
    compact=dict(report,runs=[{k:v for k,v in r.items() if k!="rows"} for r in report["runs"]])
    (output/"diagnosis.json").write_text(json.dumps(compact,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (output/"report.md").write_text(markdown(report),encoding="utf-8")
    with (output/"episode_diagnosis.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,required=True,help="Export collision_terminal directory, excluding legacy runs")
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    report=analyze(args.source_root);write_report(report,args.output_dir)
    print(json.dumps(dict(runs=len(report["runs"]),duplicates=len(report["duplicates"]),output=str(args.output_dir)),ensure_ascii=True))


if __name__=="__main__":main()
