"""Read-only audit of the user-run F development evaluation; no simulator calls.

Run from the repository root with python -B -m
docs.chapter3.search_diagnostics.bser_final_v1.analyze_completed.
Only derived files under results_20260930 are written.
"""
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

from chapter3_bser.experiments.bser_final_v1 import run_windows as runner

ROOT = Path.cwd()
RUN = ROOT / "runs/bser_final_v1/development_v1"
OUT = ROOT / "docs/chapter3/search_diagnostics/bser_final_v1/results_20260930"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    return statistics.mean(values) if values else None


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def signatures(t):
    state = t["after"]
    agents = state["agents"]
    groups = {
        "search_motion": [[a[k] for k in ("position", "velocity")] for a in agents[:3]],
        "search_guidance": [[a[k] for k in ("semantic_waypoint", "tracking_waypoint", "planned_path", "hold")] for a in agents[:3]],
        "executor_motion": [[a[k] for k in ("position", "velocity")] for a in agents[3:]],
        "public_map_counts": [state[k] for k in ("cached_known_cells", "cached_occupied_cells", "last_full_refresh_step")],
    }
    return {k: digest(v) for k, v in groups.items()}


def main():
    plan, rows, saved = [read(RUN / f) for f in ("identity.json", "episodes.json", "summary.json")]
    runner.verify_sources(plan["sources"])
    assert all(runner.file_hash(p) == sha for p, sha in plan["input_sha256"].items())
    guard = read(ROOT / "docs/chapter3/search_diagnostics/bser_final_v1/preservation_check.json")
    assert all(runner.file_hash(ROOT / p) == sha for p, sha in guard["retained_files_sha256"].items())
    assert plan["sources"]["inventory"]["sha256"] == guard["source_inventory_sha256"]
    expected = {(arm, s["original_episode_index"]) for arm in plan["arms"] for s in plan["selected"]}
    by = {(r["arm"], r["original_episode_index"]): r for r in rows}
    assert len(rows) == len(by) == len(expected) == 140 and set(by) == expected
    audits, traces, windows = {}, {}, {}
    for n, r in enumerate(rows, 1):
        key = r["arm"], r["original_episode_index"]
        scene = next(s for s in plan["selected"] if s["original_episode_index"] == key[1])
        directory = (RUN / r["artifact_dir"]).resolve()
        assert RUN in directory.parents
        checked = runner.read_completed(directory, plan, "F", key[0], scene)
        assert dict(checked, artifact_dir=r["artifact_dir"]) == r
        audits[key] = read(directory / "planning_audit.json")
        compact, tail, hold_runs = [], collections.deque(maxlen=5), {}
        with (directory / "step_trace.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                t = json.loads(line)
                if not t["search_transition"]:
                    continue
                for a in t["before"]["agents"]:
                    i = a["agent_id"]
                    if a["hold"]:
                        if i not in hold_runs:
                            hold_runs[i] = dict(start_step=t["step_before"], transitions=0,
                                initial_speed=a["speed"], initial_position=a["position"])
                        hold_runs[i]["transitions"] += 1
                    else:
                        hold_runs.pop(i, None)
                compact.append(dict(step=t["step_after"], signatures=signatures(t)))
                tail.append({k: t[k] for k in ("step_before", "step_after", "before", "after", "collision_records")})
        traces[key] = compact
        if r["pre_found_collision"]:
            ids = r["episode_result"]["first_collision_agent_ids"]
            windows[key] = dict(index=key[1], step=r["physical_steps"], agents=ids,
                consecutive_hold_before_collision={i: hold_runs.get(i) for i in ids},
                records=tail[-1]["collision_records"],
                last_five=[dict(step=t["step_before"], reason=t["before"]["decision_reason"],
                    agents=[a for a in t["before"]["agents"] if a["agent_id"] in ids]) for t in tail])
        if n % 20 == 0:
            print("Verified", n, "/140 terminal artifacts and traces", flush=True)
    recomputed = runner.summarize(rows, plan)
    assert all(saved[k] == v for k, v in recomputed.items())
    assert saved["source_and_input_verification_passed"] is True

    groups = {}
    for arm in plan["arms"]:
        rs = [r for r in rows if r["arm"] == arm]
        g = saved["groups"][arm]
        assert g["found_rate"] == sum(r["found_step"] is not None for r in rs) / 20
        assert g["pre_found_collision_rate"] == sum(r["pre_found_collision"] for r in rs) / 20
        assert g["penalized_found_steps_mean_400"] == mean([r["found_step"] if r["found_step"] is not None else 400 for r in rs])
        proposals, changes = [], []
        rejection, failures, fallback, counts, safe = [collections.Counter() for _ in range(5)]
        for r in rs:
            index = r["original_episode_index"]
            safe.update(r["controller"]["safe_search"]["counts"])
            records = audits[arm, index]["final"]
            for p in records:
                proposals.append(p)
                rejection.update(p.get("rejections", {}))
                failures.update(p.get("candidate_model_failures", {}))
                counts.update(p.get("counts", {}))
                if p.get("fallback_reason"):
                    fallback[p["fallback_reason"]] += 1
                if not p["selected_is_reference"]:
                    changes.append(dict(index=index, step=p["step"], trigger=p["trigger"],
                        installed=p.get("selected_assignment_installed", False),
                        search_changed=p["selected_ids"] != p["reference_ids"],
                        standby_changed=p["selected_standby"] != p["reference_standby"],
                        near_relative_gain=p["selected_score"]["near_search"] / p["reference_score"]["near_search"] - 1,
                        response_relative_gain=(p["selected_score"]["response"] / p["reference_score"]["response"] - 1)
                            if p["reference_score"]["response"] else None,
                        full_relative_gain=p["selected_score"]["full_search"] / p["reference_score"]["full_search"] - 1,
                        reference_score=p["reference_score"], selected_score=p["selected_score"]))
        if arm != "F0":
            assert len(changes) == g["computation"]["nonreference_proposals"]
            assert sum(c["installed"] for c in changes) == g["computation"]["installed_nonreference_proposals"]
            assert dict(fallback) == g["computation"]["fallback_counts"]
        cs = [w for (a, _), w in windows.items() if a == arm]
        hold = sum(all(a["hold"] for a in w["last_five"][-1]["agents"]) for w in cs)
        changed_indices = sorted({c["index"] for c in changes if c["installed"]})
        groups[arm] = dict(g, final_audit_proposals=len(proposals), changes=changes,
            installed_changed_scene_indices=changed_indices, guard_first_rejections=dict(rejection),
            candidate_model_failures=dict(failures), budget_counts=dict(counts),
            safe_search_counts=dict(safe), collision_windows=cs,
            pre_found_collisions_while_hold=hold,
            pre_found_collision_agent_counts=dict(collections.Counter(str(i) for w in cs for i in w["agents"])),
            coverage_unique_grid_fraction_mean=mean([r["search_coverage"]["unique_observed_grid_fraction"] for r in rs]))

    comparisons = {}
    compare = [("F1", "F0"), ("F1", "F2"), ("F2", "F0"), ("F1", "F3"), ("F1", "F4"), ("F1", "F5"), ("F1", "F6")]
    compare += [(a, "F0") for a in ("F3", "F4", "F5", "F6")]
    for arm, base in compare:
        differences = []
        for index in sorted(s["original_episode_index"] for s in plan["selected"]):
            a, b = by[arm, index], by[base, index]
            first = {}
            for x, y in zip(traces[arm, index], traces[base, index]):
                assert x["step"] == y["step"]
                for kind in x["signatures"]:
                    if kind not in first and x["signatures"][kind] != y["signatures"][kind]:
                        first[kind] = x["step"]
            differences.append(dict(index=index, arm_found=a["found_step"], base_found=b["found_step"],
                arm_pre_collision=a["pre_found_collision"], base_pre_collision=b["pre_found_collision"],
                arm_physical_steps=a["physical_steps"], base_physical_steps=b["physical_steps"],
                physical_signature_equal=a["signature_sha256"] == b["signature_sha256"],
                first_divergence_in_common_search_steps=first,
                common_search_steps=min(len(traces[arm,index]),len(traces[base,index]))))
        name = arm + "_vs_" + base
        comparisons[name] = dict(saved["paired"].get(name, {}), scene_diagnostics=differences,
            collision_prevented_pairs=sum(not d["arm_pre_collision"] and d["base_pre_collision"] for d in differences),
            collision_added_pairs=sum(d["arm_pre_collision"] and not d["base_pre_collision"] for d in differences))

    oldrows = read(ROOT / "runs/bser_effect_v1/development_20260929_v1/episodes.json")
    oldby = {r["original_episode_index"]: r for r in oldrows if r["arm"] == "D2"}
    reproduction = []
    keys = ("found_step", "pre_found_collision", "physical_steps", "episode_result", "signature_sha256",
            "search_coverage", "searcher_hold_agent_steps", "effective_search_steps")
    for index, b in oldby.items():
        a = by["F0", index]
        reproduction.append(dict(index=index, fields_equal={k: a[k] == b[k] for k in keys}))
    assert all(all(r["fields_equal"].values()) for r in reproduction)
    result = dict(verified_at=datetime.datetime.now().astimezone().isoformat(),
        validation=dict(complete_episodes=140, paired_scenarios=20,
            physical_steps=sum(r["physical_steps"] for r in rows),
            all_terminal_artifacts_traces_and_planning_hashes_checked=True,
            summaries_recomputed=True, headline_metrics_independently_recomputed=True,
            current_sources_and_inputs_match=True, retained_nine_reference_hashes_unchanged=True,
            F0_reproduces_D2_all_20_physical_signatures=True,
            source_inventory_sha256=plan["sources"]["inventory"]["sha256"]),
        run_progress=read(RUN / "progress.json"), groups=groups, comparisons=comparisons,
        F0_historical_D2_reproduction=reproduction,
        source_files={f: runner.file_hash(RUN/f) for f in ("identity.json", "episodes.json", "summary.json", "progress.json")})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result["validation"]), flush=True)
    print("SAVED", OUT / "analysis.json", flush=True)


if __name__ == "__main__":
    main()
