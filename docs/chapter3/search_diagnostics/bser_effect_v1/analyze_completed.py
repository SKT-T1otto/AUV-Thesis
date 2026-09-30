"""Read-only audit of retained D results. Run from the repository root with -m."""
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

from chapter3_bser.experiments.bser_effect_v1 import run_windows as runner

ROOT = Path.cwd()
RUN = ROOT / "runs/bser_effect_v1/development_20260929_v1"
OUT = ROOT / "docs/chapter3/search_diagnostics/bser_effect_v1/results_20260929"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    return statistics.mean(values) if values else None


def exact_p(wins, losses):
    n = wins + losses
    return min(1., 2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.


def main():
    identity, rows, saved = [read(RUN/name) for name in ("identity.json", "episodes.json", "summary.json")]
    runner.verify_sources(identity["sources"])
    for path, sha in identity["input_sha256"].items():
        assert runner.file_hash(path) == sha, path
    expected = {(a,s["original_episode_index"]) for a in identity["arms"] for s in identity["selected"]}
    assert len(rows) == len(expected) == 80
    assert {(r["arm"],r["original_episode_index"]) for r in rows} == expected
    traces, audits = {}, {}
    for r in rows:
        key = r["arm"],r["original_episode_index"]
        scene = next(s for s in identity["selected"] if s["original_episode_index"] == key[1])
        directory = (RUN/r["artifact_dir"]).resolve()
        assert RUN in directory.parents
        checked = runner.read_completed(directory,identity,"D",key[0],scene)
        assert dict(checked,artifact_dir=r["artifact_dir"]) == r
        audits[key] = read(directory/"planning_audit.json")
        compact = []
        with (directory/"step_trace.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                t = json.loads(line)
                if not t["search_transition"]:
                    continue
                compact.append({k:t[k] for k in ("step_before","step_after","before","after","collision_records")})
        traces[key] = compact
    recomputed = runner.summarize(rows,identity)
    assert all(saved[k] == v for k,v in recomputed.items())
    assert saved["source_and_input_verification_passed"] is True
    groups = {}
    for arm in identity["arms"]:
        rs = [r for r in rows if r["arm"] == arm]
        # Independent arithmetic from individual results, not the summarizer.
        found = sum(r["found_step"] is not None for r in rs)
        collision = sum(r["pre_found_collision"] for r in rs)
        penalized = sum(r["found_step"] if r["found_step"] is not None else 400 for r in rs)/20
        assert saved["groups"][arm]["found_rate"] == found/20
        assert saved["groups"][arm]["pre_found_collision_rate"] == collision/20
        assert saved["groups"][arm]["penalized_found_steps_mean_400"] == penalized
        records = [v for r in rs for v in audits[arm,r["original_episode_index"]]]
        proposals = [v for v in records if v["kind"] == "proposal"]
        decisions = [v for v in records if v["kind"] == "decision"]
        safe = collections.Counter()
        for r in rs:
            safe.update(r["controller"]["safe_search"]["counts"])
        collision_windows = []
        for r in rs:
            if not r["pre_found_collision"]:
                continue
            key = arm,r["original_episode_index"]
            terminal = traces[key][-1]
            agents = r["episode_result"]["first_collision_agent_ids"]
            collision_windows.append(dict(index=key[1],step=terminal["step_after"],agents=agents,
                records=terminal["collision_records"],
                before=[a for a in terminal["before"]["agents"] if a["agent_id"] in agents],
                decision_reason=terminal["before"]["decision_reason"]))
        groups[arm] = dict(saved["groups"][arm],
            pre_found_collisions_while_hold=sum(all(a["hold"] for a in c["before"]) for c in collision_windows),
            surrogate_joint_minus_sequential_mean=mean([p["joint_score"]-p["sequential_joint_score"] for p in proposals]),
            surrogate_joint_below_sequential_count=sum(p["joint_score"] < p["sequential_joint_score"]-1e-12 for p in proposals),
            pre_found_collision_agent_counts=dict(collections.Counter(str(i) for r in rs if r["pre_found_collision"]
                for i in r["episode_result"]["first_collision_agent_ids"])),
            search_change_proposal_fraction=sum(v["search_differs_from_pure"] for v in proposals)/len(proposals),
            changed_search_scene_count=sum(any(v.get("search_differs_from_pure",False) for v in audits[arm,r["original_episode_index"]]) for r in rs),
            search_change_proposals_accepted_same_ids=sum(any(d["step"] == v["step"] and d["installed"]
                and d["selected_ids"] == v["selected_ids"] for d in decisions)
                for v in proposals if v["search_differs_from_pure"]),
            decision_count=len(decisions),accepted_decisions=sum(v["installed"] for v in decisions),
            stabilized_score_changed_count=sum(v["kind"] == "stabilized" and abs(v["score_before_recompute"]-v["objective"])>1e-9 for v in records),
            safe_counts=dict(safe),collision_windows=collision_windows,
            executor_standby_arrived_fraction=sum(r["controller"]["bser_effect_v1"]["executor_standby_arrived_steps"] for r in rs)/sum(r["pre_found_exposure_steps"] for r in rs))
    by = {(r["arm"],r["original_episode_index"]):r for r in rows}
    comparisons = {}
    for arm,base in (("D2","D0"),("D3","D2"),("D3","D0"),("D1","D0")):
        differences = []
        for index in sorted(r["original_episode_index"] for r in rows if r["arm"] == arm):
            a,b = by[arm,index],by[base,index]
            ta,tb = traces[arm,index],traces[base,index]
            initial_equal = ta[0]["before"]["agents"][:3] == tb[0]["before"]["agents"][:3]
            first = {}
            for x,y in zip(ta,tb):
                for kind,fields in (("search_motion",("position","velocity")),
                        ("search_guidance",("semantic_waypoint","tracking_waypoint","planned_path","hold"))):
                    if kind not in first and any(any(ax[f] != ay[f] for f in fields)
                        for ax,ay in zip(x["after"]["agents"][:3],y["after"]["agents"][:3])):
                        first[kind] = dict(step=x["step_after"],arm_reason=x["after"]["decision_reason"],base_reason=y["after"]["decision_reason"])
                if "public_map" not in first and any(x["after"][k] != y["after"][k] for k in ("cached_known_cells","cached_occupied_cells","last_full_refresh_step")):
                    first["public_map"] = dict(step=x["step_after"],arm_known=x["after"]["cached_known_cells"],base_known=y["after"]["cached_known_cells"],arm_refresh=x["after"]["last_full_refresh_step"],base_refresh=y["after"]["last_full_refresh_step"])
            differences.append(dict(index=index,arm_found=a["found_step"],base_found=b["found_step"],
                arm_pre_collision=a["pre_found_collision"],base_pre_collision=b["pre_found_collision"],
                arm_agents=a["episode_result"]["first_collision_agent_ids"] if a["pre_found_collision"] else [],
                initial_search_equal=initial_equal,first_divergences=first))
        cw=sum(not v["arm_pre_collision"] and v["base_pre_collision"] for v in differences)
        cl=sum(v["arm_pre_collision"] and not v["base_pre_collision"] for v in differences)
        comparisons[arm+"_vs_"+base] = dict(saved["paired"][arm+"_vs_"+base],
            collision_prevented_pairs=cw,collision_added_pairs=cl,collision_mcnemar_exact=exact_p(cw,cl),
            initial_search_equal_scenes=sum(v["initial_search_equal"] for v in differences),
            motion_divergent_scenes=sum("search_motion" in v["first_divergences"] for v in differences),
            map_diverged_before_search_motion=sum("search_motion" in v["first_divergences"] and
                v["first_divergences"].get("public_map",{}).get("step",10000)<v["first_divergences"]["search_motion"]["step"]
                for v in differences),scene_diagnostics=differences)
    guard=read(ROOT/"docs/chapter3/search_diagnostics/safe_search_v2/bser_reference_guard_20260929.json")
    assert all(runner.file_hash(ROOT/p)==sha for p,sha in guard["sha256"].items())
    result=dict(verified_at=datetime.datetime.now().astimezone().isoformat(),
        validation=dict(complete_episodes=80,paired_scenarios=20,physical_steps=sum(r["physical_steps"] for r in rows),
            every_trace_hash_and_terminal_checked=True,every_planning_audit_hash_checked=True,
            summaries_recomputed=True,headline_metrics_independently_recomputed=True,current_sources_and_inputs_match=True,
            retained_reference_hashes_unchanged=True,source_inventory_sha256=identity["sources"]["inventory"]["sha256"]),
        run_progress=read(RUN/"progress.json"),groups=groups,comparisons=comparisons,
        source_files={p:runner.file_hash(RUN/p) for p in ("identity.json","episodes.json","summary.json","progress.json")})
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result["validation"]))
    for arm,g in groups.items():
        print(arm,json.dumps({k:v for k,v in g.items() if k not in ("collision_windows",)}))
    for key,c in comparisons.items():
        print(key,json.dumps({k:v for k,v in c.items() if k not in ("pairs","scene_diagnostics","secondary_common_observed","statistics")}))
    print("SAVED",OUT/"analysis.json")


if __name__ == "__main__":
    main()
