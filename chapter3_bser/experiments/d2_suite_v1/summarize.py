"""Strict paired descriptive comparisons; partial jobs never become failures."""
import csv
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev

from .plan import ARMS, load_plan, read, write
from .run import receipt

METRICS = ("found_rate", "pre_found_collision_rate", "found_penalized_steps",
           "found_steps_conditional", "pre_found_steps", "pre_found_stagnant_steps",
           "pre_found_moving_steps", "collision_rate", "success_rate")


def metrics(rows):
    found = [r["found_step"] for r in rows if r["found"]]
    return dict(found_rate=mean(int(r["found"]) for r in rows),
                pre_found_collision_rate=mean(int(r["pre_found_collision"]) for r in rows),
                found_penalized_steps=mean(r["found_penalized_steps"] for r in rows),
                found_steps_conditional=mean(found) if found else None,
                pre_found_steps=mean(r["pre_found_steps"] for r in rows),
                pre_found_stagnant_steps=mean(r["pre_found_stagnant_steps"] for r in rows),
                pre_found_moving_steps=mean(r["pre_found_moving_steps"] for r in rows),
                collision_rate=mean(int(r["collision_episode"]) for r in rows),
                success_rate=mean(int(r["success"]) for r in rows))


def paired(left, right):
    key = lambda row: (row["scenario_id"], row["scenario_seed"], row["environment_innovation_seed"])
    if len({key(r) for r in left}) != len(left) or [key(r) for r in left] != [key(r) for r in right]:
        raise ValueError("paired rows differ in scenario order/seed or contain duplicates")
    both = [(a, b) for a, b in zip(left, right) if a["found"] and b["found"]]
    return dict(n_pairs=len(left), right_only_found=sum(not a["found"] and b["found"] for a,b in zip(left,right)),
                left_only_found=sum(a["found"] and not b["found"] for a,b in zip(left,right)),
                both_found=len(both),
                found_rate_delta=mean(int(b["found"])-int(a["found"]) for a,b in zip(left,right)),
                pre_found_collision_rate_delta=mean(int(b["pre_found_collision"])-int(a["pre_found_collision"]) for a,b in zip(left,right)),
                found_penalized_steps_delta=mean(b["found_penalized_steps"]-a["found_penalized_steps"] for a,b in zip(left,right)),
                both_found_steps_delta=mean(b["found_step"]-a["found_step"] for a,b in both) if both else None)


def summarize(root, output):
    from chapter3_bser.experiments.phase1c_prrac.task_metrics import validated_rows
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("summary requires a new directory: " + str(output))
    # Summary stays readable after source upgrades, but still checks plan/assets
    # and every completed receipt's raw result/checkpoint bytes.
    plan = load_plan(root, verify_source=False)
    scenes = read(root / "inputs/evaluation.json")["scenarios"]
    settings = plan["settings"]
    records, episodes, costs = [], {}, []
    for job in plan["jobs"]:
        record = dict(job=job["name"], arm=job["arm"], training_seed=job["training_seed"], complete=False)
        done = receipt(root, plan, job, "evaluate")
        if job["arm"] != "D2":
            trained = receipt(root, plan, job, "train")
            if trained:
                cost = read(root / job["train"] / "summary.json")
                actual = cost["actual_total_environment_steps"]
                costs.append(dict(job=job["name"], actual_environment_steps=actual,
                      budget_environment_steps=settings["training_environment_steps"],
                      overshoot_steps=max(0, actual-settings["training_environment_steps"]),
                      completed_main_trajectories=cost["completed_main_trajectories"],
                      wall_seconds=cost["wall_seconds"], native_costs=cost.get("costs")))
            elif done:
                raise ValueError("learned evaluation has no verified training receipt")
        if done:
            identity = read(root / job["evaluation"] / "identity.json")
            mode = ("zero_residual" if job["arm"] == "D2" else
                    settings["hgr_policy_mode"] if job["arm"] == "D2_HGR" else "deterministic")
            checkpoint_sha = None if job["arm"] == "D2" else trained["artifacts"][trained["checkpoint"]]
            if (identity["plan_sha256"] != plan["sha256"] or identity["arm"] != job["arm"]
                    or identity["training_seed"] != job["training_seed"]
                    or identity["source_sha256"] != plan["source_sha256"]
                    or identity["policy_mode"] != mode or identity["checkpoint_sha256"] != checkpoint_sha
                    or identity["evaluation_seed"] != settings["evaluation_seed"]
                    or identity["optimizer_updates"] != 0
                    or identity["scenario_sha256"] != plan["assets"]["inputs/evaluation.json"]):
                raise ValueError("evaluation identity mismatch")
            rows = validated_rows(read(root / job["evaluation"] / "episodes.json"), require_complete=True)
            if len(rows) != len(scenes) or not read(root / job["evaluation"] / "summary.json")["evaluation_complete"]:
                raise ValueError("receipt claims completion for an incomplete evaluation")
            for index, (row, scene) in enumerate(zip(rows, scenes)):
                if (row["scenario_id"] != scene["scenario_id"] or row["scenario_seed"] != scene["scenario_seed"]
                        or row["environment_innovation_seed"] != settings["evaluation_seed"]+index
                        or row["plan_sha256"] != plan["sha256"] or row["arm"] != job["arm"]
                        or row["training_seed"] != job["training_seed"]):
                    raise ValueError("evaluation row pairing identity mismatch")
            record.update(complete=True, episodes=len(rows), **metrics(rows))
            episodes[job["name"]] = rows
        records.append(record)
    groups = []
    for arm in ARMS:
        values = [r for r in records if r["arm"] == arm]
        complete = all(r["complete"] for r in values)
        group = dict(arm=arm, complete=complete, expected_replicates=len(values),
                     completed_replicates=sum(r["complete"] for r in values))
        for metric in METRICS:
            numbers = [r[metric] for r in values] if complete else []
            valid = numbers and all(n is not None for n in numbers)
            group[metric] = mean(numbers) if valid else None
            group[metric+"_seed_sd"] = stdev(numbers) if valid and len(numbers)>1 else None
        groups.append(group)
    pairs = []
    for left, right in combinations(ARMS, 2):
        for seed in settings["training_seeds"]:
            a, b = ("D2" if left == "D2" else f"{left}_seed{seed}"), f"{right}_seed{seed}"
            if a in episodes and b in episodes:
                pairs.append(dict(left=left, right=right, training_seed=seed, **paired(episodes[a], episodes[b])))
    result = dict(schema="ch3.d2_suite.summary.v1", plan_sha256=plan["sha256"],
                  suite_complete=all(r["complete"] for r in records), jobs=records, groups=groups,
                  paired_comparisons=pairs, training_costs=costs,
                  interpretation=["Primary: Found, pre-Found collisions, failure-penalized Found steps (failure=400).",
                    "Conditional Found steps exclude failures; use paired both-found steps only as secondary context.",
                    "Moving search steps are a kinematic proxy, not measured useful coverage. Collision ties count as pre-Found.",
                    "D2 is evaluated once; reusing it for paired comparisons does not create independent replicates.",
                    "Seed SD is between training seeds on a shared scenario set, not a confidence interval.",
                    "Equal start budgets permit complete episode/cycle overshoot; compare actual costs as well.",
                    "Incomplete jobs have no rates and are never counted as failed missions.", plan["heldout_claim"]])
    output.mkdir(parents=True)
    write(output / "summary.json", result)
    for filename, data in (("groups.csv", groups), ("jobs.csv", records), ("paired.csv", pairs),
                           ("training_costs.csv", costs)):
        keys = list(dict.fromkeys(k for row in data for k in row))
        with (output / filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
    def show(value):
        return "pending" if value is None else f"{value:.4f}"
    lines = ["# D2 four-arm experiment", "", f"Complete: {result['suite_complete']}", "",
             "| Arm | Seeds complete | Found | Pre-Found collision | Penalized Found steps | Success |",
             "|---|---:|---:|---:|---:|---:|"]
    for g in groups:
        lines.append(f"| {g['arm']} | {g['completed_replicates']}/{g['expected_replicates']} | " +
                     " | ".join(show(g[k]) for k in ("found_rate", "pre_found_collision_rate", "found_penalized_steps", "success_rate"))+" |")
    lines += ["", *["- " + s for s in result["interpretation"]]]
    (output / "report.md").write_bytes(("\n".join(lines)+"\n").encode())
    return dict(output=str(output), suite_complete=result["suite_complete"], groups=groups)
