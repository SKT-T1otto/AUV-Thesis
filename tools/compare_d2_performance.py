"""Fail closed on any semantic difference in two bounded D2 benchmarks."""
import argparse
import json
from pathlib import Path


def semantic_summary(value):
    if isinstance(value, dict):
        return {k: semantic_summary(v) for k, v in value.items() if k != "wall_seconds"}
    if isinstance(value, list):
        return [semantic_summary(v) for v in value]
    return value


def compare(before, after):
    errors, cases = [], []
    for label, value in (("before", before), ("after", after)):
        if value.get("schema") != "ch3.d2.performance.benchmark.v1" or value.get("formal_experiment") is not False:
            errors.append(label + ".schema")
        if any(value.get(field) is None for field in ("arm", "seed", "threads", "device", "manifest_sha256", "torch")):
            errors.append(label + ".missing_metadata")
        if not any(r["name"] in ("training_episode", "episode_after_5_warmup") for r in value["cases"]):
            errors.append(label + ".no_completed_episode")
        if any(r["name"] == "training_episode" for r in value["cases"]) and any(
                value.get(field) is None for field in ("global_step", "optimizer_updates")):
            errors.append(label + ".missing_training_counters")
        keys = [(r["name"], r.get("scene"), r.get("episode")) for r in value["cases"]]
        if len(keys) != len(set(keys)):
            errors.append(label + ".duplicate_cases")
        for row in value["cases"]:
            if row["name"] in ("training_episode", "episode_after_5_warmup"):
                required = {"transition_sha256", "rewards", "summary"}
                if row["name"] == "training_episode":
                    required.add("model_sha256")
                if any(row.get(field) is None for field in required):
                    errors.append(label + ".missing_signature")
    for key in ("arm", "seed", "threads", "device", "manifest_sha256", "torch"):
        if before.get(key) != after.get(key):
            errors.append("metadata." + key)
    for key in ("global_step", "optimizer_updates"):
        if before.get(key) != after.get(key):
            errors.append(key)
    # Full-episode timing was added in round two. It carries no new semantic
    # signature; the existing completed-rollout case includes all transitions.
    def indexed(value):
        return {(r["name"], r.get("scene"), r.get("episode")): r
                for r in value["cases"] if r["name"] != "full_episode"}
    left, right = indexed(before), indexed(after)
    if left.keys() != right.keys():
        errors.append("case_inventory")
    for key in left.keys() & right.keys():
        a, b = left[key], right[key]
        failures = []
        for field in ("steps", "actual_length", "transition_sha256", "rewards", "model_sha256", "summary"):
            av, bv = a.get(field), b.get(field)
            if field == "summary":
                av, bv = semantic_summary(av), semantic_summary(bv)
            if av != bv:
                failures.append(field)
        for label in ("B_env_step", "C_snapshot", "F_controller", "E_build_planning_graph", "H_route_query", "H_astar"):
            if a.get("timings", {}).get(label, {}).get("calls", 0) != b.get("timings", {}).get(label, {}).get("calls", 0):
                failures.append(label + ".calls")
        errors.extend(f"{key}: {field}" for field in failures)
        cases.append(dict(name=key[0], scene=key[1], episode=key[2], exact=not failures,
                          differences=failures, transition_sha256=b.get("transition_sha256"),
                          model_sha256=b.get("model_sha256"),
                          before_seconds=a["wall_seconds"], after_seconds=b["wall_seconds"],
                          speedup=a["wall_seconds"] / b["wall_seconds"]))
    training = [r for r in cases if r["name"] == "training_episode"]
    result = dict(exact_passed=not errors, errors=errors, cases=sorted(cases, key=lambda r: str((r['name'], r['scene'], r['episode']))),
                  bounded_training_verified=bool(training) and all(r["model_sha256"] is not None for r in training),
                  global_step=after.get("global_step"), optimizer_updates=after.get("optimizer_updates"))
    for name in ("episode_after_5_warmup", "training_episode"):
        rows = [r for r in cases if r["name"] == name]
        if rows:
            old, new = sum(r["before_seconds"] for r in rows), sum(r["after_seconds"] for r in rows)
            result[name] = dict(before_seconds=old, after_seconds=new, speedup=old / new,
                                throughput_increase_percent=100 * (old / new - 1),
                                wall_reduction_percent=100 * (1 - new / old))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-training", action="store_true")
    args = parser.parse_args()
    result = compare(json.loads(args.before.read_text(encoding="utf-8")),
                     json.loads(args.after.read_text(encoding="utf-8")))
    if args.require_training and not result["bounded_training_verified"]:
        result["errors"].append("bounded_training_missing")
        result["exact_passed"] = False
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["exact_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
