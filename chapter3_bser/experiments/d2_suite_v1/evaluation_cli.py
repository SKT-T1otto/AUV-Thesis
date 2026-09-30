"""Separate manual D2 reference and post-training evaluation entrypoints."""
import argparse
import json
from pathlib import Path

from . import linux
from .plan import ARMS, load_plan
from .run import receipt


def evaluate(root, *, scope, arms=None, seeds=None, confirmed=False):
    if scope == "d2":
        if arms is not None or seeds is not None:
            raise ValueError("D2 reference has no training arm or seed selection")
        selected = ["D2"]
    elif scope == "trained":
        selected = list(ARMS[1:]) if arms is None else arms
        if not selected or set(selected) - set(ARMS[1:]):
            raise ValueError("trained evaluation accepts only D2_B2, D2_B3, D2_HGR")
    else:
        raise ValueError("scope must be d2 or trained")
    root = Path(root).resolve()
    options = dict(stage="evaluate", arms=selected, seeds=seeds)
    preview = linux.launch(root, **options, confirmed=False)
    plan = load_plan(root)
    jobs = {job["name"]: job for job in plan["jobs"]}
    missing = []
    # Check every selected final training receipt before starting any evaluation.
    # No model construction, checkpoint restore or checkpoint selection here.
    for item in preview["jobs"]:
        job = jobs[item["job"]]
        if job["arm"] != "D2" and receipt(root, plan, job, "train") is None:
            missing.append(job["name"])
    readiness = dict(scope=scope, ready=not missing, missing_training_jobs=missing)
    if not confirmed:
        return dict(**preview, **readiness)
    if missing:
        raise ValueError("evaluation blocked; verified final training receipts required for: " + ", ".join(missing))
    return dict(**linux.launch(root, **options, confirmed=True), **readiness)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="scope", required=True)
    for scope in ("d2", "trained"):
        p = commands.add_parser(scope, help="preview only; --execute starts evaluation")
        p.add_argument("--root", required=True)
        p.add_argument("--execute", dest="confirmed", action="store_true")
        if scope == "trained":
            p.add_argument("--arms", nargs="+", choices=ARMS[1:])
            p.add_argument("--seeds", nargs="+", type=int)
    result = evaluate(**vars(parser.parse_args(argv)))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
