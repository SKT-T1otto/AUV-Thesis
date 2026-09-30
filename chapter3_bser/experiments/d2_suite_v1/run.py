"""Manual execution gate, verified completed-job reuse, and immutable receipts."""
from pathlib import Path

from .plan import digest, load_plan, read, sha, write


def receipt(root, plan, job, stage):
    path = root / "receipts" / f"{job['name']}_{stage}.json"
    if not path.exists():
        return None
    value = read(path)
    if (value.get("plan_sha256") != plan["sha256"] or value.get("job") != job["name"]
            or value.get("stage") != stage or not value.get("complete")
            or value.get("sha256") != digest({k: v for k,v in value.items() if k != "sha256"})):
        raise ValueError("invalid completed-job receipt: " + str(path))
    directory = job["train" if stage == "train" else "evaluation"]
    required = {directory + "/summary.json", directory + "/episodes.json",
                directory + ("/config.json" if stage == "train" else "/identity.json")}
    if stage == "train":
        checkpoint = value.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint.startswith(directory + "/"):
            raise ValueError("training receipt is missing its own final checkpoint")
        required.add(checkpoint)
    elif value.get("checkpoint") is not None:
        raise ValueError("evaluation receipt cannot claim a training checkpoint")
    if set(value.get("artifacts", {})) != required:
        raise ValueError("completed-job artifact inventory mismatch")
    for name, expected in value["artifacts"].items():
        resolved = (root / name).resolve()
        if root not in resolved.parents or sha(resolved) != expected:
            raise ValueError("completed job artifact changed: " + name)
    return value


def save_receipt(root, plan, job, stage, checkpoint=None):
    directory = root / job["train" if stage == "train" else "evaluation"]
    names = [directory / "summary.json", directory / "episodes.json"]
    if stage == "evaluate":
        names.append(directory / "identity.json")
    else:
        names.append(directory / "config.json")
    if checkpoint:
        names.append(Path(checkpoint))
    value = dict(plan_sha256=plan["sha256"], job=job["name"], stage=stage, complete=True,
                 checkpoint=None if checkpoint is None else Path(checkpoint).relative_to(root).as_posix(),
                 artifacts={p.relative_to(root).as_posix(): sha(p) for p in names})
    value["sha256"] = digest(value)
    write(root / "receipts" / f"{job['name']}_{stage}.json", value)
    return value


def execute(root, *, stage="all", arms=None, seeds=None, execute=False):
    if stage not in ("all", "train", "evaluate"):
        raise ValueError("unknown execution stage")
    root = Path(root).resolve()
    plan = load_plan(root)
    selected = [j for j in plan["jobs"] if (not arms or j["arm"] in arms)
                and (j["training_seed"] is None or not seeds or j["training_seed"] in seeds)]
    if not selected or (arms and set(arms)-set(plan["settings"]["arms"])) or (seeds and set(seeds)-set(plan["settings"]["training_seeds"])):
        raise ValueError("unknown arm/seed or empty job selection")
    actions = [(j, s) for j in selected for s in ("train", "evaluate")
               if (stage == "all" or stage == s) and not (j["arm"] == "D2" and s == "train")]
    result = []
    for job, action in actions:
        done = receipt(root, plan, job, action)
        result.append(dict(job=job["name"], stage=action, status="verified_complete" if done else "planned"))
    if not execute:
        return dict(executed=False, jobs=result, note="No training/evaluation started. Add --execute to run manually.")
    for (job, action), item in zip(actions, result):
        if item["status"] == "verified_complete":
            print("verified complete; skip " + job["name"] + " " + action, flush=True)
            continue
        load_plan(root)
        print("starting " + job["name"] + " " + action, flush=True)
        if action == "train":
            config = read(root / job["config"])
            output = root / job["train"]
            if output.exists():
                raise FileExistsError("unfinished job is retained; prepare a new run instead of overwriting: " + str(output))
            if job["arm"] == "D2_B2":
                from chapter3_bser.experiments.baselines.direct_mc.train import DirectMCTrainer as Trainer
            elif job["arm"] == "D2_B3":
                from chapter3_bser.experiments.baselines.direct_boundary.train import DirectBoundaryTrainer as Trainer
            else:
                from chapter3_bser.experiments.hgr.train import Trainer
            summary = Trainer(config, output).run()
            if summary["actual_total_environment_steps"] < plan["settings"]["training_environment_steps"]:
                raise ValueError("training stopped before its declared environment-step budget")
            load_plan(root)
            save_receipt(root, plan, job, "train", Path(summary["latest_checkpoint"]).resolve())
        else:
            checkpoint = None
            if job["arm"] != "D2":
                training = receipt(root, plan, job, "train")
                if training is None:
                    raise ValueError("evaluation requires this job's verified final training receipt")
                checkpoint = root / training["checkpoint"]
            from .evaluate import evaluate
            summary = evaluate(root, plan, job, checkpoint)
            if not summary["evaluation_complete"]:
                raise ValueError("incomplete evaluation cannot receive a completion receipt")
            load_plan(root)
            save_receipt(root, plan, job, "evaluate")
        item["status"] = "complete"
    return dict(executed=True, jobs=result)
