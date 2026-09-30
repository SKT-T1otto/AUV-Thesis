"""Prepare immutable experiment inputs without constructing a trainer."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT / "configs/chapter3/d2_suite_v1/experiment.json"
ARMS = ("D2", "D2_B2", "D2_B3", "D2_HGR")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def sha(path):
    hashed = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hashed.update(block)
    return hashed.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n").encode())
    temporary.replace(path)


def settings(path=DEFAULT):
    value = read(path)
    if value.get("schema") != "ch3.d2_suite.v1" or value.get("arms") != list(ARMS):
        raise ValueError("suite requires exactly D2, D2_B2, D2_B3, D2_HGR")
    for key in ("training_environment_steps", "checkpoint_interval", "evaluation_episodes", "generated_train_count"):
        if type(value.get(key)) is not int or value[key] < 1:
            raise ValueError("positive integer required: " + key)
    seeds = value.get("training_seeds")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("unique training seeds required")
    for seed in seeds + [value.get(k) for k in ("evaluation_seed", "generated_train_seed", "generated_eval_seed")]:
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError("seed must fit nonnegative int31")
    if value.get("hgr_policy_mode") not in ("stochastic", "deterministic_mean"):
        raise ValueError("invalid HGR evaluation policy mode")
    distance = value.get("stagnation_distance_per_step")
    if type(distance) not in (int, float) or not 0 < distance < 1:
        raise ValueError("invalid stagnation threshold")
    if set(value.get("templates", {})) != set(ARMS):
        raise ValueError("all four templates required")
    return value


def scenes(manifest, split):
    rows = manifest.get("scenarios")
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty scenario manifest required")
    ids, seeds, content = set(), set(), set()
    for row in rows:
        name, seed = row.get("scenario_id"), row.get("scenario_seed")
        if (not isinstance(name, str) or not name.strip() or name in ids
                or type(seed) is not int or not 0 <= seed < 2**63 or seed in seeds
                or row.get("scenario_split") != split or row.get("scenario_role") != split
                or row.get("scenario_profile") != "M20_MOVING_UNKNOWN_MULTI"
                or row.get("protocol") not in ("ch3_unknown_map_v1", "CH3_UNKNOWN_MAP_V1")
                or row.get("max_steps") != 400):
            raise ValueError("scenario identity/split/profile/protocol/400-step mismatch: " + str(name))
        ids.add(name)
        seeds.add(seed)
        # A relabeled copy is still the same scenario. Exclude identity metadata.
        content.add(digest({k: v for k, v in row.items() if k not in
                           ("scenario_id", "scenario_seed", "scenario_split", "scenario_role", "pair_group_id", "protocol")}))
    if len(content) != len(rows):
        raise ValueError("duplicated scenario content")
    return ids, seeds, content


def validate_pair(train, evaluation, count):
    a, b = scenes(train, "train"), scenes(evaluation, "validation")
    if len(evaluation["scenarios"]) < count:
        raise ValueError("insufficient evaluation scenarios")
    if any(x & y for x, y in zip(a, b)):
        raise ValueError("train/evaluation overlap by ID, seed, or scenario content")


def prepare(output, *, config=DEFAULT, train_manifest=None, eval_manifest=None, generate_scenes=False):
    from chapter3_bser.experiments.d2_v1.provenance import framework_sources
    from tools.ch3_baselines.registry import task_conditions
    from chapter3_bser.experiments.hgr.train import validate_config as hgr_config
    from chapter3_bser.experiments.baselines.common.checkpoint import validate_config as baseline_config
    value = settings(config)
    output = Path(output).resolve()
    if "collision_terminal" not in output.parts:
        raise ValueError("output path must contain collision_terminal")
    if output.exists():
        raise FileExistsError("prepare requires a new directory: " + str(output))
    source = framework_sources()
    if generate_scenes:
        if train_manifest or eval_manifest:
            raise ValueError("choose existing manifests OR explicit scene generation")
        from core.scenarios.ch3_generator_impl import build_scenario_manifests
        def generate(split, count, seed):
            return build_scenario_manifests(count=count, generator_seed=seed, split=split,
                       profiles=["M20_MOVING_UNKNOWN_MULTI"])["M20_MOVING_UNKNOWN_MULTI"]
        train = generate("train", value["generated_train_count"], value["generated_train_seed"])
        evaluation = generate("validation", value["evaluation_episodes"], value["generated_eval_seed"])
        origin = dict(kind="generated", train_seed=value["generated_train_seed"], eval_seed=value["generated_eval_seed"])
    else:
        if not train_manifest or not eval_manifest:
            raise ValueError("provide --train-manifest and --eval-manifest, or explicitly --generate-scenes")
        train, evaluation = read(train_manifest), read(eval_manifest)
        origin = dict(kind="existing", train_path=str(Path(train_manifest).resolve()),
                      eval_path=str(Path(eval_manifest).resolve()), train_sha256=sha(train_manifest),
                      eval_sha256=sha(eval_manifest))
    validate_pair(train, evaluation, value["evaluation_episodes"])
    evaluation = dict(scenarios=copy.deepcopy(evaluation["scenarios"][:value["evaluation_episodes"]]))
    train = dict(scenarios=copy.deepcopy(train["scenarios"]))
    templates = {arm: read(ROOT / value["templates"][arm]) for arm in ARMS}
    for arm, template in templates.items():
        if arm in ARMS[1:3]:
            baseline = "B2_direct_mc" if arm == "D2_B2" else "B3_direct_boundary"
            templates[arm] = baseline_config(template, expected_baseline=baseline)
        else:
            templates[arm] = hgr_config(template)
            if template.get("algorithm") != "hgr" or template.get("method") != "ch3_hgr":
                raise ValueError("D2 reference and D2_HGR require the HGR template identity")
        if template.get("planner_protocol") != "d2_v1":
            raise ValueError("all arms must use frozen D2")
    common = task_conditions(templates["D2"])
    if any(task_conditions(t) != common for t in templates.values()):
        raise ValueError("task conditions differ across arms")
    # All validation above is read-only. Each prepared run owns new files only.
    write(output / "inputs/train.json", train)
    write(output / "inputs/evaluation.json", evaluation)
    assets = {name: sha(output / name) for name in ("inputs/train.json", "inputs/evaluation.json")}
    jobs = []
    for arm in ARMS:
        for seed in ([None] if arm == "D2" else value["training_seeds"]):
            name = arm if seed is None else f"{arm}_seed{seed}"
            cfg = copy.deepcopy(templates[arm])
            cfg.update(seed=value["evaluation_seed"] if seed is None else seed,
                       scenario_manifest=str(output / "inputs/train.json"),
                       scenario_manifest_sha256=assets["inputs/train.json"],
                       max_total_environment_steps=value["training_environment_steps"],
                       total_main_trajectories=value["training_environment_steps"],
                       checkpoint_interval=value["checkpoint_interval"],
                       output_dir=str(output / "jobs" / name / "train"))
            path = f"inputs/{name}.json"
            write(output / path, cfg)
            assets[path] = sha(output / path)
            jobs.append(dict(name=name, arm=arm, training_seed=seed, config=path,
                             train=f"jobs/{name}/train", evaluation=f"jobs/{name}/evaluation"))
    result = dict(schema="ch3.d2_suite.plan.v1", settings=value, jobs=jobs, assets=assets,
                  source_sha256=source["inventory"]["sha256"], source_profile=source["checkout_profile"],
                  scene_origin=origin, common_task_conditions=common,
                  checkpoint_selection="last completed episode/cycle at environment-step budget; no test-set selection",
                  training_pairing="same population and seed labels; native sampling schedules differ",
                  evaluation_pairing="same ordered scenarios and innovation seed+index; trajectories may consume different RNG events",
                  heldout_claim="train/eval disjointness checked; prior human tuning exposure is not knowable",
                  budget_policy="finish started episode/cycle, report actual steps and overshoot")
    result["sha256"] = digest(result)
    write(output / "plan.json", result)
    return result


def load_plan(root, *, verify_source=True):
    root = Path(root).resolve()
    value = read(root / "plan.json")
    if value.get("schema") != "ch3.d2_suite.plan.v1" or value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("experiment plan identity mismatch")
    for name, expected in value["assets"].items():
        path = (root / name).resolve()
        if root not in path.parents or sha(path) != expected:
            raise ValueError("prepared input changed: " + name)
    if verify_source:
        from chapter3_bser.experiments.d2_v1.provenance import framework_sources
        if framework_sources()["inventory"]["sha256"] != value["source_sha256"]:
            raise ValueError("source changed after preparation; use a new run directory")
    return value
