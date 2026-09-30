"""Prepare and freeze a fresh paired evaluation population, without simulation."""
import argparse
from pathlib import Path

from chapter3_bser.experiments.safe_search_v1.run_paired import (
    ROOT, digest, read_json, file_hash, output_directory, validate_manifest)
from chapter3_bser.experiments.safe_search_v1.run_development import CONFIG

SCHEMA = "ch3.bser_final.confirmation_scenarios.v1"
GENERATOR_SEED = 20260930
INNOVATION_SEED = 492929
DEFAULT_REFERENCE = ROOT / "3090结果/collision_terminal/B0_search_prior_eval100_seed12729_v1/evaluation_manifest.json"


def content_key(scene):
    ignored = {"scenario_id", "scenario_seed", "scenario_split", "scenario_role", "scenario_index"}
    return digest({k: v for k, v in scene.items() if k not in ignored})


def validate_confirmation(manifest, config):
    if (manifest.get("schema") != SCHEMA or manifest.get("sha256") != digest(
            {k:v for k,v in manifest.items() if k != "sha256"})
            or manifest.get("generator_seed") != GENERATOR_SEED
            or manifest.get("innovation_seed") != INNOVATION_SEED
            or manifest.get("count") != len(manifest.get("scenarios", []))
            or not 1 <= manifest["count"] <= 1000):
        raise ValueError("confirmation requires a sealed bser_final scenario manifest")
    from .provenance import framework_sources
    if manifest["sources"] != framework_sources():
        raise ValueError("code changed after freezing confirmation scenarios")
    scenarios = validate_manifest(manifest, config, ",".join(str(i) for i in range(manifest["count"])), INNOVATION_SEED)
    excluded = manifest["excluded_reference"]
    if (len(excluded["scenario_ids"]) != 100 or len(excluded["scenario_seeds"]) != 100
            or len(excluded["content_keys"]) != 100):
        raise ValueError("missing complete original population exclusion")
    keys = []
    selected = []
    for i, scene in scenarios:
        key = content_key(scene)
        if (scene["scenario_id"] in excluded["scenario_ids"] or scene["scenario_seed"] in excluded["scenario_seeds"]
                or key in excluded["content_keys"] or key in keys or scene["max_steps"] != 400):
            raise ValueError("confirmation scenario overlap/duplicate/horizon mismatch")
        keys.append(key)
        selected.append(dict(original_episode_index=i, scenario_id=scene["scenario_id"],
            scenario_seed=scene["scenario_seed"], environment_innovation_seed=INNOVATION_SEED+i,
            scenario_sha256=digest(scene)))
    return selected, INNOVATION_SEED


def prepare(output, reference=DEFAULT_REFERENCE, count=100):
    if type(count) is not int or not 1 <= count <= 1000:
        raise ValueError("scenario count must be in [1,1000]")
    from .provenance import framework_sources
    sources = framework_sources()
    output = Path(output).resolve()
    if (ROOT / "runs").resolve() not in output.parents:
        raise ValueError("scenario preparation requires a new subdirectory under runs/")
    output_directory(output, (reference, CONFIG))
    config = read_json(CONFIG)
    original = read_json(reference)
    # Require the exact development population, not an arbitrary exclusion list.
    from chapter3_bser.experiments.safe_search_v1.run_development import PLAN, validate_development_inputs
    validate_development_inputs(original, config, read_json(PLAN))
    from core.scenarios.ch3_generator_impl import build_scenario_manifests
    generated = build_scenario_manifests(count=count, generator_seed=GENERATOR_SEED,
        split="validation", profiles=[config["profile"]])[config["profile"]]
    scenes = generated["scenarios"]
    # Runtime and all arms use the same unchanged collision-terminal 400 steps.
    for scene in scenes:
        scene["max_steps"] = 400
    manifest = dict(schema=SCHEMA, generator_seed=GENERATOR_SEED, innovation_seed=INNOVATION_SEED,
        count=count, scenarios=scenes, sources=sources,
        excluded_reference=dict(sha256=file_hash(reference),
            scenario_ids=[s["scenario_id"] for s in original["scenarios"]],
            scenario_seeds=[s["scenario_seed"] for s in original["scenarios"]],
            content_keys=[content_key(s) for s in original["scenarios"]]),
        scope="New generation seed, disjoint from original 100 by ID, seed and content; no global train-overlap claim.",
        training=False, simulator_steps=0)
    manifest["sha256"] = digest(manifest)
    validate_confirmation(manifest, config)
    output.mkdir(parents=True, exist_ok=True)
    from .run_windows import write_json
    write_json(output / "evaluation_manifest.json", manifest)
    return output / "evaluation_manifest.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--count", type=int, default=100,
        help="freeze sample size before running; do not increase it after inspecting significance")
    args = parser.parse_args(argv)
    print(prepare(args.output_dir, args.reference, args.count))


if __name__ == "__main__":
    main()
