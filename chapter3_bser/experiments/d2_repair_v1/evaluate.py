"""Manual held-out evaluation and return/metric ordering diagnostics."""
import argparse
import json
from pathlib import Path
import torch

from chapter3_bser.experiments.baselines.common.checkpoint import digest, validated_output, write_json
from chapter3_bser.experiments.d2_performance.logging import JsonlJournal
from .diagnostics import paired_ordering
from .config import scenario_fingerprint
from .runtime import RepairedMissionRuntime
from .train import load_for_evaluation


def evaluate(checkpoint, manifest_path, output):
    from .provenance import verify_reviewed_changes
    verify_reviewed_changes()
    model, config, payload = load_for_evaluation(checkpoint)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    scenarios = manifest.get("scenarios", [])
    ids = [s.get("scenario_id") for s in scenarios]
    if not scenarios or len(set(ids)) != len(ids) or set(ids) & set(payload["training_scenario_ids"]):
        raise ValueError("evaluation requires unique held-out scenarios disjoint from training")
    for scene in scenarios:
        if (scene.get("scenario_seed") in payload["training_scenario_seeds"]
                or scenario_fingerprint(scene) in payload["training_scenario_fingerprints"]):
            raise ValueError("evaluation task seed or initial configuration overlaps training")
        if (scene.get("scenario_role") not in ("eval", "smoke_eval")
                or scene.get("scenario_split") != scene["scenario_role"]
                or scene.get("scenario_profile") != config["profile"]
                or scene.get("max_steps") != config["max_steps"]):
            raise ValueError("evaluation scenario role/profile/horizon mismatch")
    output = validated_output(config, output)
    output.mkdir(parents=True, exist_ok=True)
    journal = JsonlJournal(output / "episodes.jsonl")
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for scene in scenarios:
            seed = int(scene["scenario_seed"])
            runtime = RepairedMissionRuntime(config, scene, seed=seed)
            try:
                while not runtime.terminal:
                    runtime.advance(model, explore=False)
                row = runtime.summary()
                collision_step = runtime.env.unwrapped.episode_outcome.first_collision_step
                collided = runtime.env.unwrapped.episode_outcome.termination_reason == "obstacle_collision"
                row.update(policy_id=payload["model_state_sha256"], method=config["method"],
                    scenario_sha256=digest(scene),
                    trajectory_complete=True, environment_innovation_seed=seed, max_steps=config["max_steps"],
                    pre_found_collision=bool(collided and (row["found_step"] is None or collision_step <= row["found_step"])),
                    found_penalized_steps=config["max_steps"] if row["found_step"] is None else row["found_step"])
                journal.append(row)
            finally:
                runtime.close()
        journal.materialize()
        write_json(output / "evaluation_identity.json", dict(checkpoint=str(checkpoint),
                   scenario_manifest_sha256=digest(manifest), completed=len(journal), performance_passed=False))
    finally:
        torch.set_num_threads(threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ev = commands.add_parser("evaluate")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--manifest", required=True)
    ev.add_argument("--output", required=True)
    ev.add_argument("--execute", action="store_true")
    rank = commands.add_parser("ordering")
    rank.add_argument("--episodes", nargs="+", required=True)
    rank.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        if not args.execute:
            print(json.dumps(dict(executed=False, checkpoint=args.checkpoint, manifest=args.manifest)))
            return
        evaluate(args.checkpoint, args.manifest, args.output)
    else:
        rows = []
        for path in args.episodes:
            text = Path(path).read_text(encoding="utf-8")
            rows.extend(json.loads(text) if text.lstrip().startswith("[") else
                        [json.loads(line) for line in text.splitlines() if line.strip()])
        destination = Path(args.output)
        if destination.exists():
            raise FileExistsError(destination)
        write_json(destination, paired_ordering(rows))


if __name__ == "__main__":
    main()
