"""Manual paired evaluation and strict, simulation-free CSV analysis.

Importing or preparing a plan never starts an environment. Only the explicit
CLI `run` command without --prepare-only executes the two evaluation commands.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/chapter3/bser_phase1c_prrac_eval.json"
CONTROLLERS = ("prior_only", "full_prrac")
BOOL_FIELDS = ("found", "success", "contact_episode", "hold_episode", "collision_episode")
RUN_SPECIFIC_FIELDS = {
    "controller", "output_dir", "resolved_output_dir",
    "resolved_config_output_path", "resolved_config_hash",
}
FIXED_INPUT_FIELDS = (
    "method", "implementation_version", "architecture_version", "checkpoint_schema",
    "base_candidate", "profile", "split", "scenario_seed", "evaluation_episodes",
    "max_steps", "observation_dim", "action_dim", "critic_dim", "device", "workers",
    "explore", "training_update", "modes", "execution_variants", "failure_trace",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def parse_optional_bool(value):
    if value is None or isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1"):
            return True
        if text in ("false", "0"):
            return False
    raise ValueError(f"invalid CSV boolean: {value!r}")


def read_episode_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for name, value in tuple(row.items()):
            if value is None or not value.strip():
                row[name] = None
        for name in BOOL_FIELDS:
            row[name] = parse_optional_bool(row.get(name))
        if row["success"] is True and row["found"] is False:
            raise ValueError("success=true with found=false is inconsistent")
    return rows


def outcome_summary(rows):
    """Missing outcomes remain unavailable; never treat blank as failure."""
    found = [row for row in rows if row["found"] is True]
    found_complete = all(row["found"] is not None for row in rows)
    success_complete = all(row["success"] is not None for row in rows)
    conditional_complete = found_complete and all(row["success"] is not None for row in found)
    numerator = sum(row["success"] is True for row in found)
    return {
        "episodes": len(rows),
        "found_count": len(found),
        "found_missing_count": sum(row["found"] is None for row in rows),
        "success_missing_count": sum(row["success"] is None for row in rows),
        "found_rate": len(found) / len(rows) if rows and found_complete else None,
        "success_rate": sum(row["success"] is True for row in rows) / len(rows) if rows and success_complete else None,
        "success_if_found_numerator": numerator,
        "success_if_found_denominator": len(found),
        "success_if_found_rate": numerator / len(found) if found and conditional_complete else None,
    }


def reject_fixture(checkpoint, metadata=None):
    if "untrained_test_fixture" in Path(checkpoint).name.lower():
        raise ValueError("untrained_test_fixture is forbidden for PRRAC performance comparison")
    if metadata is not None:
        if metadata.get("config_hash") == "test-config-hash" or int(metadata.get("completed_episode") or 0) <= 0:
            raise ValueError("checkpoint has test/untrained metadata; supply a trained PRRAC checkpoint")


def prepare_pair(*, checkpoint, config_path=DEFAULT_CONFIG, episodes=1, workers=1, output_root=None):
    checkpoint = Path(checkpoint).resolve(strict=True)
    reject_fixture(checkpoint)
    if not checkpoint.is_file() or episodes not in (1, 10) or workers < 1:
        raise ValueError("a checkpoint file, 1 or 10 scenarios, and positive workers are required")
    config_path = Path(config_path).resolve(strict=True)
    config = read_json(config_path)
    if config.get("modes", ["full_prrac"]) != ["full_prrac"]:
        raise ValueError("paired controller evaluation requires modes=['full_prrac']")
    if config.get("explore") is not False or config.get("training_update") is not False:
        raise ValueError("paired evaluation requires explore=false and training_update=false")
    if "scenario_ids" in config:
        raise ValueError("use the shared generated manifest, not a diagnostic scenario_ids subset")
    # Exactly one runtime/recovery combination: N means N episodes per controller.
    variants = config.get("execution_variants", ["B0_LEGACY_V2_1"])
    recovery = config.get("search_recovery_variants", config.get("search_collision_recovery", {}).get("variants", ["S2A_C0_BASELINE"]))
    if len(variants) != 1 or len(recovery) != 1:
        raise ValueError("paired evaluation requires one execution and one recovery variant")
    config.update(checkpoints=[str(checkpoint)], checkpoint_globs=[], evaluation_episodes=episodes, workers=workers)
    config.pop("controller", None)
    parent = Path(output_root or ROOT / "outputs/chapter3/phase1c_prrac/paired").resolve()
    root = parent / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    config_file = root / "paired_eval_config.json"
    write_new_json(config_file, config)
    commands = {
        mode: [sys.executable, "-B", "-m", "chapter3_bser.experiments.phase1c_prrac.evaluate_prrac_checkpoints",
               "--config", str(config_file), "--controller", mode, "--output-dir", str(root / mode)]
        for mode in CONTROLLERS
    }
    plan = {
        "schema": "prrac.manual_controller_pair.v1", "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint), "source_config": str(config_path),
        "shared_config": str(config_file), "shared_config_sha256": sha256(config_file),
        "episodes_per_controller": episodes, "total_planned_episodes": 2 * episodes,
        "commands": commands, "execution_started": False,
    }
    write_new_json(root / "pair_plan.json", plan)
    return root, plan


def validate_checkpoint_for_pair(plan):
    # Reuse, never change, the production loader and its runtime compatibility gates.
    from . import evaluate_prrac_checkpoints as evaluator
    config = evaluator._load_config(Path(plan["shared_config"]))
    learner, payload = evaluator.load_prrac_checkpoint(Path(plan["checkpoint"]), device="cpu", config=config)
    metadata = dict(payload["metadata"])
    metadata["completed_episode"] = payload.get("completed_episode", metadata.get("completed_episode", 0))
    reject_fixture(plan["checkpoint"], metadata)
    del learner


def analyze_pair(root):
    """Read only: return a comparison, never launch an evaluator or edit its output."""
    root = Path(root)
    plan = read_json(root / "pair_plan.json")
    reject_fixture(plan["checkpoint"])
    if sha256(plan["checkpoint"]) != plan["checkpoint_sha256"]:
        raise ValueError("checkpoint bytes changed since pair preparation")
    if sha256(plan["shared_config"]) != plan["shared_config_sha256"]:
        raise ValueError("shared configuration changed since pair preparation")
    shared_config = read_json(plan["shared_config"])
    data, configs, manifests = {}, {}, {}
    for mode in CONTROLLERS:
        directory = root / mode
        configs[mode] = read_json(directory / "resolved_evaluation_config.json")
        manifests[mode] = read_json(directory / "evaluation_manifest.json")
        summary = read_json(directory / "evaluation_summary.json")
        if configs[mode].get("controller") != mode or summary.get("controller_mode") != mode:
            raise ValueError("controller identity mismatch")
        if configs[mode].get("checkpoints") != [plan["checkpoint"]]:
            raise ValueError("checkpoint identity mismatch")
        if any(configs[mode].get(key) != shared_config[key] for key in FIXED_INPUT_FIELDS if key in shared_config):
            raise ValueError("paired scenario/configuration mismatch with prepared inputs")
        if not manifests[mode].get("manifest_sha256"):
            raise ValueError("missing manifest identity")
        data[mode] = read_episode_csv(directory / "episode_evaluation.csv")
        if len(data[mode]) != plan["episodes_per_controller"]:
            raise ValueError("incomplete or unexpected episode count")
        for row in data[mode]:
            if row.get("controller_mode") != mode or row.get("checkpoint") != plan["checkpoint"]:
                raise ValueError("episode controller/checkpoint identity mismatch")
            reject_fixture(plan["checkpoint"], {"config_hash": row.get("checkpoint_config_hash"), "completed_episode": row.get("checkpoint_episode")})
            if row.get("manifest_sha256") != manifests[mode].get("manifest_sha256"):
                raise ValueError("episode manifest mismatch")
    fixed = [{key: value for key, value in configs[mode].items() if key not in RUN_SPECIFIC_FIELDS} for mode in CONTROLLERS]
    if fixed[0] != fixed[1] or manifests[CONTROLLERS[0]] != manifests[CONTROLLERS[1]]:
        raise ValueError("paired scenario/configuration mismatch")
    indexed = {}
    for mode in CONTROLLERS:
        indexed[mode] = {}
        for row in data[mode]:
            if row.get("scenario_id") is None or row.get("scenario_seed") is None:
                raise ValueError("missing scenario identity")
            key = (row["scenario_id"], int(row["scenario_seed"]))
            if key in indexed[mode]:
                raise ValueError("duplicate scenario identity")
            indexed[mode][key] = row
        expected = {(str(row["scenario_id"]), int(row["scenario_seed"]))
                    for row in manifests[mode].get("scenarios", ())}
        if set(indexed[mode]) != expected or len(expected) != plan["episodes_per_controller"]:
            raise ValueError("episode scenario identities do not match manifest")
    if indexed["prior_only"].keys() != indexed["full_prrac"].keys():
        raise ValueError("paired scenario id/seed mismatch")
    summaries = {mode: outcome_summary(data[mode]) for mode in CONTROLLERS}
    return {
        "schema": "prrac.controller_pair_analysis.v1", "paired_scenarios": len(data["prior_only"]),
        "checkpoint_sha256": plan["checkpoint_sha256"], "manifest_sha256": manifests["prior_only"]["manifest_sha256"],
        "controllers": summaries, "difference_direction": "prior_only_minus_full_prrac",
        "differences": {field: None if any(summaries[mode][field] is None for mode in CONTROLLERS)
                        else summaries["prior_only"][field] - summaries["full_prrac"][field]
                        for field in ("found_rate", "success_rate", "success_if_found_rate")},
        "performance_passed": None,
        "conditional_rate_note": "Each controller uses its own Found episodes; denominators may differ. No Found => null.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--episodes", type=int, choices=(1, 10), required=True)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--output-root", type=Path)
    run.add_argument("--prepare-only", action="store_true")
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--pair-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "analyze":
        print(json.dumps(analyze_pair(args.pair_dir), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    root, plan = prepare_pair(checkpoint=args.checkpoint, config_path=args.config, episodes=args.episodes,
                              workers=args.workers, output_root=args.output_root)
    print(f"Pair directory: {root}", flush=True)
    if args.prepare_only:
        print("Prepared only; no checkpoint loaded and no episode started.")
        return 0
    validate_checkpoint_for_pair(plan)
    write_new_json(root / "execution_started.json", {"manual_run": True, "controllers": list(CONTROLLERS)})
    for mode in CONTROLLERS:
        if sha256(plan["checkpoint"]) != plan["checkpoint_sha256"] or sha256(plan["shared_config"]) != plan["shared_config_sha256"]:
            raise ValueError("paired input changed; evaluation stopped")
        subprocess.run(plan["commands"][mode], cwd=ROOT, check=True)
    result = analyze_pair(root)
    write_new_json(root / "paired_analysis.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
