"""Explicit train/resume/fixed-policy evaluation entry points."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import torch

from .train import DEFAULT_CONFIG, ROOT, Trainer, load_config, load_checkpoint, write_json
from .runtime import collect_trajectory
from chapter3_bser.models.hgr import METHODS
from chapter3_bser.models.hgr.policy import HandoffPolicy, weights_hash
from core.scenarios.ch3_generator_impl import build_scenario_manifests


def evaluate(checkpoint, output, *, episodes=100, seed=12729, policy_mode="stochastic", manifest=None):
    payload = load_checkpoint(checkpoint)
    config = copy.deepcopy(payload["config"])
    policy = HandoffPolicy(config["policy"])
    policy.load_state_dict(payload["policy"], strict=True)
    policy.eval(); policy.requires_grad_(False)
    initial_hash = weights_hash(policy)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    if manifest:
        scenarios = json.loads(Path(manifest).read_text(encoding="utf-8"))["scenarios"]
        if any(s.get("scenario_split") != "validation" for s in scenarios):
            raise ValueError("evaluation requires a held-out validation manifest")
    else:
        scenarios = build_scenario_manifests(count=episodes, generator_seed=seed, split="validation", profiles=[config["profile"]])[config["profile"]]["scenarios"]
    if len(scenarios) < episodes:
        raise ValueError("evaluation manifest is too short")
    rows = []
    for i, scenario in enumerate(scenarios[:episodes]):
        trajectory = collect_trajectory(config, scenario, policy, seed=seed+i, episode_id=i,
                                         deterministic=policy_mode == "deterministic_mean")
        rows.append(dict(**trajectory["summary"], method=config["method"], policy_mode=policy_mode,
                         checkpoint=str(Path(checkpoint).resolve()), policy_hash=initial_hash))
    if weights_hash(policy) != initial_hash:
        raise RuntimeError("evaluation mutated the policy")
    result = dict(method=config["method"], policy_mode=policy_mode, episodes=len(rows),
                  run_mode="same_method_evaluation",
                  safe_success_rate=sum(bool(r["safe_success"]) for r in rows)/len(rows),
                  mean_team_discounted_return=sum(r["team_discounted_return"] for r in rows)/len(rows),
                  mean_team_undiscounted_return=sum(r["team_undiscounted_return"] for r in rows)/len(rows),
                  reward_objective=config["reward_objective"], gamma=config["rl"]["gamma"],
                  training_update=False, performance_claims_supported=False)
    write_json(output / "episodes.json", rows); write_json(output / "summary.json", result)
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # The existing PRRAC production parsers retain their full CLI and runtime checks.
    if argv and argv[0] in ("prrac-train", "prrac-evaluate"):
        if argv[0] == "prrac-train":
            from chapter3_bser.experiments.phase1c_prrac.train_phase1c_prrac import main as entry
            default = ROOT / "configs/chapter3/bser_phase1c_prrac_team_train.json"
        else:
            from chapter3_bser.experiments.phase1c_prrac.evaluate_prrac_checkpoints import main as entry
            default = ROOT / "configs/chapter3/bser_phase1c_prrac_team_eval.json"
        args = argv[1:]
        return entry(args if "--config" in args else ["--config", str(default), *args])
    parser = argparse.ArgumentParser(description="Full-task HGR, stochastic MC and corrected-boundary methods; PRRAC team: prrac-train / prrac-evaluate")
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "resume"):
        sub = subs.add_parser(command)
        sub.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        sub.add_argument("--output-dir", type=Path)
        sub.add_argument("--algorithm", choices=METHODS)
        sub.add_argument("--total-main-trajectories", type=int)
        sub.add_argument("--max-total-environment-steps", type=int)
        if command == "resume":
            sub.add_argument("--checkpoint", type=Path, required=True)
        else:
            sub.add_argument("--mean-initialization", type=Path)
    sub = subs.add_parser("evaluate")
    sub.add_argument("--checkpoint", type=Path, required=True)
    sub.add_argument("--output-dir", type=Path, required=True)
    sub.add_argument("--episodes", type=int, default=100)
    sub.add_argument("--seed", type=int, default=12729)
    sub.add_argument("--policy-mode", choices=("stochastic", "deterministic_mean"), default="stochastic")
    sub.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        if args.episodes < 1:
            parser.error("episodes must be positive")
        result = evaluate(args.checkpoint, args.output_dir, episodes=args.episodes, seed=args.seed,
                          policy_mode=args.policy_mode, manifest=args.manifest)
    else:
        config = load_config(args.config)
        if args.algorithm:
            config["algorithm"] = args.algorithm; config["method"] = "ch3_" + args.algorithm
        if args.total_main_trajectories is not None:
            if args.total_main_trajectories < 1:
                parser.error("total-main-trajectories must be positive")
            config["total_main_trajectories"] = args.total_main_trajectories
        if args.max_total_environment_steps is not None:
            if args.max_total_environment_steps < 1:
                parser.error("max-total-environment-steps must be positive")
            config["max_total_environment_steps"] = args.max_total_environment_steps
        result = Trainer(config, args.output_dir or config["output_dir"],
                         resume=args.checkpoint if args.command == "resume" else None,
                         mean_initialization=getattr(args, "mean_initialization", None)).run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
