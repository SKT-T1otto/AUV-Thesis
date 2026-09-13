"""Manual collision_terminal_v1 entry points; parsing/preparation never trains."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "configs/chapter3/bser_phase1c_prrac_collision_terminal_train.json"
EVAL = ROOT / "configs/chapter3/bser_phase1c_prrac_collision_terminal_eval.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode in ("evaluate", "train", "warmstart", "resume"):
        command = commands.add_parser(mode)
        command.add_argument("--output-dir", type=Path, required=True,
                             help="Independent run path containing a collision_terminal directory")
        command.add_argument("--workers", type=int, default=4)
        command.add_argument("--device", default="cpu")
        command.add_argument("--episodes", type=int, default=100 if mode == "evaluate" else 1000)
        if mode != "train":
            command.add_argument("--checkpoint", type=Path, required=True)
        if mode == "evaluate":
            command.add_argument("--prepare-only", action="store_true")
            command.add_argument("--allow-protocol-transfer", action="store_true")
        else:
            command.add_argument("--seed", type=int, default=2729)
        if mode == "warmstart":
            command.add_argument("--critic-warmup-updates", type=int, default=256)
    args = parser.parse_args(argv)
    if args.episodes <= 0 or args.workers <= 0:
        parser.error("episodes and workers must be positive")
    if "collision_terminal" not in args.output_dir.parts:
        parser.error("output path must contain collision_terminal as a directory component")
    if args.mode != "train":
        from .checkpoint_transfer import checkpoint_path
        checkpoint_path(args.checkpoint)
    if args.mode == "evaluate":
        from .paired_evaluation import main as paired_main
        # Same production pairing, manifest, checkpoint and provenance gates.
        arguments = ["run", "--checkpoint", str(args.checkpoint), "--config", str(EVAL),
                     "--episodes", str(args.episodes), "--workers", str(args.workers),
                     "--output-root", str(args.output_dir), "--device", args.device]
        if args.allow_protocol_transfer:
            arguments.append("--allow-protocol-transfer")
        if args.prepare_only:
            arguments.append("--prepare-only")
        return paired_main(arguments)
    from .train_phase1c_prrac import run_training
    run_training(config_path=TRAIN, output_dir=args.output_dir, seed_override=args.seed,
                 episodes_override=args.episodes, workers_override=args.workers,
                 device_override=args.device,
                 resume=args.checkpoint if args.mode == "resume" else None,
                 init_actors_from=args.checkpoint if args.mode == "warmstart" else None,
                 critic_warmup_updates=args.critic_warmup_updates if args.mode == "warmstart" else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
