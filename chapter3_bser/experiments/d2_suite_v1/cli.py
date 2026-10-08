"""Unified four-arm preparation, manual launch, and paired summary."""
import argparse
import json
from .plan import ARMS, DEFAULT, prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="freeze inputs only; never train/evaluate")
    p.add_argument("--output", required=True)
    p.add_argument("--config", default=str(DEFAULT))
    p.add_argument("--train-manifest")
    p.add_argument("--eval-manifest")
    p.add_argument("--generate-scenes", action="store_true")
    p.add_argument("--learner-device", choices=("cpu", "cuda"), help="explicit B2/B3 learner/rollout device; HGR stays CPU")
    p.add_argument("--cpu-threads", type=int, help="explicit PyTorch CPU threads per seed process")
    p = commands.add_parser("run", help="preview by default; execution requires --execute")
    p.add_argument("--root", required=True)
    p.add_argument("--stage", choices=("all", "train", "evaluate"), default="all")
    p.add_argument("--arms", nargs="+", choices=ARMS)
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--execute", action="store_true")
    p = commands.add_parser("summarize")
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True, help="new report directory; existing results are never overwritten")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "prepare":
        value = prepare(**args)
        result = dict(prepared=True, jobs=len(value["jobs"]), plan_sha256=value["sha256"],
                      note="Inputs frozen. No training or evaluation started.")
    elif command == "run":
        from .run import execute
        result = execute(**args)
    else:
        from .summarize import summarize
        result = summarize(**args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
