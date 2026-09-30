"""Manual Linux launch support; native algorithms and CPU device are unchanged."""
import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import platform
import socket
import sys
import traceback
import uuid

from .plan import ARMS, load_plan, read, write
from .run import execute


def dependencies():
    """Import native dependencies without constructing policies or simulators."""
    import numpy
    import torch
    for name in ("baselines.direct_mc.train", "baselines.direct_boundary.train", "hgr.train"):
        importlib.import_module("chapter3_bser.experiments." + name)
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                numpy=numpy.__version__, torch=torch.__version__,
                training_device="cpu", cuda_available=bool(torch.cuda.is_available()))


def check(root):
    root = Path(root).resolve()
    plan = load_plan(root)
    for job in plan["jobs"]:
        config = read(root / job["config"])
        for key, expected in (("scenario_manifest", root / "inputs/train.json"),
                              ("output_dir", root / job["train"])):
            actual = Path(config[key])
            if not actual.is_absolute() or actual.resolve() != expected.resolve():
                raise ValueError("prepared paths belong to another location; prepare a new run on this machine: "
                                 + job["name"] + ":" + key)
    return dict(root=str(root), plan_sha256=plan["sha256"], source_sha256=plan["source_sha256"],
                learning_arms=list(ARMS[1:]), training_jobs=sum(j["arm"] != "D2" for j in plan["jobs"]),
                evaluation_jobs=len(plan["jobs"]), runtime=dependencies())


@contextmanager
def run_lock(root):
    """Exclusive wrapper lock; never take over a stale or foreign lock."""
    path = Path(root) / ".d2_linux.lock"
    owner = dict(pid=os.getpid(), host=socket.gethostname(), token=uuid.uuid4().hex)
    payload = json.dumps(owner, sort_keys=True)
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise FileExistsError("run is locked: " + str(path)
                              + "; check the owning process before manually removing a stale lock") from exc
    try:
        with stream:
            stream.write(payload)
        yield
    finally:
        if path.exists() and path.read_text(encoding="utf-8") == payload:
            path.unlink()


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


def launch(root, *, stage, arms=None, seeds=None, confirmed=False):
    if stage not in ("train", "evaluate"):
        raise ValueError("stage must be train or evaluate")
    if stage == "train":
        if arms and set(arms) - set(ARMS[1:]):
            raise ValueError("D2 has no trainable policy; select D2_B2, D2_B3 or D2_HGR")
        arms = list(ARMS[1:]) if arms is None else arms
    root = Path(root).resolve()
    preflight = check(root)
    options = dict(stage=stage, arms=arms, seeds=seeds)
    preview = execute(root, **options, execute=False)
    if not confirmed:
        return dict(preflight=preflight, **preview)
    # Only explicit execution creates launch artifacts. Generic run.execute is
    # still available; callers must not mix it with concurrent wrapper launches.
    with run_lock(root):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        directory = root / "launch_logs" / stamp
        directory.mkdir(parents=True, exist_ok=False)
        status = dict(stage=stage, arms=arms, seeds=seeds, preflight=preflight,
                      started_utc=datetime.now(timezone.utc).isoformat(), status="running")
        write(directory / "launch.json", status)
        with (directory / "console.log").open("x", encoding="utf-8", buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                print("launch log: " + str(directory), flush=True)
                print(json.dumps(preflight, ensure_ascii=False), flush=True)
                try:
                    result = execute(root, **options, execute=True)
                    status.update(status="complete", result=result)
                except BaseException as exc:
                    status.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                    traceback.print_exc()
                    raise
                finally:
                    status["finished_utc"] = datetime.now(timezone.utc).isoformat()
                    write(directory / "launch.json", status)
        return dict(**result, launch_directory=str(directory))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("check", help="read-only source, path and dependency checks")
    p.add_argument("--root", required=True)
    for command in ("train", "evaluate"):
        p = commands.add_parser(command, help="preview only unless --execute is provided")
        p.add_argument("--root", required=True)
        p.add_argument("--arms", nargs="+", choices=ARMS[1:] if command == "train" else ARMS)
        p.add_argument("--seeds", nargs="+", type=int)
        p.add_argument("--execute", dest="confirmed", action="store_true")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    result = check(**args) if command == "check" else launch(stage=command, **args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
