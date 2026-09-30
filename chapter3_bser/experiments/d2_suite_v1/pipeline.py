"""One manual launch: all training, then all evaluation, then paired summary."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback
import uuid

from .linux import Tee, check, run_lock
from .plan import load_plan, write
from .run import execute, receipt
from .summarize import summarize


def require_complete(root, stage):
    """Verify the entire preceding phase, including final checkpoint bytes."""
    plan = load_plan(root)
    missing = [j["name"] for j in plan["jobs"]
               if not (stage == "train" and j["arm"] == "D2")
               and receipt(root, plan, j, stage) is None]
    if missing:
        raise ValueError(stage + " phase is incomplete; missing verified receipts: " + ", ".join(missing))


def run(root, *, confirmed=False):
    root = Path(root).resolve()
    preflight = check(root)
    phases = {stage: execute(root, stage=stage, execute=False) for stage in ("train", "evaluate")}
    if not confirmed:
        return dict(executed=False, preflight=preflight, phases=phases,
                    order=["all training", "all evaluation", "summary"],
                    d2_checkpoint_required=False,
                    note="Read-only preview. Add --execute to run the complete pipeline manually.")
    # Keep one shared lock throughout all phases. Call the underlying executor
    # directly to avoid nesting the standalone launcher's identical lock.
    with run_lock(root):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        directory = root / "launch_logs" / ("pipeline_" + stamp)
        directory.mkdir(parents=True, exist_ok=False)
        report = directory / "report"
        state = dict(status="running", stage="train", preflight=preflight, results={},
                     d2_checkpoint_required=False, report_directory=str(report),
                     started_utc=datetime.now(timezone.utc).isoformat())
        write(directory / "pipeline.json", state)
        with (directory / "console.log").open("x", encoding="utf-8", buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                try:
                    print("pipeline directory: " + str(directory), flush=True)
                    print(json.dumps(preflight, ensure_ascii=False), flush=True)
                    for stage in ("train", "evaluate"):
                        state["stage"] = stage
                        write(directory / "pipeline.json", state)
                        print("phase: " + stage, flush=True)
                        state["results"][stage] = execute(root, stage=stage, execute=True)
                        # This barrier runs before even the checkpoint-free D2
                        # evaluation: all requested training must finish first.
                        require_complete(root, stage)
                        write(directory / "pipeline.json", state)
                    state["stage"] = "summarize"
                    write(directory / "pipeline.json", state)
                    result = summarize(root, report)
                    if not result["suite_complete"]:
                        raise ValueError("pipeline cannot complete with a partial summary")
                    load_plan(root)
                    state.update(status="complete", stage="complete", summary=result)
                except BaseException as exc:
                    state.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                    traceback.print_exc()
                    raise
                finally:
                    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
                    write(directory / "pipeline.json", state)
        return dict(executed=True, suite_complete=True, launch_directory=str(directory),
                    report_directory=str(report), phases=state["results"], d2_checkpoint_required=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="existing prepared suite directory")
    parser.add_argument("--execute", dest="confirmed", action="store_true",
                        help="run every training job before any evaluation; default is preview only")
    result = run(**vars(parser.parse_args(argv)))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
