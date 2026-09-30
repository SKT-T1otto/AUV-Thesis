"""Explicit, checkpoint-free D2 evaluation of the frozen 20-scene split.

Each child collects one scenario serially across V0--V4. Parallelism only
separates independent processes; scenario indices and simulator seeds stay
unchanged. A complete-stage result requires all 100 terminal arm records.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import os
from pathlib import Path
import subprocess
import sys
import time

from .run_paired import (ROOT, VARIANTS, digest, file_hash, output_directory,
                         read_json, summarize, validate_manifest, write_json)

PLAN = ROOT / "docs/chapter3/search_diagnostics/3090_20260928/experiment_plan.json"
CONFIG = ROOT / "configs/chapter3/hgr_train.json"
INDICES = (0, 3, 7, 10, 22, 28, 31, 40, 42, 46, 62, 63, 67, 76, 77, 78, 86, 91, 95, 97)
SEED = 12729
BASELINE = "B0_search_prior"
IDENTITY_SCHEMA = "ch3.safe_search.development_identity.v1"


def validate_development_inputs(manifest, config, plan):
    """Require the frozen population and original scenario content, not a subset manifest."""
    if config.get("max_steps") != 400:
        raise ValueError("D2 requires the unchanged 400-step task horizon")
    if len(manifest.get("scenarios", [])) != 100:
        raise ValueError("D2 requires the complete original 100-scene manifest")
    if plan.get("schema") != "ch3.safe_search.experiment_plan.v1":
        raise ValueError("unexpected frozen experiment plan schema")
    records = plan.get("splits", {}).get("development", [])
    if [r.get("source_episode_index") for r in records] != list(INDICES):
        raise ValueError("development split must contain the exact 20 frozen original indices")
    selected = validate_manifest(manifest, config, ",".join(map(str, INDICES)), SEED)
    result = []
    for expected, (index, scene) in zip(records, selected):
        actual = dict(source_episode_index=index, scenario_id=scene["scenario_id"],
                      scenario_seed=scene["scenario_seed"],
                      environment_innovation_seed=SEED + index,
                      scene_content_sha256=digest(scene))
        if actual != expected:
            raise ValueError(f"frozen scene identity/content mismatch at original index {index}")
        result.append(dict(original_episode_index=index, scenario_id=scene["scenario_id"],
                           scenario_seed=scene["scenario_seed"],
                           environment_innovation_seed=SEED + index,
                           scenario_sha256=digest(scene)))
    return result


def child_command(manifest, config, output, index):
    return [sys.executable, "-B", "-m", "chapter3_bser.experiments.safe_search_v1.run_paired",
            "--manifest", str(manifest), "--config", str(config),
            "--output-dir", str(output), "--baseline", BASELINE,
            "--variants", ",".join(VARIANTS), "--episode-indices", str(index),
            "--steps", "400", "--seed", str(SEED), "--full-episodes", "--search-coverage"]


def _run_child(command, log_path):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    started = time.perf_counter()
    with Path(log_path).open("x", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, check=False)
    return dict(returncode=completed.returncode, wall_seconds=time.perf_counter() - started)


def validate_child(output, expected, parent):
    """Read one finished child; never turn a process failure into a task outcome."""
    output = Path(output)
    if (output / "failure.json").exists():
        raise ValueError(f"child has a recorded program failure: {output.name}")
    identity = read_json(output / "identity.json")
    summary = read_json(output / "summary.json")
    rows = read_json(output / "episodes.json")
    required = dict(schema="ch3.safe_search.paired_identity.v1", baseline=BASELINE,
                    variants=list(VARIANTS), seed=SEED, steps=400, task_horizon=400,
                    full_episodes=True, training=False, checkpoint_loaded=False,
                    formal_thesis_evaluation=False, selected=[expected],
                    search_coverage_enabled=True,
                    input_sha256=parent["child_input_sha256"],
                    sources_before=parent["sources_before"],
                    sources_after=parent["sources_before"],
                    source_and_input_verification_passed=True)
    for key, value in required.items():
        if identity.get(key) != value:
            raise ValueError(f"child identity mismatch for {key}: {output.name}")
    if (summary.get("source_and_input_verification_passed") is not True
            or summary.get("all_requested_runs_recorded") is not True):
        raise ValueError(f"child source/completion verification missing: {output.name}")
    if not isinstance(rows, list) or len(rows) != len(VARIANTS):
        raise ValueError(f"child must have exactly five arms: {output.name}")
    if [row.get("variant") for row in rows] != list(VARIANTS):
        raise ValueError(f"child contains missing, duplicate, or reordered arms: {output.name}")
    for row in rows:
        for key in ("original_episode_index", "scenario_id", "scenario_seed", "environment_innovation_seed"):
            if row.get(key) != expected[key]:
                raise ValueError(f"arm identity mismatch for {key}: {output.name}")
        steps = row.get("physical_steps")
        if (row.get("baseline") != BASELINE or row.get("terminal") is not True
                or row.get("full_episode_completed") is not True
                or row.get("full_episode_requested") is not True
                or row.get("stop_reason") in (None, "budget_cutoff")
                or type(steps) is not int or not 1 <= steps <= 400
                or not isinstance(row.get("complete_episode_row"), dict)
                or type(row.get("found_within_budget")) is not bool):
            raise ValueError(f"nonterminal or invalid full-episode record: {output.name}/{row['variant']}")
        arm = output / f"episode_{expected['original_episode_index']:04d}" / row["variant"] / "summary.json"
        if read_json(arm) != row:
            raise ValueError(f"child row differs from arm artifact: {output.name}/{row['variant']}")
        arm_summary = summary.get("variants", {}).get(row["variant"], {})
        if (arm_summary.get("n_expected") != 1 or arm_summary.get("n_recorded") != 1
                or arm_summary.get("n_terminal") != 1
                or arm_summary.get("full_episode_evaluation_complete") is not True):
            raise ValueError(f"child arm summary incomplete: {output.name}/{row['variant']}")
    return rows


def collect_completed(output_dir, *, require_root_complete=True, plan=None):
    """Strict aggregation input: exactly 20 x 5 complete, identically sourced records."""
    output = Path(output_dir)
    parent = read_json(output / "identity.json")
    plan = read_json(PLAN) if plan is None else plan
    if (output / "failure.json").exists():
        raise ValueError("development has a recorded program failure")
    if require_root_complete and (parent.get("experiment_complete") is not True
            or parent.get("source_and_input_verification_passed") is not True
            or parent.get("sources_after") != parent.get("sources_before")):
        raise ValueError("development root has not completed source/input verification")
    selected = parent.get("selected", [])
    if (parent.get("schema") != IDENTITY_SCHEMA
            or [r.get("original_episode_index") for r in selected] != list(INDICES)
            or parent.get("variants") != list(VARIANTS)
            or parent.get("planned_episode_runs") != 100
            or parent.get("max_physical_steps") != 40000
            or parent.get("seed") != SEED or parent.get("task_horizon") != 400
            or parent.get("baseline") != BASELINE
            or parent.get("search_coverage_enabled") is not True
            or parent.get("formal_thesis_evaluation") is not False
            or parent.get("training") is not False or parent.get("checkpoint_loaded") is not False):
        raise ValueError("invalid fixed-development root identity")
    if "sources_after" in parent and parent["sources_after"] != parent["sources_before"]:
        raise ValueError("root source changed during development evaluation")
    frozen = plan.get("splits", {}).get("development", [])
    if (plan.get("schema") != "ch3.safe_search.experiment_plan.v1"
            or [r.get("source_episode_index") for r in frozen] != list(INDICES)):
        raise ValueError("aggregation requires the fixed development plan")
    expected_selection = [dict(original_episode_index=r["source_episode_index"],
                               scenario_id=r["scenario_id"], scenario_seed=r["scenario_seed"],
                               environment_innovation_seed=r["environment_innovation_seed"],
                               scenario_sha256=r["scene_content_sha256"]) for r in frozen]
    if selected != expected_selection:
        raise ValueError("root scene identity/content differs from the frozen development plan")
    rows = []
    for expected in selected:
        rows.extend(validate_child(output / f"scene_{expected['original_episode_index']:04d}", expected, parent))
    if len(rows) != 100 or len({(r["original_episode_index"], r["variant"]) for r in rows}) != 100:
        raise ValueError("development aggregation requires exactly 100 unique complete arms")
    return parent, rows


def run_development(manifest_path, output_dir, *, config_path=CONFIG, workers=4):
    if type(workers) is not int or not 1 <= workers <= 20:
        raise ValueError("workers must be an integer in 1..20")
    manifest_path, config_path = Path(manifest_path).resolve(), Path(config_path).resolve()
    inputs = (manifest_path, config_path, PLAN)
    output = output_directory(output_dir, inputs)
    manifest, config, plan = (read_json(p) for p in inputs)
    selected = validate_development_inputs(manifest, config, plan)
    # The reviewed gate is deliberately lazy so CLI help and unit tests need
    # neither the simulator nor a running experiment.
    from .provenance import framework_sources, verify_sources
    sources = framework_sources()
    hashes = {str(p): file_hash(p) for p in inputs}

    def verify():
        verify_sources(sources)
        if any(file_hash(p) != sha for p, sha in hashes.items()):
            raise RuntimeError("development input changed during execution")

    identity = dict(schema=IDENTITY_SCHEMA, stage="D2", baseline=BASELINE,
                    variants=list(VARIANTS), seed=SEED, task_horizon=400,
                    full_episodes=True, selected=selected, planned_episode_runs=100,
                    search_coverage_enabled=True,
                    max_physical_steps=40000, independent_scenarios=20,
                    workers=workers, collection_mode="serial_arms_in_independent_scene_processes",
                    user_authorized_development=True, training=False, checkpoint_loaded=False,
                    formal_thesis_evaluation=False, performance_passed=None,
                    input_sha256=hashes,
                    child_input_sha256={str(p): hashes[str(p)] for p in (config_path, manifest_path)},
                    sources_before=sources, python=sys.version,
                    heldout_scenarios_executed=0)
    output_directory(output, inputs)
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir()
    write_json(output / "identity.json", identity)
    pending, completed, failures = deque(selected), [], []
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            running = {}
            while pending or running:
                while pending and len(running) < workers and not failures:
                    verify()
                    item = pending.popleft()
                    index = item["original_episode_index"]
                    child = output / f"scene_{index:04d}"
                    future = executor.submit(_run_child,
                        child_command(manifest_path, config_path, child, index),
                        output / "logs" / f"scene_{index:04d}.log")
                    running[future] = item
                    print(f"SAFE_SEARCH_D2_START index={index}", flush=True)
                if not running:
                    break
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    item = running.pop(future)
                    index = item["original_episode_index"]
                    try:
                        result = future.result()
                        if result["returncode"] != 0:
                            raise RuntimeError(f"child exited with status {result['returncode']}")
                        verify()
                        validate_child(output / f"scene_{index:04d}", item, identity)
                        completed.append(index)
                        print(f"SAFE_SEARCH_D2_COMPLETE index={index} scenes={len(completed)}/20", flush=True)
                    except Exception as exc:
                        failures.append(dict(original_episode_index=index,
                            exception_type=type(exc).__name__, message=str(exc)))
                        print(f"SAFE_SEARCH_D2_FAILURE index={index} {exc}", flush=True)
                write_json(output / "progress.json", dict(completed_original_indices=sorted(completed),
                    running_original_indices=sorted(r["original_episode_index"] for r in running.values()),
                    unstarted_original_indices=[r["original_episode_index"] for r in pending],
                    program_failures=failures, planned_episode_runs=100,
                    experiment_complete=False))
        if failures:
            raise RuntimeError(f"D2 has {len(failures)} failed child process(es); no complete-stage verdict")
        verify()
        _, rows = collect_completed(output, require_root_complete=False)
        summary = summarize(rows, list(VARIANTS), 20, True)
        summary.update(stage="D2", schema="ch3.safe_search.development_summary.v1",
                       planned_episode_runs=100, program_failures=[], experiment_complete=True,
                       source_and_input_verification_passed=True,
                       formal_thesis_evaluation=False, wall_seconds=time.perf_counter() - started)
        identity.update(sources_after=framework_sources(), source_and_input_verification_passed=True,
                        experiment_complete=True)
        write_json(output / "episodes.json", rows)
        write_json(output / "identity.json", identity)
        write_json(output / "summary.json", summary)
        write_json(output / "progress.json", dict(completed_original_indices=list(INDICES),
                   running_original_indices=[], unstarted_original_indices=[],
                   program_failures=[], planned_episode_runs=100, experiment_complete=True))
        print("SAFE_SEARCH_D2_FINISHED scenes=20 arms=100", flush=True)
        return summary
    except BaseException as exc:
        write_json(output / "failure.json", dict(exception_type=type(exc).__name__, message=str(exc),
                   completed_original_indices=sorted(completed), program_failures=failures,
                   unstarted_original_indices=[r["original_episode_index"] for r in pending],
                   planned_episode_runs=100, experiment_complete=False))
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", required=True, type=Path)
    result.add_argument("--config", type=Path, default=CONFIG)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--execute", required=True, action="store_true",
                        help="explicitly execute the fixed 100-arm development evaluation")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    run_development(args.manifest, args.output_dir, config_path=args.config, workers=args.workers)


if __name__ == "__main__":
    main()
