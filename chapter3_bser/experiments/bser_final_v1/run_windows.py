"""Manual Windows-compatible paired F evaluation (plan by default)."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from chapter3_bser.experiments.safe_search_v1.run_paired import (
    ROOT, digest, execute_episode, file_hash, load_dependencies, output_directory, read_json)
from chapter3_bser.experiments.safe_search_v1.run_development import (
    CONFIG, INDICES, PLAN, SEED, validate_development_inputs)
from chapter3_bser.experiments.safe_search_v1.run_ac_development import validate_trace
from .options import ARMS, SETTINGS, parse_arms
from .analysis import summarize
from .provenance import framework_sources, verify_sources

DEFAULT_MANIFEST = ROOT / "3090结果/collision_terminal/B0_search_prior_eval100_seed12729_v1/evaluation_manifest.json"
BASELINES = {"F": "bser_final_v1"}
ZERO_FIELDS = ("actor_forward_calls", "action_sampling_calls", "optimizer_update_count",
               "residual_action_max_abs", "physical_residual_acceleration_max_abs")


def write_json(path, value):
    """Publish complete JSON atomically so an interruption remains resumable."""
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def request(manifest_path, config_path, baselines, arms, stage="development"):
    names = baselines.split(",")
    if not names or len(names) != len(set(names)) or any(x not in BASELINES for x in names):
        raise ValueError("this experiment has one controlled F framework")
    arms = parse_arms(arms)
    manifest, config = read_json(manifest_path), read_json(config_path)
    if config != read_json(CONFIG):
        raise ValueError("development requires the unchanged reference runtime configuration")
    if stage == "development":
        selected = validate_development_inputs(manifest, config, read_json(PLAN))
        seed = SEED
    else:
        from .scenarios import validate_confirmation
        selected, seed = validate_confirmation(manifest, config)
    count = len(selected)
    return dict(schema="ch3.bser_final.paired.v1", baselines=names, arms=arms,
        parent_variant="V4", stage=stage, acceptance_protocol="D2_reference_material_gain_guards",
        options={arm: ARMS[arm] for arm in arms}, settings=SETTINGS, selected=selected,
        independent_scenarios=count, planned_episode_runs=count*len(names)*len(arms),
        seed=seed, task_horizon=400, full_episodes=True, training=False,
        checkpoint_loaded=False, formal_thesis_evaluation=False, performance_passed=None,
        heldout_scenarios_executed=0, primary_metrics=["found_rate", "pre_found_collision_rate",
            "penalized_found_steps_mean_400"],
        primary_comparison="F1_vs_F0", mechanism_comparisons=["F1_vs_F2", "F1_vs_F3", "F1_vs_F4", "F1_vs_F5", "F1_vs_F6"],
        statistical_unit="paired_scenario", conditional_found_time_role="secondary",
        post_found_metrics_role="record_only_not_selection_objective",
        input_sha256={str(Path(p).resolve()): file_hash(p) for p in (manifest_path, config_path, PLAN)},
        sources=framework_sources())


def validate_terminal(row, scene, baseline, arm):
    for key in ("original_episode_index", "scenario_id", "scenario_seed", "environment_innovation_seed"):
        if row.get(key) != scene[key]:
            raise ValueError("episode identity mismatch: " + key)
    steps, found = row.get("physical_steps"), row.get("found_step")
    if (row.get("baseline") != BASELINES[baseline] or row.get("arm") != arm
            or row.get("parent_variant") != "V4"
            or any(row.get(k) is not True for k in ("terminal", "full_episode_completed", "full_episode_requested"))
            or type(steps) is not int or not 1 <= steps <= 400
            or found is not None and (type(found) is not int or not 0 <= found <= steps)
            or row.get("found_within_budget") is not (found is not None)):
        raise ValueError("expected an actual complete terminal episode")
    result, reason = row.get("episode_result", {}), row.get("stop_reason")
    collision = result.get("first_collision_step")
    if (reason not in ("success", "timeout", "obstacle_collision")
            or result.get("termination_reason") != reason or result.get("terminal_step") != steps
            or result.get("task_protocol") != "collision_terminal_v1"
            or result.get("success") is not (reason == "success")
            or reason == "success" and found is None
            or reason == "timeout" and steps != 400
            or (reason == "obstacle_collision") != (collision is not None)
            or collision is not None and (type(collision) is not int or collision != steps)):
        raise ValueError("inconsistent physical terminal result")
    exposure = found if found is not None else steps
    if (row.get("pre_found_exposure_steps") != exposure or row.get("pre_found_collision") is not
            (collision is not None and (found is None or collision <= found))):
        raise ValueError("inconsistent pre-Found metrics")
    if any(row.get("controller", {}).get(name) != 0 for name in ZERO_FIELDS):
        raise ValueError("learned action/training or physical residual detected")
    coverage = row.get("search_coverage") or {}
    if coverage.get("available") is not True or coverage.get("pre_found_exposure_steps") != exposure:
        raise ValueError("missing actual coverage diagnostics")
    options = row.get("controller", {}).get("bser_final_v1")
    if not options or options.get("arm") != arm or options.get("options") != ARMS[arm] or options.get("settings") != SETTINGS:
        raise ValueError("requested D intervention was not executed")
    complete = row.get("complete_episode_row") or {}
    expected = dict(method="ch3_bser_final_v1_"+arm, found=found is not None, found_step=found,
        actual_length=steps, episode_length=steps, termination_reason=reason,
        scenario_id=scene["scenario_id"], scenario_seed=scene["scenario_seed"],
        evaluation_episode_index=scene["original_episode_index"],
        environment_innovation_seed=scene["environment_innovation_seed"],
        training_update=False, optimizer_update_count=0)
    if any(complete.get(k) != v for k, v in expected.items()):
        raise ValueError("complete episode identity/outcome disagrees with D result")


def run_episode(args, plan):
    from .runtime import make_runtime
    from chapter3_bser.experiments.safe_search_v1.telemetry import capture_runtime
    from chapter3_bser.experiments.hgr.train import validate_config
    import torch
    torch.set_num_threads(1)
    baseline, arm, index = args.child_baseline, args.child_arm, args.child_index
    if baseline not in plan["baselines"] or arm not in plan["arms"] or index not in {s["original_episode_index"] for s in plan["selected"]}:
        raise ValueError("child not in requested development plan")
    selected = next(s for s in plan["selected"] if s["original_episode_index"] == index)
    scenario = read_json(args.manifest)["scenarios"][index]
    config = validate_config(read_json(args.config))
    output = output_directory(args.output_dir, (args.manifest, args.config, PLAN))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "identity.json", plan)
    try:
        deps = load_dependencies(config, scenario, index, plan["seed"]+index, "B1_bser_prior", search_coverage=True)
        def capture(runtime):
            record = capture_runtime(runtime)
            record["bser_final_v1"] = copy.deepcopy(runtime.controller_diagnostics()["bser_final_v1"])
            return record
        deps["capture_runtime"] = capture
        instances = []
        def factory():
            runtime = make_runtime(copy.deepcopy(config), copy.deepcopy(scenario),
                arm=arm, seed=plan["seed"]+index, episode_id=index)
            instances.append(runtime)
            return runtime
        episode_started = time.perf_counter()
        signatures, row = execute_episode(factory, output, steps=400, observed=True, deps=deps, full_episodes=True)
        row["episode_wall_seconds"] = time.perf_counter()-episode_started
        audit = dict(common=instances[0].controller.allocator.audit,
            final=getattr(instances[0].controller.allocator, "final_audit", []),
            prediction_errors=instances[0].final_error_records)
        write_json(output / "planning_audit.json", audit)
        row["planning_audit_sha256"] = file_hash(output / "planning_audit.json")
        row["complete_episode_row"].update(method="ch3_bser_final_v1_"+arm,
            reference_baseline="B1_bser_prior", experiment_protocol="bser_final_v1")
        verify_sources(plan["sources"])
        if any(file_hash(p) != sha for p, sha in plan["input_sha256"].items()):
            raise ValueError("inputs changed during child execution")
        row.update(selected, baseline=BASELINES[baseline], arm=arm, variant=arm,
                   parent_variant="V4", signature_sha256=digest(signatures))
        validate_terminal(row, selected, baseline, arm)
        row["trace_sha256"] = validate_trace(output / "step_trace.jsonl", row)
        row["source_and_input_verification_passed"] = True
        write_json(output / "result.json", row)
    except BaseException as exc:
        write_json(output / "failure.json", dict(type=type(exc).__name__, message=str(exc), experiment_complete=False))
        raise


def read_completed(directory, plan, baseline, arm, scene):
    directory = Path(directory)
    if (directory / "failure.json").exists() or read_json(directory / "identity.json") != plan:
        raise ValueError("child failed or used a different source/input/plan")
    row = read_json(directory / "result.json")
    validate_terminal(row, scene, baseline, arm)
    if row.get("source_and_input_verification_passed") is not True or row.get("trace_sha256") != validate_trace(directory / "step_trace.jsonl", row):
        raise ValueError("unverified or changed child trace")
    if row.get("planning_audit_sha256") != file_hash(directory / "planning_audit.json"):
        raise ValueError("changed planning audit")
    return row


def run_jobs(args, plan):
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in [1,8]")
    output = Path(args.output_dir).resolve()
    runs = (ROOT / "runs").resolve()
    if runs not in output.parents:
        raise ValueError("manual development outputs must be a new subdirectory under runs/")
    rows = []
    if args.resume:
        if read_json(output / "identity.json") != plan:
            raise ValueError("resume requires identical sources, settings, scenarios and input paths")
        for entry in read_json(output / "episodes.json"):
            baseline = next(k for k, v in BASELINES.items() if v == entry["baseline"])
            scene = next(s for s in plan["selected"] if s["original_episode_index"] == entry["original_episode_index"])
            directory = (output / entry["artifact_dir"]).resolve()
            if output not in directory.parents:
                raise ValueError("resume artifact escaped output directory")
            row = read_completed(directory, plan, baseline, entry["arm"], scene)
            if dict(row, artifact_dir=entry["artifact_dir"]) != entry:
                raise ValueError("resume row differs from original artifact")
            rows.append(entry)
    else:
        output_directory(output, (args.manifest, args.config, PLAN))
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "identity.json", plan)
        write_json(output / "episodes.json", rows)
    key = lambda r: (r["baseline"], r["arm"], r["original_episode_index"])
    done = {key(r) for r in rows}
    if len(done) != len(rows):
        raise ValueError("duplicate completed episode identities")
    jobs = [(b, a, s) for s in plan["selected"] for b in plan["baselines"] for a in plan["arms"]
            if (BASELINES[b], a, s["original_episode_index"]) not in done]
    running, cursor = [], 0
    started = time.perf_counter()
    def verify():
        verify_sources(plan["sources"])
        if any(file_hash(p) != sha for p, sha in plan["input_sha256"].items()):
            raise ValueError("input changed during experiment")
    try:
        while cursor < len(jobs) or running:
            while cursor < len(jobs) and len(running) < args.workers:
                verify()
                b, a, scene = jobs[cursor]
                index = scene["original_episode_index"]
                parent = output / "jobs" / f"{b}_{a}_{index:04d}"
                attempt = 1
                while (parent / f"attempt_{attempt}").exists():
                    attempt += 1
                directory = parent / f"attempt_{attempt}"
                directory.mkdir(parents=True)
                log_path = parent / f"attempt_{attempt}.log"
                handle = log_path.open("x", encoding="utf-8")
                command = [sys.executable, "-B", "-m", "chapter3_bser.experiments.bser_final_v1.run_windows",
                    "--manifest", str(Path(args.manifest).resolve()), "--config", str(Path(args.config).resolve()),
                    "--output-dir", str(directory), "--baselines", args.baselines, "--arms", args.arms,
                    "--stage", args.stage, "--execute", "--child-index", str(index), "--child-baseline", b, "--child-arm", a]
                env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                try:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                except BaseException:
                    handle.close()
                    raise
                running.append((process, handle, directory, b, a, scene))
                cursor += 1
                print(f"BSER_FINAL_START {b} {a} scene={index} attempt={attempt}", flush=True)
            for item in list(running):
                process, handle, directory, b, a, scene = item
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                running.remove(item)
                if code != 0:
                    raise RuntimeError(f"{b}/{a}/{scene['original_episode_index']} exited {code}; see {directory.parent}")
                verify()
                row = read_completed(directory, plan, b, a, scene)
                row["artifact_dir"] = directory.relative_to(output).as_posix()
                rows.append(row)
                rows.sort(key=key)
                write_json(output / "episodes.json", rows)
                interim = summarize(rows, plan, intervals=False)
                interim["experiment_complete"] = False
                interim["source_and_input_verification_passed"] = False
                write_json(output / "summary.json", interim)
                write_json(output / "progress.json", dict(completed=len(rows), planned=plan["planned_episode_runs"],
                    experiment_complete=False, elapsed_seconds=time.perf_counter()-started))
                print(f"BSER_FINAL_DONE {b} {a} scene={scene['original_episode_index']} found={row['found_within_budget']} "
                      f"stop={row['stop_reason']} completed={len(rows)}/{plan['planned_episode_runs']}", flush=True)
            if running:
                time.sleep(.25)
        verify()
        summary = summarize(rows, plan)
        summary["source_and_input_verification_passed"] = True
        write_json(output / "summary.json", summary)
        write_json(output / "progress.json", dict(completed=len(rows), planned=plan["planned_episode_runs"],
            experiment_complete=summary["experiment_complete"], elapsed_seconds=time.perf_counter()-started))
        print("BSER_FINAL_FINISHED", flush=True)
        return summary
    except BaseException as exc:
        # Stop only children owned by this invocation. Completed artifacts stay
        # untouched; a resume creates a new attempt for each unfinished job.
        for process, handle, *_ in running:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            handle.close()
        incomplete = summarize(rows, plan, intervals=False)
        incomplete["experiment_complete"] = False
        incomplete["source_and_input_verification_passed"] = False
        write_json(output / "summary.json", incomplete)
        write_json(output / f"interruption_{time.time_ns()}.json", dict(type=type(exc).__name__,
            message=str(exc), completed=len(rows), experiment_complete=False))
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--baselines", default="F", help=argparse.SUPPRESS)
    p.add_argument("--arms", default="F0,F1,F2,F3,F4,F5,F6")
    p.add_argument("--stage", choices=("development", "confirmation"), default="development")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--execute", action="store_true", help="run episodes; without this flag validate and print the plan only")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--child-index", type=int, help=argparse.SUPPRESS)
    p.add_argument("--child-baseline", help=argparse.SUPPRESS)
    p.add_argument("--child-arm", help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in [1,8]")
    plan = request(args.manifest, args.config, args.baselines, args.arms, args.stage)
    if not args.execute:
        print(json.dumps({k: v for k, v in plan.items() if k not in ("sources", "selected")}, ensure_ascii=False, indent=2))
        print("PLAN_ONLY: no simulator, training, checkpoint or episode was started.")
    elif args.child_index is not None:
        run_episode(args, plan)
    else:
        run_jobs(args, plan)


if __name__ == "__main__":
    main()
