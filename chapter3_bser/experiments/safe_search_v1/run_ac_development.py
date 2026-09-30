"""Explicit V5=A+C development extension; never overwrites the historical D2.

Runs the same 20 original scenes with a new, separately identified source seal.
The completed historical V0--V4 cohort is read-only reference evidence, not
silently relabelled as having run under the new source. No checkpoint/training.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from pathlib import Path
import sys
import time

from .run_development import (BASELINE, CONFIG, INDICES, PLAN, SEED, _run_child,
                             collect_completed, validate_development_inputs)
from .run_paired import ROOT, file_hash, output_directory, read_json, summarize, write_json

AC_PLAN = ROOT / "docs/chapter3/search_diagnostics/safe_search_v1/ac_experiment_plan.json"
VARIANT = "V5"
OPTIONS = dict(planning_state_consistency=True, failure_policy=False, path_safety=True)
IDENTITY_SCHEMA = "ch3.safe_search.ac_development_identity.v1"


def validate_ac_plan(plan, reference_identity, identity_sha256, episodes_sha256):
    options = plan.get("options")
    if (plan.get("schema") != "ch3.safe_search.ac_plan.v1" or plan.get("variant") != VARIANT
            or options != OPTIONS or not isinstance(options, dict)
            or any(type(value) is not bool for value in options.values())
            or plan.get("development_indices") != list(INDICES)
            or any(type(index) is not int for index in plan.get("development_indices", []))
            or type(plan.get("seed")) is not int or plan["seed"] != SEED
            or type(plan.get("max_steps")) is not int or plan["max_steps"] != 400):
        raise ValueError("AC plan must specify only V5=A+C on the exact frozen 20 scenes/seed/horizon")
    expected = dict(reference_identity_sha256=identity_sha256,
                    reference_episodes_sha256=episodes_sha256,
                    reference_source_inventory_sha256=reference_identity["sources_before"]["inventory"]["sha256"])
    if any(plan.get(name) != value for name, value in expected.items()):
        raise ValueError("AC plan does not bind the selected completed historical D2 reference")


def read_reference(reference_dir):
    reference_dir = Path(reference_dir).resolve()
    identity, rows = collect_completed(reference_dir)
    if read_json(reference_dir / "episodes.json") != rows:
        raise ValueError("reference root episodes differ from its complete 100 child/arm records")
    from .analyze_development import validate_rows
    validate_rows(rows, read_json(PLAN))
    return identity, rows, dict(run_directory=str(reference_dir),
        identity_sha256=file_hash(reference_dir / "identity.json"),
        episodes_sha256=file_hash(reference_dir / "episodes.json"),
        source_inventory_sha256=identity["sources_before"]["inventory"]["sha256"],
        source_evolution_sha256=identity["sources_before"]["evolution_sha256"],
        source_checkout_profile=identity["sources_before"]["checkout_profile"],
        input_sha256=identity["child_input_sha256"], selected=identity["selected"],
        completed_episode_runs=100, reference_complete_verified=True)


def child_command(manifest, config, output, index):
    return [sys.executable, "-B", "-m", "chapter3_bser.experiments.safe_search_v1.run_paired",
        "--manifest", str(manifest), "--config", str(config), "--output-dir", str(output),
        "--baseline", BASELINE, "--variants", VARIANT, "--episode-indices", str(index),
        "--steps", "400", "--seed", str(SEED), "--full-episodes", "--search-coverage"]


def validate_terminal(row, selected):
    for name in ("original_episode_index", "scenario_id", "scenario_seed", "environment_innovation_seed"):
        if row.get(name) != selected[name]:
            raise ValueError("AC arm identity differs from the frozen scene: " + name)
    steps, found = row.get("physical_steps"), row.get("found_step")
    if (row.get("variant") != VARIANT or row.get("baseline") != BASELINE
            or row.get("terminal") is not True or row.get("full_episode_completed") is not True
            or row.get("full_episode_requested") is not True
            or type(steps) is not int or not 1 <= steps <= 400
            or type(row.get("found_within_budget")) is not bool
            or row["found_within_budget"] != (found is not None)
            or found is not None and (type(found) is not int or not 0 <= found <= steps)):
        raise ValueError("AC arm must be an actual complete full-horizon task terminal")
    reason, result = row.get("stop_reason"), row.get("episode_result", {})
    if (reason not in ("success", "timeout", "obstacle_collision")
            or reason != result.get("termination_reason") or result.get("terminal_step") != steps
            or result.get("task_protocol") != "collision_terminal_v1"
            or reason == "timeout" and steps != 400
            or result.get("success") is not (reason == "success")
            or reason == "success" and found is None):
        raise ValueError("AC task outcome/protocol is inconsistent")
    collision = result.get("first_collision_step")
    if ((reason == "obstacle_collision") != (collision is not None)
            or collision is not None and (type(collision) is not int or collision != steps)):
        raise ValueError("collision terminal step is inconsistent")
    pre_collision = collision is not None and (found is None or collision <= found)
    exposure = found if found is not None else steps
    if row.get("pre_found_collision") is not pre_collision or row.get("pre_found_exposure_steps") != exposure:
        raise ValueError("AC pre-Found outcome/exposure is inconsistent")
    for name in ("searcher_motion_stall_proxy_agent_steps", "searcher_hold_agent_steps"):
        if type(row.get(name)) is not int or not 0 <= row[name] <= 3 * exposure:
            raise ValueError("AC search-agent step metric is invalid")
    complete = row.get("complete_episode_row", {})
    expected = dict(found=found is not None, found_step=found, success=reason == "success",
        actual_length=steps, episode_length=steps, terminal_step=steps, termination_reason=reason,
        scenario_id=selected["scenario_id"], scenario_seed=selected["scenario_seed"],
        environment_innovation_seed=selected["environment_innovation_seed"],
        evaluation_episode_index=selected["original_episode_index"], optimizer_update_count=0,
        training_update=False, first_collision_step=collision)
    if any(name not in complete or complete[name] != value for name, value in expected.items()):
        raise ValueError("AC complete_episode_row differs from the terminal summary")
    coverage = row.get("search_coverage")
    if not isinstance(coverage, dict) or coverage.get("pre_found_exposure_steps") != exposure:
        raise ValueError("AC requested coverage diagnostics are missing or have different exposure")


def validate_trace(path, row):
    """Require a contiguous physical trace, including Found and terminal transitions."""
    def reject(value):
        raise ValueError("nonfinite trace value: " + value)
    n = exposure = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line, parse_constant=reject)
            n += 1
            if (record.get("step_before") != n - 1 or record.get("step_after") != n
                    or record.get("before", {}).get("step") != n - 1
                    or record.get("after", {}).get("step") != n or n > row["physical_steps"]):
                raise ValueError("AC trace is not the complete contiguous physical episode")
            found = row["found_step"]
            before_found, after_found = found is not None and found <= n - 1, found is not None and found <= n
            if (record["before"].get("found") is not before_found
                    or record["after"].get("found") is not after_found
                    or record.get("search_transition") is not (not before_found)):
                raise ValueError("AC trace Found transition disagrees with terminal summary")
            coverage = record.get("search_coverage", {})
            if coverage.get("step") != n or coverage.get("pre_found_exposure_steps") != int(not before_found):
                raise ValueError("AC trace coverage timestamp/exposure mismatch")
            exposure += int(not before_found)
    if n != row["physical_steps"] or exposure != row["pre_found_exposure_steps"]:
        raise ValueError("AC trace is truncated or differs from the terminal search exposure")
    return file_hash(path)


def validate_child_ac(output, selected, parent):
    output = Path(output)
    if (output / "failure.json").exists():
        raise ValueError("AC child contains a program failure")
    identity, summary, rows = (read_json(output / name) for name in ("identity.json", "summary.json", "episodes.json"))
    required = dict(schema="ch3.safe_search.paired_identity.v1", baseline=BASELINE, variants=[VARIANT],
        seed=SEED, steps=400, task_horizon=400, full_episodes=True, training=False, checkpoint_loaded=False,
        formal_thesis_evaluation=False, selected=[selected], search_coverage_enabled=True,
        input_sha256=parent["child_input_sha256"], sources_before=parent["sources_before"],
        sources_after=parent["sources_before"], source_and_input_verification_passed=True)
    if any(identity.get(name) != value for name, value in required.items()):
        raise ValueError("AC child source/input/request identity differs from its parent")
    if (not isinstance(rows, list) or len(rows) != 1
            or summary.get("source_and_input_verification_passed") is not True
            or summary.get("all_requested_runs_recorded") is not True
            or set(summary.get("variants", {})) != {VARIANT}):
        raise ValueError("AC child must contain exactly one verified V5 arm")
    row = rows[0]
    validate_terminal(row, selected)
    arm = output / f"episode_{selected['original_episode_index']:04d}" / VARIANT
    if read_json(arm / "summary.json") != row:
        raise ValueError("AC terminal row differs from its arm artifact")
    variant = summary["variants"][VARIANT]
    if (variant.get("n_expected") != 1 or variant.get("n_recorded") != 1 or variant.get("n_terminal") != 1
            or variant.get("full_episode_evaluation_complete") is not True):
        raise ValueError("AC arm summary is incomplete")
    return row, validate_trace(arm / "step_trace.jsonl", row)


def collect_completed_ac(output_dir, *, require_root_complete=True):
    output = Path(output_dir)
    identity = read_json(output / "identity.json")
    if (output / "failure.json").exists():
        raise ValueError("AC development contains a recorded program failure")
    if (identity.get("schema") != IDENTITY_SCHEMA or identity.get("variants") != [VARIANT]
            or identity.get("options") != OPTIONS or identity.get("baseline") != BASELINE
            or identity.get("planned_episode_runs") != 20 or identity.get("max_physical_steps") != 8000
            or identity.get("seed") != SEED or identity.get("task_horizon") != 400
            or identity.get("training") is not False or identity.get("checkpoint_loaded") is not False
            or identity.get("formal_thesis_evaluation") is not False
            or identity.get("search_coverage_enabled") is not True):
        raise ValueError("invalid AC development root identity")
    if require_root_complete and (identity.get("experiment_complete") is not True
            or identity.get("source_and_input_verification_passed") is not True
            or identity.get("sources_after") != identity.get("sources_before")):
        raise ValueError("AC root source/input completion verification is missing")
    ac_plan = read_json(AC_PLAN)
    reference = identity["reference"]
    validate_ac_plan(ac_plan, {"sources_before": {"inventory": {"sha256": reference["source_inventory_sha256"]}}},
                     reference["identity_sha256"], reference["episodes_sha256"])
    if identity.get("ac_plan_sha256") != file_hash(AC_PLAN):
        raise ValueError("AC plan changed from the executed request")
    frozen = read_json(PLAN)["splits"]["development"]
    selected = [dict(original_episode_index=r["source_episode_index"], scenario_id=r["scenario_id"],
        scenario_seed=r["scenario_seed"], environment_innovation_seed=r["environment_innovation_seed"],
        scenario_sha256=r["scene_content_sha256"]) for r in frozen]
    if ([r["original_episode_index"] for r in selected] != list(INDICES)
            or identity.get("selected") != selected or reference.get("selected") != selected
            or reference.get("completed_episode_runs") != 100 or reference.get("reference_complete_verified") is not True):
        raise ValueError("AC and historical reference must use the frozen 20 scene identities")
    if sorted(identity["child_input_sha256"].values()) != sorted(reference["input_sha256"].values()):
        raise ValueError("AC input bytes differ from the historical comparison")
    expected_source_change = reference["source_inventory_sha256"] != identity["sources_before"]["inventory"]["sha256"]
    if identity.get("source_changed_from_reference") is not expected_source_change:
        raise ValueError("AC reference source difference is mislabelled")
    rows, trace_hashes = [], {}
    for scene in selected:
        row, trace_hash = validate_child_ac(output / f"scene_{scene['original_episode_index']:04d}", scene, identity)
        rows.append(row)
        trace_hashes[str(scene["original_episode_index"])] = trace_hash
    if identity.get("trace_sha256") != trace_hashes:
        raise ValueError("AC trace contents differ from the completed child fingerprints")
    if require_root_complete and read_json(output / "episodes.json") != rows:
        raise ValueError("AC root episodes differ from the 20 complete child records")
    return identity, rows


def run_ac_development(manifest_path, output_dir, *, reference_dir, config_path=CONFIG, workers=4):
    if type(workers) is not int or not 1 <= workers <= 20:
        raise ValueError("workers must be an integer in 1..20")
    manifest_path, config_path, reference_dir = Path(manifest_path).resolve(), Path(config_path).resolve(), Path(reference_dir).resolve()
    inputs = (manifest_path, config_path, PLAN, AC_PLAN, reference_dir / "identity.json", reference_dir / "episodes.json")
    output = output_directory(output_dir, inputs)
    if output == reference_dir or reference_dir in output.parents:
        raise ValueError("AC output must be separate from the retained historical D2 reference")
    selected = validate_development_inputs(read_json(manifest_path), read_json(config_path), read_json(PLAN))
    reference_identity, _, reference = read_reference(reference_dir)
    validate_ac_plan(read_json(AC_PLAN), reference_identity, reference["identity_sha256"], reference["episodes_sha256"])
    if reference_identity["selected"] != selected:
        raise ValueError("AC scene identities differ from the verified historical D2 reference")
    hashes = {str(path): file_hash(path) for path in inputs}
    child_inputs = {str(path): hashes[str(path)] for path in (config_path, manifest_path)}
    if len(reference["input_sha256"]) != 2 or sorted(child_inputs.values()) != sorted(reference["input_sha256"].values()):
        raise ValueError("AC manifest/config bytes must equal the historical reference inputs")
    from .provenance import framework_sources, verify_sources
    sources = framework_sources()
    def verify():
        verify_sources(sources)
        if any(file_hash(path) != value for path, value in hashes.items()):
            raise RuntimeError("AC source/input/reference identity changed during evaluation")
    identity = dict(schema=IDENTITY_SCHEMA, stage="D2_AC_EXTENSION", baseline=BASELINE, variants=[VARIANT], options=OPTIONS,
        selected=selected, seed=SEED, task_horizon=400, full_episodes=True, search_coverage_enabled=True,
        planned_episode_runs=20, max_physical_steps=8000, independent_scenarios=20, workers=workers,
        collection_mode="single_V5_arm_in_independent_scene_processes", user_authorized_development=True,
        training=False, checkpoint_loaded=False, formal_thesis_evaluation=False, performance_passed=None,
        heldout_scenarios_executed=0, input_sha256=hashes, child_input_sha256=child_inputs,
        ac_plan_sha256=hashes[str(AC_PLAN)], reference=reference,
        source_changed_from_reference=sources["inventory"]["sha256"] != reference["source_inventory_sha256"],
        sources_before=sources, python=sys.version, trace_sha256={})
    output_directory(output, inputs)
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir()
    write_json(output / "identity.json", identity)
    pending, completed, failures, trace_hashes = deque(selected), [], [], {}
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            running = {}
            while pending or running:
                while pending and len(running) < workers and not failures:
                    verify()
                    scene = pending.popleft()
                    index = scene["original_episode_index"]
                    future = executor.submit(_run_child, child_command(manifest_path, config_path, output / f"scene_{index:04d}", index),
                                             output / "logs" / f"scene_{index:04d}.log")
                    running[future] = scene
                    print(f"SAFE_SEARCH_AC_START index={index} variant=V5", flush=True)
                if not running:
                    break
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    scene = running.pop(future)
                    index = scene["original_episode_index"]
                    try:
                        process = future.result()
                        if process["returncode"] != 0:
                            raise RuntimeError(f"AC child exited with status {process['returncode']}")
                        verify()
                        _, trace_hash = validate_child_ac(output / f"scene_{index:04d}", scene, identity)
                        trace_hashes[str(index)] = trace_hash
                        completed.append(index)
                        print(f"SAFE_SEARCH_AC_COMPLETE index={index} scenes={len(completed)}/20", flush=True)
                    except Exception as exc:
                        failures.append(dict(original_episode_index=index, exception_type=type(exc).__name__, message=str(exc)))
                        print(f"SAFE_SEARCH_AC_FAILURE index={index} {exc}", flush=True)
                write_json(output / "progress.json", dict(completed_original_indices=sorted(completed),
                    running_original_indices=sorted(r["original_episode_index"] for r in running.values()),
                    unstarted_original_indices=[r["original_episode_index"] for r in pending],
                    program_failures=failures, planned_episode_runs=20, experiment_complete=False))
        if failures:
            raise RuntimeError("AC development has program failures; no complete-stage verdict")
        verify()
        if read_reference(reference_dir)[2] != reference:
            raise RuntimeError("historical D2 reference changed during AC evaluation")
        identity["trace_sha256"] = trace_hashes
        write_json(output / "identity.json", identity)
        _, rows = collect_completed_ac(output, require_root_complete=False)
        final = summarize(rows, [VARIANT], 20, True)
        final.pop("paired_vs_v0", None)
        final.update(schema="ch3.safe_search.ac_development_summary.v1", stage="D2_AC_EXTENSION", planned_episode_runs=20,
            experiment_complete=True, source_and_input_verification_passed=True, program_failures=[],
            reference=reference, source_changed_from_reference=identity["source_changed_from_reference"],
            formal_thesis_evaluation=False, wall_seconds=time.perf_counter() - started)
        identity.update(sources_after=framework_sources(), source_and_input_verification_passed=True, experiment_complete=True)
        write_json(output / "episodes.json", rows)
        write_json(output / "identity.json", identity)
        write_json(output / "summary.json", final)
        write_json(output / "progress.json", dict(completed_original_indices=list(INDICES), running_original_indices=[],
            unstarted_original_indices=[], program_failures=[], planned_episode_runs=20, experiment_complete=True))
        print("SAFE_SEARCH_AC_FINISHED scenes=20 arms=20 variant=V5", flush=True)
        return final
    except BaseException as exc:
        write_json(output / "failure.json", dict(exception_type=type(exc).__name__, message=str(exc),
            completed_original_indices=sorted(completed), program_failures=failures,
            unstarted_original_indices=[scene["original_episode_index"] for scene in pending],
            planned_episode_runs=20, experiment_complete=False))
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", required=True, type=Path)
    result.add_argument("--config", type=Path, default=CONFIG)
    result.add_argument("--reference-dir", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--execute", required=True, action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    run_ac_development(args.manifest, args.output_dir, reference_dir=args.reference_dir,
                       config_path=args.config, workers=args.workers)


if __name__ == "__main__":
    main()
