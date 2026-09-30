"""B1 transfer of frozen V0/C/A+C on the same 20 development scenarios.

Each scene/variant is an independent zero-residual subprocess. All 60 actual
task terminals are required; this entry point never trains or loads weights.
The B0 reference runs retain their original source identities and files.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
import sys
import time

from .run_ac_development import collect_completed_ac, read_reference, validate_trace
from .run_development import CONFIG, INDICES, PLAN, SEED, _run_child, validate_development_inputs
from .run_paired import ROOT, file_hash, output_directory, read_json, summarize, write_json

BASELINE = "B1_bser_prior"
VARIANTS = ("V0", "V3", "V5")
B1_PLAN = ROOT / "docs/chapter3/search_diagnostics/safe_search_v1/b1_experiment_plan.json"
SCHEMA = "ch3.safe_search.b1_development_identity.v1"
OPTIONS = {
    "V0": dict(planning_state_consistency=False, failure_policy=False, path_safety=False),
    "V3": dict(planning_state_consistency=False, failure_policy=False, path_safety=True),
    "V5": dict(planning_state_consistency=True, failure_policy=False, path_safety=True),
}
ZERO_FIELDS = ("actor_forward_calls", "action_sampling_calls", "optimizer_update_count",
               "residual_action_max_abs", "physical_residual_acceleration_max_abs")


def validate_plan(plan):
    required = dict(schema="ch3.safe_search.b1_plan.v1", baseline=BASELINE,
        variants=list(VARIANTS), options=OPTIONS, development_indices=list(INDICES),
        seed=SEED, max_steps=400, planned_episode_runs=60, training=False,
        checkpoint_loaded=False, heldout_scenarios_executed=0,
        primary_comparison=["V5", "V3"], secondary_comparisons=[["V5", "V0"], ["V3", "V0"]],
        bootstrap_replicates=10000, bootstrap_seed=20260928)
    if any(plan.get(k) != v for k, v in required.items()):
        raise ValueError("B1 plan differs from the frozen transfer design")
    if any(type(x) is not bool for options in plan["options"].values() for x in options.values()):
        raise ValueError("B1 options must be boolean")
    if any(type(x) is not int for x in plan["development_indices"]):
        raise ValueError("B1 indices must be integers")
    if plan.get("development_gate") != dict(comparator="V3", found_count_not_lower=True,
            pre_found_collision_count_not_higher=True, stall_plus_hold_fraction_strictly_lower=True,
            effective_observation_fraction_not_lower=True):
        raise ValueError("B1 transfer gate differs from the preregistered comparison")


def reference_metadata(d2_dir, ac_dir):
    d2_dir, ac_dir = Path(d2_dir).resolve(), Path(ac_dir).resolve()
    d2, old_rows, _ = read_reference(d2_dir)
    ac, new_rows = collect_completed_ac(ac_dir)
    if d2["selected"] != ac["selected"] or len(old_rows) != 100 or len(new_rows) != 20:
        raise ValueError("B0 references do not cover the same completed development scenes")
    return {label: dict(directory=str(directory),
                identity_sha256=file_hash(directory / "identity.json"),
                episodes_sha256=file_hash(directory / "episodes.json"),
                source_inventory_sha256=identity["sources_before"]["inventory"]["sha256"],
                selected=identity["selected"], child_input_sha256=identity["child_input_sha256"])
            for label, directory, identity in (("B0_D2", d2_dir, d2), ("B0_AC", ac_dir, ac))}


def validate_references(plan, references, selected, child_inputs):
    if set(references) != {"B0_D2", "B0_AC"} or set(plan.get("reference_hashes", {})) != set(references):
        raise ValueError("both distinct B0 historical references are required")
    for label, ref in references.items():
        pins = {k: ref[k] for k in ("identity_sha256", "episodes_sha256", "source_inventory_sha256")}
        if (plan["reference_hashes"][label] != pins or ref["selected"] != selected
                or sorted(ref["child_input_sha256"].values()) != sorted(child_inputs.values())):
            raise ValueError("B1 input/population or historical B0 identity changed")


def child_command(manifest, config, output, index, variant):
    if index not in INDICES or variant not in VARIANTS:
        raise ValueError("unplanned B1 scene/variant")
    return [sys.executable, "-B", "-m", "chapter3_bser.experiments.safe_search_v1.run_paired",
        "--manifest", str(manifest), "--config", str(config), "--output-dir", str(output),
        "--baseline", BASELINE, "--variants", variant, "--episode-indices", str(index),
        "--steps", "400", "--seed", str(SEED), "--full-episodes", "--search-coverage"]


def arm_directory(output, index, variant):
    return Path(output) / f"scene_{index:04d}_{variant}"


def validate_terminal(row, scene, variant):
    expected = {k: scene[k] for k in ("original_episode_index", "scenario_id", "scenario_seed", "environment_innovation_seed")}
    expected.update(baseline=BASELINE, variant=variant, terminal=True,
                    full_episode_requested=True, full_episode_completed=True)
    if any(row.get(k) != v for k, v in expected.items()):
        raise ValueError("B1 terminal identity/complete flag mismatch")
    if not all(row.get(k) is True for k in ("terminal", "full_episode_requested", "full_episode_completed")):
        raise ValueError("B1 requires actual complete task terminals")
    steps, found = row.get("physical_steps"), row.get("found_step")
    if (type(steps) is not int or not 1 <= steps <= 400
            or found is not None and (type(found) is not int or not 0 <= found <= steps)
            or row.get("found_within_budget") is not (found is not None)):
        raise ValueError("invalid B1 physical/Found steps")
    result, reason = row.get("episode_result", {}), row.get("stop_reason")
    collision = result.get("first_collision_step")
    if (reason not in ("success", "timeout", "obstacle_collision")
            or reason != result.get("termination_reason") or result.get("terminal_step") != steps
            or result.get("task_protocol") != "collision_terminal_v1"
            or reason == "timeout" and steps != 400
            or result.get("success") is not (reason == "success")
            or reason == "success" and found is None
            or (reason == "obstacle_collision") != (collision is not None)
            or collision is not None and (type(collision) is not int or collision != steps)):
        raise ValueError("inconsistent B1 task outcome")
    exposure = found if found is not None else steps
    pre_collision = collision is not None and (found is None or collision <= found)
    if row.get("pre_found_exposure_steps") != exposure or row.get("pre_found_collision") is not pre_collision:
        raise ValueError("inconsistent B1 pre-Found metrics")
    names = ("searcher_motion_stall_proxy_agent_steps", "searcher_hold_agent_steps")
    if (any(type(row.get(k)) is not int or row[k] < 0 for k in names)
            or sum(row[k] for k in names) > 3 * exposure):
        raise ValueError("invalid disjoint B1 motion counts")
    complete = row.get("complete_episode_row") or {}
    fields = dict(method="ch3_baseline_bser_prior", reference_runtime_method="ch3_baseline_bser_prior",
        found=found is not None, found_step=found, success=reason == "success", actual_length=steps,
        episode_length=steps, terminal_step=steps, termination_reason=reason,
        scenario_id=scene["scenario_id"], scenario_seed=scene["scenario_seed"],
        environment_innovation_seed=scene["environment_innovation_seed"],
        evaluation_episode_index=scene["original_episode_index"], optimizer_update_count=0,
        training_update=False, first_collision_step=collision)
    if any(k not in complete or complete[k] != v for k, v in fields.items()):
        raise ValueError("B1 complete episode row differs from its terminal record")
    controller = row.get("controller", {})
    if any(k not in controller or controller[k] != 0 for k in ZERO_FIELDS):
        raise ValueError("B1 must have zero learned actions and zero physical residuals")
    if variant != "V0" and controller.get("safe_search", {}).get("options") != OPTIONS[variant]:
        raise ValueError("B1 runtime did not execute the requested repair options")
    coverage = row.get("search_coverage") or {}
    if coverage.get("available") is not True or coverage.get("pre_found_exposure_steps") != exposure:
        raise ValueError("B1 complete measured coverage is required")


def validate_child(directory, scene, variant, parent):
    directory = Path(directory)
    if (directory / "failure.json").exists():
        raise ValueError("B1 child contains a program failure")
    identity, summary, rows = (read_json(directory / n) for n in ("identity.json", "summary.json", "episodes.json"))
    expected = dict(schema="ch3.safe_search.paired_identity.v1", baseline=BASELINE, variants=[variant],
        seed=SEED, steps=400, task_horizon=400, full_episodes=True, training=False, checkpoint_loaded=False,
        formal_thesis_evaluation=False, residual_source="zeros_4x3", selected=[scene], search_coverage_enabled=True,
        input_sha256=parent["child_input_sha256"], sources_before=parent["sources_before"],
        sources_after=parent["sources_before"], source_and_input_verification_passed=True)
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError("B1 child source/input/request mismatch")
    if (len(rows) != 1 or summary.get("all_requested_runs_recorded") is not True
            or summary.get("source_and_input_verification_passed") is not True
            or set(summary.get("variants", {})) != {variant}):
        raise ValueError("B1 child must contain exactly one verified arm")
    row = rows[0]
    validate_terminal(row, scene, variant)
    arm = directory / f"episode_{scene['original_episode_index']:04d}" / variant
    if row != read_json(arm / "summary.json"):
        raise ValueError("B1 row differs from arm artifact")
    metrics = summary["variants"][variant]
    if any(metrics.get(k) != v for k, v in dict(n_expected=1, n_recorded=1, n_terminal=1,
                                                full_episode_evaluation_complete=True).items()):
        raise ValueError("B1 arm summary is incomplete")
    return row, validate_trace(arm / "step_trace.jsonl", row)


def collect_completed_b1(output, *, require_root_complete=True):
    output = Path(output)
    parent = read_json(output / "identity.json")
    if (output / "failure.json").exists():
        raise ValueError("B1 run has a recorded program failure")
    expected = dict(schema=SCHEMA, baseline=BASELINE, variants=list(VARIANTS), options=OPTIONS,
        seed=SEED, task_horizon=400, full_episodes=True, search_coverage_enabled=True,
        planned_episode_runs=60, max_physical_steps=24000, independent_scenarios=20,
        training=False, checkpoint_loaded=False, formal_thesis_evaluation=False, heldout_scenarios_executed=0)
    if any(parent.get(k) != v for k, v in expected.items()):
        raise ValueError("invalid B1 root identity")
    if require_root_complete and (parent.get("experiment_complete") is not True
            or parent.get("source_and_input_verification_passed") is not True
            or parent.get("sources_before") != parent.get("sources_after")):
        raise ValueError("B1 root is not complete/source-verified")
    plan = read_json(B1_PLAN)
    validate_plan(plan)
    if parent.get("b1_plan_sha256") != file_hash(B1_PLAN):
        raise ValueError("B1 plan changed")
    frozen = read_json(PLAN)["splits"]["development"]
    selected = [dict(original_episode_index=r["source_episode_index"], scenario_id=r["scenario_id"],
        scenario_seed=r["scenario_seed"], environment_innovation_seed=r["environment_innovation_seed"],
        scenario_sha256=r["scene_content_sha256"]) for r in frozen]
    if parent.get("selected") != selected or tuple(s["original_episode_index"] for s in selected) != INDICES:
        raise ValueError("B1 must preserve all frozen scene identities")
    validate_references(plan, parent["references"], selected, parent["child_input_sha256"])
    rows, traces = [], {}
    for scene in selected:
        index = scene["original_episode_index"]
        for variant in VARIANTS:
            row, sha = validate_child(arm_directory(output, index, variant), scene, variant, parent)
            rows.append(row)
            traces[f"{index}:{variant}"] = sha
    if parent.get("trace_sha256") != traces:
        raise ValueError("B1 completed trace fingerprint mismatch")
    if require_root_complete and rows != read_json(output / "episodes.json"):
        raise ValueError("B1 root rows do not equal all 60 complete children")
    return parent, rows


def run_b1_development(manifest, output, *, b0_reference_dir, b0_ac_dir, config=CONFIG, workers=6):
    if type(workers) is not int or not 1 <= workers <= 20:
        raise ValueError("workers must be an integer in 1..20")
    manifest, config = Path(manifest).resolve(), Path(config).resolve()
    b0_reference_dir, b0_ac_dir = Path(b0_reference_dir).resolve(), Path(b0_ac_dir).resolve()
    inputs = (manifest, config, PLAN, B1_PLAN, *(d/n for d in (b0_reference_dir, b0_ac_dir)
                                               for n in ("identity.json", "episodes.json")))
    output = output_directory(output, inputs)
    if any(output == d or d in output.parents for d in (b0_reference_dir, b0_ac_dir)):
        raise ValueError("B1 output must be outside both retained B0 reference directories")
    selected = validate_development_inputs(read_json(manifest), read_json(config), read_json(PLAN))
    plan = read_json(B1_PLAN)
    validate_plan(plan)
    refs = reference_metadata(b0_reference_dir, b0_ac_dir)
    hashes = {str(p): file_hash(p) for p in inputs}
    child_inputs = {str(p): hashes[str(p)] for p in (manifest, config)}
    validate_references(plan, refs, selected, child_inputs)
    from .provenance import framework_sources, verify_sources
    sources = framework_sources()

    def verify():
        verify_sources(sources)
        if any(file_hash(Path(p)) != sha for p, sha in hashes.items()):
            raise RuntimeError("B1 source/input/reference changed during execution")

    parent = dict(schema=SCHEMA, stage="B1_TRANSFER_DEVELOPMENT", baseline=BASELINE, variants=list(VARIANTS),
        options=OPTIONS, selected=selected, seed=SEED, task_horizon=400, full_episodes=True,
        search_coverage_enabled=True, planned_episode_runs=60, max_physical_steps=24000,
        independent_scenarios=20, workers=workers, collection_mode="independent_scene_variant_processes",
        training=False, checkpoint_loaded=False, formal_thesis_evaluation=False, performance_passed=None,
        heldout_scenarios_executed=0, user_authorized_development=True, input_sha256=hashes,
        child_input_sha256=child_inputs, b1_plan_sha256=hashes[str(B1_PLAN)], references=refs,
        sources_before=sources, python=sys.version, trace_sha256={})
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir()
    write_json(output / "identity.json", parent)
    pending = deque((scene, v) for scene in selected for v in VARIANTS)
    completed, failures, trace_hashes = [], [], {}
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            running = {}
            while pending or running:
                while pending and len(running) < workers and not failures:
                    verify()
                    scene, variant = pending.popleft()
                    index = scene["original_episode_index"]
                    cmd = child_command(manifest, config, arm_directory(output, index, variant), index, variant)
                    future = executor.submit(_run_child, cmd, output / "logs" / f"scene_{index:04d}_{variant}.log")
                    running[future] = (scene, variant)
                    print(f"SAFE_SEARCH_B1_START index={index} variant={variant}", flush=True)
                if not running:
                    break
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    scene, variant = running.pop(future)
                    index = scene["original_episode_index"]
                    try:
                        process = future.result()
                        if process["returncode"]:
                            raise RuntimeError(f"child exit {process['returncode']}")
                        verify()
                        _, sha = validate_child(arm_directory(output, index, variant), scene, variant, parent)
                        trace_hashes[f"{index}:{variant}"] = sha
                        completed.append([index, variant])
                        print(f"SAFE_SEARCH_B1_COMPLETE index={index} variant={variant} arms={len(completed)}/60", flush=True)
                    except Exception as exc:
                        failures.append(dict(index=index, variant=variant, exception=type(exc).__name__, message=str(exc)))
                        print(f"SAFE_SEARCH_B1_FAILURE index={index} variant={variant} {exc}", flush=True)
                write_json(output / "progress.json", dict(completed_pairs=sorted(completed),
                    running_pairs=sorted([s["original_episode_index"], v] for s, v in running.values()),
                    unstarted_pairs=[[s["original_episode_index"], v] for s, v in pending],
                    program_failures=failures, planned_episode_runs=60, experiment_complete=False))
        if failures:
            raise RuntimeError("B1 program failure; no complete evaluation verdict")
        verify()
        if reference_metadata(b0_reference_dir, b0_ac_dir) != refs:
            raise RuntimeError("historical B0 references changed")
        parent["trace_sha256"] = trace_hashes
        write_json(output / "identity.json", parent)
        _, rows = collect_completed_b1(output, require_root_complete=False)
        final = summarize(rows, VARIANTS, 20, True)
        final.update(schema="ch3.safe_search.b1_development_summary.v1", experiment_complete=True,
            baseline=BASELINE, planned_episode_runs=60, program_failures=[],
            source_and_input_verification_passed=True, wall_seconds=time.perf_counter()-started,
            formal_thesis_evaluation=False)
        parent.update(sources_after=framework_sources(), source_and_input_verification_passed=True, experiment_complete=True)
        write_json(output / "episodes.json", rows)
        write_json(output / "summary.json", final)
        write_json(output / "identity.json", parent)
        write_json(output / "progress.json", dict(completed_pairs=sorted(completed), running_pairs=[], unstarted_pairs=[],
            program_failures=[], planned_episode_runs=60, experiment_complete=True))
        print("SAFE_SEARCH_B1_FINISHED scenes=20 arms=60", flush=True)
        return final
    except BaseException as exc:
        write_json(output / "failure.json", dict(exception=type(exc).__name__, message=str(exc),
            completed_pairs=sorted(completed), program_failures=failures, planned_episode_runs=60,
            unstarted_pairs=[[s["original_episode_index"], v] for s, v in pending], experiment_complete=False))
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--config", default=CONFIG, type=Path)
    p.add_argument("--b0-reference-dir", required=True, type=Path)
    p.add_argument("--b0-ac-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--execute", required=True, action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    run_b1_development(args.manifest, args.output_dir, config=args.config,
        b0_reference_dir=args.b0_reference_dir, b0_ac_dir=args.b0_ac_dir, workers=args.workers)


if __name__ == "__main__":
    main()
