"""Full-horizon observer noninterference check for the failed D2 scene 10/V3.

Run as a module from the repository root. This auxiliary script does not alter
the reviewed source gate. Both new runs use exactly the same safe-search V3;
the old failed run contributes only its persisted public 280-step prefix.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import traceback

from chapter3_bser.experiments.safe_search_v1.run_paired import (
    ROOT, digest, execute_episode, file_hash, load_dependencies, output_directory,
    read_json, write_json,
)
from chapter3_bser.experiments.safe_search_v1.run_development import (
    PLAN, validate_development_inputs,
)

INDEX = 10
SEED = 12739
BASELINE = "B0_search_prior"
VARIANT = "V3"
OLD_PREFIX = (ROOT / "runs/safe_search_v1/development_B0_20260928_v1/scene_0010"
              / "episode_0010/V3/step_trace.jsonl")
AGENT_FIELDS = (
    "agent_id", "role", "position", "velocity", "assignment_id", "assignment_kind",
    "semantic_waypoint", "tracking_waypoint", "hold", "reachable", "planned_path",
)
SNAPSHOT_FIELDS = (
    "step", "found", "executor_knows_target", "allocation_hash", "decision_reason",
    "cached_map_revision", "live_map_revision", "last_full_refresh_step",
    "full_refresh", "replan_count", "replan_steps",
)


def read_trace(path):
    def reject(value):
        raise ValueError("nonfinite trace value: " + value)
    with Path(path).open(encoding="utf-8") as handle:
        result = [json.loads(line, parse_constant=reject) for line in handle]
    for step, row in enumerate(result, 1):
        if row["step_before"] != step - 1 or row["step_after"] != step:
            raise ValueError("trace is not a contiguous physical prefix")
    return result


def common_public_row(row):
    """Explicit whitelist excludes changed path-index/geometry telemetry."""
    result = {key: row[key] for key in ("step_before", "step_after", "search_transition", "collision_records")}
    for side in ("before", "after"):
        snapshot = row[side]
        common = {key: snapshot[key] for key in SNAPSHOT_FIELDS}
        common["agents"] = [{key: agent[key] for key in AGENT_FIELDS}
                            for agent in sorted(snapshot["agents"], key=lambda a: a["agent_id"])]
        result[side] = common
    return result


def compare_old_prefix(old_path, new_path):
    old, new = read_trace(old_path), read_trace(new_path)
    if len(old) != 280:
        raise ValueError("expected the known failed V3 persisted 280-step prefix")
    if len(new) < len(old):
        raise ValueError("new observed trajectory ended before the old persisted prefix")
    first_mismatch = next((step for step, (left, right) in enumerate(zip(old, new), 1)
                           if common_public_row(left) != common_public_row(right)), None)
    return dict(compared_steps=len(old), equal=first_mismatch is None,
        first_mismatching_step=first_mismatch,
        old_public_prefix_sha256=digest([common_public_row(row) for row in old]),
        new_public_prefix_sha256=digest([common_public_row(row) for row in new[:len(old)]]),
        agent_fields=list(AGENT_FIELDS), snapshot_fields=list(SNAPSHOT_FIELDS),
        includes_old_reward_or_rng=False,
        limitation="The failed old arm has no complete reward/RNG signature sequence; this is public prefix equality only.")


def check_terminal(signatures, row):
    if (row.get("terminal") is not True or row.get("full_episode_completed") is not True
            or row.get("full_episode_requested") is not True
            or row.get("complete_episode_row") is None
            or row.get("stop_reason") not in ("success", "timeout", "obstacle_collision")
            or type(row.get("physical_steps")) is not int or not 1 <= row["physical_steps"] <= 400
            or len(signatures) != row["physical_steps"] + 1):
        raise ValueError("equivalence run did not reach a valid full task terminal")
    if row["stop_reason"] == "timeout" and row["physical_steps"] != 400:
        raise ValueError("a prefix cannot be called a task timeout")
    if row["episode_result"].get("task_protocol") != "collision_terminal_v1":
        raise ValueError("task protocol changed")


def run(manifest_path, config_path, output_dir, old_prefix_path=OLD_PREFIX):
    manifest_path, config_path, old_prefix_path = (Path(p).resolve()
        for p in (manifest_path, config_path, old_prefix_path))
    inputs = (manifest_path, config_path, PLAN, old_prefix_path, Path(__file__).resolve())
    output = output_directory(output_dir, inputs)
    if output.exists():
        raise FileExistsError("equivalence check requires a new directory")
    output.mkdir(parents=True)
    stage = "validate_inputs"
    completed = []
    torch = None
    previous_threads = None
    try:
        raw_config, manifest, plan = (read_json(p) for p in (config_path, manifest_path, PLAN))
        selected = validate_development_inputs(manifest, raw_config, plan)
        scenario_record = next(item for item in selected if item["original_episode_index"] == INDEX)
        scenario = copy.deepcopy(manifest["scenarios"][INDEX])
        if scenario_record["environment_innovation_seed"] != SEED:
            raise ValueError("original scenario innovation seed changed")
        hashes = {str(p): file_hash(p) for p in inputs}
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        from chapter3_bser.experiments.hgr.train import validate_config
        from chapter3_bser.experiments.safe_search_v1.runtime import make_runtime
        from chapter3_bser.experiments.safe_search_v1.provenance import framework_sources, verify_sources
        import torch as torch_module
        torch = torch_module
        config = validate_config(raw_config)
        sources = framework_sources()
        immutable = digest(dict(config=config, scenario=scenario))

        def verify():
            verify_sources(sources)
            if any(file_hash(path) != value for path, value in hashes.items()):
                raise RuntimeError("equivalence source/input/script changed during execution")
            if digest(dict(config=config, scenario=scenario)) != immutable:
                raise RuntimeError("in-memory config/scenario changed during execution")

        identity = dict(schema="ch3.safe_search.telemetry_equivalence_identity.v1",
            scenario=scenario_record, original_episode_index=INDEX, variant=VARIANT,
            baseline=BASELINE, environment_innovation_seed=SEED, task_horizon=400,
            full_episodes=True, training=False, checkpoint_loaded=False,
            formal_thesis_evaluation=False, sources_before=sources, input_sha256=hashes,
            script_sha256=hashes[str(Path(__file__).resolve())],
            comparison="same V3 with no observers versus QueryTap, TransitionObserver and CoverageObserver",
            experiment_complete=False)
        write_json(output / "identity.json", identity)
        write_json(output / "resolved_config.json", config)
        deps = load_dependencies(config, scenario, INDEX, SEED, BASELINE, search_coverage=True)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        outputs = {}
        for observed in (False, True):
            name = "observed" if observed else "unobserved"
            stage = name
            arm = output / name
            arm.mkdir()
            verify()
            signatures, row = execute_episode(
                lambda: make_runtime(copy.deepcopy(config), copy.deepcopy(scenario),
                    baseline=BASELINE, variant=VARIANT, seed=SEED, episode_id=INDEX),
                arm, steps=400, observed=observed, deps=deps, full_episodes=True)
            verify()
            write_json(arm / "signatures.json", signatures)
            row.update(baseline=BASELINE, variant=VARIANT, original_episode_index=INDEX,
                scenario_id=scenario["scenario_id"], scenario_seed=scenario["scenario_seed"],
                environment_innovation_seed=SEED, signature_sha256=digest(signatures),
                observation_enabled=observed)
            write_json(arm / "summary.json", row)
            check_terminal(signatures, row)
            outputs[name] = (signatures, row)
            completed.append(name)
        stage = "compare"
        plain, observed = outputs["unobserved"], outputs["observed"]
        equal_signatures = plain[0] == observed[0]
        first_mismatch = next((i for i, (a, b) in enumerate(zip(plain[0], observed[0])) if a != b), None)
        if first_mismatch is None and len(plain[0]) != len(observed[0]):
            first_mismatch = min(len(plain[0]), len(observed[0]))
        same_terminal = all(plain[1][name] == observed[1][name] for name in
                            ("physical_steps", "found_step", "stop_reason", "episode_result"))
        prefix = compare_old_prefix(old_prefix_path, output / "observed/step_trace.jsonl")
        result = dict(schema="ch3.safe_search.telemetry_equivalence.v1",
            full_task_terminal_both=True, task_horizon=400,
            physical_steps=observed[1]["physical_steps"], found_step=observed[1]["found_step"],
            termination_reason=observed[1]["stop_reason"], same_terminal_outcome=same_terminal,
            equal_at_every_signature=equal_signatures, first_mismatching_signature_index=first_mismatch,
            signature_count_unobserved=len(plain[0]), signature_count_observed=len(observed[0]),
            unobserved_signature_sha256=digest(plain[0]), observed_signature_sha256=digest(observed[0]),
            signature_covers="public physical state, observations, reward record, guidance and Python/NumPy/Torch RNG; initial state plus every physical step",
            old_failed_run_public_prefix=prefix, training=False, checkpoint_loaded=False,
            formal_thesis_evaluation=False)
        write_json(output / "comparison.json", result)
        if not equal_signatures or not same_terminal or not prefix["equal"]:
            raise RuntimeError("telemetry equivalence or old public prefix comparison failed")
        verify()
        identity.update(sources_after=framework_sources(), source_and_input_verification_passed=True,
                        experiment_complete=True)
        result.update(source_and_input_verification_passed=True, experiment_complete=True)
        write_json(output / "identity.json", identity)
        write_json(output / "summary.json", result)
        print(json.dumps(result, ensure_ascii=True), flush=True)
        return result
    except BaseException as exc:
        write_json(output / "failure.json", dict(stage=stage, exception_type=type(exc).__name__,
            message=str(exc), completed_arms=completed, experiment_complete=False,
            traceback=traceback.format_exc()))
        raise
    finally:
        if torch is not None and previous_threads is not None:
            torch.set_num_threads(previous_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/chapter3/hgr_train.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--old-prefix-trace", type=Path, default=OLD_PREFIX)
    args = parser.parse_args(argv)
    run(args.manifest, args.config, args.output_dir, args.old_prefix_trace)


if __name__ == "__main__":
    main()
