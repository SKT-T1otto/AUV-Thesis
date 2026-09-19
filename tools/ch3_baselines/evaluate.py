"""CPU serial complete-task evaluator. No checkpoint, training, resume or scene generation."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time
import traceback

import torch

from chapter3_bser.experiments.hgr.train import validate_config, validated_output
from chapter3_bser.experiments.hgr.evaluation import resolved_config as resolve_reference, evaluation_manifest
from chapter3_bser.experiments.phase1c_prrac.task_metrics import strict_outcome, validated_rows, aggregate_task_outcomes
from chapter3_bser.experiments.reward_objective import objective_identity
from core.env.task_protocol import protocol_identity
from .basic_search_prior import BasicSearchPriorRuntime, METHOD, SPEC, baseline_environment_kwargs
from .provenance import ROOT, capture_sources, require_unchanged, checkout_identity, file_sha256, digest, write_json


def resolve_config(path, episodes, seed):
    path = Path(path).resolve()
    config = validate_config(json.loads(path.read_text(encoding="utf-8")))
    reference = resolve_reference(config, episodes, seed, "stochastic")
    required = dict(method="ch3_hgr", profile="M20_MOVING_UNKNOWN_MULTI", max_steps=400,
                    execution_runtime_revision="dynamic_public_intercept_v2_1")
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError("baseline v1 requires the HGR M20 / 400-step / v2.1 reference")
    if reference["gamma"] != 0.95 or reference["collision_terminal_reward"] != -2.0:
        raise ValueError("baseline v1 requires gamma=0.95 and collision reward=-2.0")
    if config.get("early_discovery", {}).get("enabled"):
        raise ValueError("BEDS is not a supported reference")
    if reference["phase1b_config"]["mechanism_version"] != "phase1b2_corrected":
        raise ValueError("baseline requires corrected public-handoff controller")
    if not reference["environment_config"].get("use_residual_prior"):
        raise ValueError("reference must enable residual prior")
    if config["reward"].get("executor_id") != 3 or config["reward"].get("searcher_ids") != [0, 1, 2]:
        raise ValueError("incompatible reward role order")
    spec = json.loads((ROOT / "configs/chapter3/baselines/basic_search_prior_v1.json").read_text(encoding="utf-8"))
    if {k: spec.get(k) for k in SPEC} != SPEC:
        raise ValueError("baseline specification and implementation disagree")
    # Use the authoritative HGR resolver only for task defaults, never its
    # policy identity or pairing hash. The complete reference is retained too.
    conditions = {k: copy.deepcopy(reference[k]) for k in (
        "base_candidate", "profile", "max_steps", "environment_config", "reward", "gamma",
        "task_protocol", "collision_detection_revision", "terminal_reward_revision",
        "collision_terminal_reward", "reward_objective", "source_reward_revision",
        "execution_runtime", "execution_runtime_revision", "phase1b_config", "phase1b_reference_config_sha256")}
    effective_environment = baseline_environment_kwargs(config)
    if {k for k in effective_environment.keys() | reference["environment_config"].keys()
        if effective_environment.get(k) != reference["environment_config"].get(k)} - {"pse_use_standby"}:
        raise ValueError("baseline environment drift outside declared standby intervention")
    # This flag is a method intervention, rather than a common task condition.
    conditions["environment_config"].pop("pse_use_standby", None)
    provenance_kind = "default_config_reference" if "outputs" not in path.parts else "run_config_reference"
    return config, dict(schema="ch3.basic_search_prior.evaluation.v1", **SPEC,
        episodes=episodes, seed=seed, device="cpu", collection_mode="serial", resume_supported=False,
        reference_training_method=config["method"], reference_kind=provenance_kind,
        reference_config_path=str(path), reference_config_sha256=file_sha256(path),
        reference_training_config=config, common_task_conditions=conditions,
        effective_environment_config=effective_environment,
        method_interventions=SPEC, spec=spec)


def select_manifest(config, episodes, seed, manifest, sources):
    if manifest is None:
        raise ValueError("an existing frozen --manifest is required; this evaluator never generates one")
    # Validate all identities, including the unselected tail, then retain the
    # original prefix and order. Reuse the production label/seed/profile checks.
    original = json.loads(Path(manifest).read_text(encoding="utf-8"))
    scenes = original.get("scenarios")
    if not isinstance(scenes, list) or len(scenes) < episodes:
        raise ValueError("manifest has insufficient scenarios")
    evaluation_manifest(config, len(scenes), seed, manifest, sources["production"])
    selected = evaluation_manifest(config, episodes, seed, manifest, sources["production"])
    selected["schema"] = "ch3.basic_search_prior.manifest.v1"
    return selected


def episode_row(runtime, config, scenario, index, seed, wall_seconds):
    row = runtime.env.finalize_episode()
    row.update(scenario_id=scenario["scenario_id"], scenario_seed=scenario["scenario_seed"],
               actual_length=runtime.step, episode_length=runtime.step,
               found_step=runtime.found_step, handoff_event_step=runtime.handoff_event_step,
               handoff_decision_step=runtime.handoff_decision_step,
               method=METHOD, evaluation_episode_index=index, environment_innovation_seed=seed,
               wall_seconds=wall_seconds, optimizer_update_count=0, training_update=False)
    for identity in (protocol_identity, objective_identity):
        if identity(row) != identity(config):
            raise ValueError("authoritative task/reward identity mismatch")
    if row.get("gamma") != config["rl"]["gamma"] or not 0 < runtime.step <= config["max_steps"]:
        raise ValueError("authoritative gamma/clock mismatch")
    if runtime.env.reward_accounting.steps != runtime.step or not runtime.terminal:
        raise ValueError("incomplete episode or reward accounting clock mismatch")
    for key in ("first_collision_step", "first_collision_agent_ids", "first_collision_phase", "source_individual_returns"):
        if key not in row:
            raise ValueError(f"missing authoritative field: {key}")
    for flag, key in (("contact_episode", "capture_contact_step_count"), ("hold_episode", "capture_full_hold_step_count")):
        count = row.get(key)
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError(f"invalid authoritative counter: {key}")
        row[flag] = None if count is None else count > 0
    for key in ("team_discounted_return", "team_undiscounted_return"):
        if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]):
            raise ValueError(f"missing/nonfinite {key}")
    if strict_outcome(row) == "INCOMPLETE":
        raise ValueError("collector returned an incomplete outcome")
    row = validated_rows([row], require_complete=True)[0]
    json.dumps(row, allow_nan=False)
    return row


def summary(rows, episodes, *, finalized, wall_seconds, actual_steps):
    result = aggregate_task_outcomes(validated_rows(rows, require_complete=True), expected_episodes=episodes)
    complete = bool(result["evaluation_complete"] and finalized)
    if not complete:
        for key in result:
            if key.endswith("_rate"):
                result[key] = None
    result.update(schema="ch3.basic_search_prior.summary.v1", method=METHOD,
                  evaluation_complete=complete, training_update=False, wall_seconds=wall_seconds,
                  actual_environment_steps=actual_steps, baseline_performance_verified=False)
    for key in ("team_discounted_return", "team_undiscounted_return"):
        result["mean_" + key] = sum(r[key] for r in rows) / len(rows) if complete else None
    return result


def evaluate(reference_training_config, manifest, output_dir, *, episodes=100, seed=12729):
    config, resolved = resolve_config(reference_training_config, episodes, seed)
    output = validated_output(config, output_dir)
    sources = capture_sources()
    selected = select_manifest(config, episodes, seed, manifest, sources)
    identity = dict(schema="ch3.basic_search_prior.identity.v1", method=METHOD,
        reference_config_path=resolved["reference_config_path"], reference_config_sha256=resolved["reference_config_sha256"],
        reference_kind=resolved["reference_kind"], reference_training_method=config["method"],
        common_task_conditions=resolved["common_task_conditions"], method_interventions=SPEC,
        effective_environment_config=resolved["effective_environment_config"],
        manifest_file_sha256=selected["origin"]["original_file_sha256"], selected_content_sha256=selected["selected_content_sha256"],
        random_streams=dict(scheme="HGR seed_innovations(seed + zero_based_episode_index), before creation and after reset; scenario/planner seed retained; no policy RNG draws",
            episodes=[dict(scenario_id=s["scenario_id"], scenario_seed=s["scenario_seed"], environment_innovation_seed=seed+i) for i, s in enumerate(selected["scenarios"])],
            pairing_scope="No claim of identical innovation events across different method trajectories"),
        sources_before=sources, checkout_before=checkout_identity())
    identity["sha256"] = digest(identity)
    documents = {"resolved_evaluation_config.json": resolved, "baseline_spec.json": resolved["spec"],
                 "evaluation_manifest.json": selected}
    frozen_inputs = digest(dict(config=config, resolved=resolved, selected=selected))

    def verify():
        require_unchanged(sources, capture_sources())
        if file_sha256(reference_training_config) != resolved["reference_config_sha256"]:
            raise ValueError("reference config changed during evaluation")
        if file_sha256(manifest) != selected["origin"]["original_file_sha256"]:
            raise ValueError("manifest changed during evaluation")
        if frozen_inputs != digest(dict(config=config, resolved=resolved, selected=selected)):
            raise ValueError("in-memory evaluation inputs changed")

    verify()
    validated_output(config, output)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in documents.items():
        write_json(output/name, value)
    write_json(output/"evaluation_identity.json", identity)
    hashes = {name: file_sha256(output/name) for name in (*documents, "evaluation_identity.json")}
    rows, diagnostics = [], []
    started, actual_steps = time.perf_counter(), 0
    runtime, index, scenario = None, None, None

    def save(status, finalized=False):
        result = summary(rows, episodes, finalized=finalized, wall_seconds=time.perf_counter()-started, actual_steps=actual_steps)
        write_json(output/"episodes.json", rows)
        write_json(output/"summary.json", result)
        write_json(output/"controller_diagnostics.json", dict(episodes=diagnostics,
            missing_information="Unavailable contact/hold counters remain null; unreachable search reasons are only exposed as a count by the shared generator."))
        write_json(output/"evaluation_progress.json", dict(status=status, n_expected_episodes=episodes,
            n_valid_episodes=len(rows), evaluation_complete=result["evaluation_complete"],
            current_scenario_id=None if scenario is None else scenario["scenario_id"], actual_environment_steps=actual_steps))
        return result

    def record_end_sources():
        # On failure record actual bytes too, even if the frozen source gate fails.
        from chapter3_bser.experiments.hgr.provenance import fresh_source_identity
        from .provenance import baseline_source_identity
        identity["sources_after"] = dict(production=fresh_source_identity(), baseline=baseline_source_identity())
        identity["checkout_after"] = checkout_identity()
        identity["end_state_sha256"] = digest(identity["sources_after"])
        write_json(output/"evaluation_identity.json", identity)

    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with (output/"run_console.log").open("x", encoding="utf-8") as log:
            save("running")
            for index, scenario in enumerate(selected["scenarios"]):
                verify()
                episode_started = time.perf_counter()
                runtime = BasicSearchPriorRuntime(config, scenario, seed=seed+index, episode_id=index)
                while not runtime.terminal:
                    before = runtime.step
                    try:
                        runtime.advance()
                    finally:
                        actual_steps += runtime.step - before
                row = episode_row(runtime, config, scenario, index, seed+index, time.perf_counter()-episode_started)
                diag = dict(runtime.controller_diagnostics(), scenario_id=scenario["scenario_id"])
                for key in ("found_step", "handoff_event_step", "handoff_decision_step", "executor_target_received_step", "capture_contact_step_count", "capture_full_hold_step_count", "first_collision_step", "first_collision_agent_ids", "first_collision_phase"):
                    diag[key] = row.get(key)
                ids = row["first_collision_agent_ids"]
                diag["first_collision_roles"] = ["Executor" if i == 3 else "Searcher" for i in ids]
                runtime.close()
                runtime = None
                verify()
                rows.append(row)
                diagnostics.append(diag)
                save("running")
                line = (f"[{index+1}/{episodes}] scenario={scenario['scenario_id']} outcome={row['failure_stage']} "
                        f"found={row['found_step']} handoff={row['handoff_event_step']} steps={row['episode_length']} "
                        f"wall={row['wall_seconds']:.2f}s elapsed={time.perf_counter()-started:.2f}s")
                print(line, flush=True)
                print(line, file=log, flush=True)
            verify()
            if any(file_sha256(output/name) != sha for name, sha in hashes.items()):
                raise ValueError("saved evaluation inputs changed during evaluation")
            record_end_sources()
            return save("complete", finalized=True)
    except (Exception, KeyboardInterrupt) as exc:
        write_json(output/"evaluation_failure.json", dict(evaluation_complete=False,
            evaluation_episode_index=index, scenario=scenario, exception_type=type(exc).__name__,
            message=str(exc), traceback=traceback.format_exc(),
            partial_controller_diagnostics=None if runtime is None else runtime.controller_diagnostics()))
        with (output/"run_console.log").open("a", encoding="utf-8") as log:
            print(f"FAILED scenario={None if scenario is None else scenario['scenario_id']}: {exc}", file=log, flush=True)
        record_end_sources()
        save("failed")
        raise
    finally:
        if runtime is not None:
            runtime.close()
        torch.set_num_threads(old_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-training-config", required=True, help="Actual HGR run config.json preferred; explicit checked-in config is marked default_config_reference")
    parser.add_argument("--manifest", required=True, help="Existing frozen validation manifest (never generated)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12729)
    parser.add_argument("--output-dir", required=True, help="New/empty path containing collision_terminal; no resume")
    args = parser.parse_args(argv)
    return evaluate(**vars(args))


if __name__ == "__main__":
    main()
