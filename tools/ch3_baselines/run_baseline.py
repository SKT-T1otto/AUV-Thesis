"""Unified B0--B3 evaluation. Training is never dispatched from this entry."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
import traceback

import torch

from chapter3_bser.experiments.hgr import evaluation as native_evaluation
from chapter3_bser.experiments.baselines.common.checkpoint import load_checkpoint, validated_output
from chapter3_bser.experiments.hgr.provenance import checkout_identity
from chapter3_bser.experiments.phase1c_prrac.task_metrics import validated_rows
from . import evaluate as prior_evaluation
from . import learned_evaluation
from .basic_search_prior import BasicSearchPriorRuntime, baseline_environment_kwargs
from .bser_prior import BSERPriorRuntime
from .framework_provenance import framework_sources, verify_framework_sources
from .provenance import ROOT, digest, file_sha256, write_json
from .registry import CONTRACTS, method_spec, load_reference, task_conditions, validate_method_config


def normalize_row(row, spec):
    value = copy.deepcopy(row)
    if value.get("method") != spec["runtime_method"]:
        raise ValueError("native evaluator returned a different method identity")
    if value.get("optimizer_update_count") != 0:
        raise ValueError("evaluation must not perform training updates")
    value.update(runtime_method=value["method"], method=spec["method"], planner_mode=spec["planner"],
                 learning_mode=spec["learning_mode"], residual_source=spec["residual_source"], training_update=False)
    return validated_rows([value], require_complete=True)[0]


def unified_summary(rows, spec, episodes, *, finalized, wall_seconds, actual_steps):
    result = prior_evaluation.summary(rows, episodes, finalized=finalized,
                                     wall_seconds=wall_seconds, actual_steps=actual_steps)
    result.update(schema="ch3.baseline_framework.summary.v1", method=spec["method"],
        runtime_method=spec["runtime_method"], planner_mode=spec["planner"], learning_mode=spec["learning_mode"],
        residual_source=spec["residual_source"], training_required=spec["training_required"],
        # Prior runtimes assert zero each step; the native learned evaluator
        # does not expose a measured residual maximum.
        residual_action_max_abs=None if spec["learning"] else 0.0,
        optimizer_update_count=0, training_update=False, evaluation_policy_mode="deterministic" if spec["learning"] else "prior_only",
        completed_episode_environment_steps=sum(row["episode_length"] for row in rows),
        actual_environment_steps_complete=actual_steps is not None)
    return result


def evaluation_plan(baseline, manifest, output_dir, *, episodes=100, seed=12729,
                    reference_training_config=None, checkpoint=None):
    spec = method_spec(baseline)
    reference_path, reference = load_reference(reference_training_config)
    conditions = task_conditions(reference, episodes=episodes, seed=seed)
    output = validated_output(reference, output_dir)
    sources = framework_sources()
    selected = prior_evaluation.select_manifest(reference, episodes, seed, manifest, sources)
    selected["schema"] = "ch3.baseline_framework.manifest.v1"
    config = reference
    checkpoint_identity = None
    if spec["learning"]:
        if checkpoint is None:
            raise ValueError(f"{baseline} requires its independently trained --checkpoint; an HGR checkpoint cannot be relabeled")
        checkpoint = Path(checkpoint).resolve()
        payload = load_checkpoint(checkpoint, expected_baseline=baseline)
        config = validate_method_config(spec, payload["config"], reference)
        if any(payload["source_identity"][key] != sources["production"][key] for key in ("files", "sha256")):
            raise ValueError("baseline checkpoint production source mismatch")
        checkpoint_identity = dict(path=str(checkpoint), file_sha256=file_sha256(checkpoint),
            runtime_method=config["method"], algorithm=config["algorithm"],
            config_hash=payload["config_hash"], completed_main=payload["completed_main_trajectories"], optimizer_updates=payload["optimizer_updates"])
    elif checkpoint is not None:
        raise ValueError("B0/B1 do not accept a checkpoint")
    else:
        evaluation_config = json.loads((ROOT / spec["evaluation_config"]).read_text(encoding="utf-8"))
        if evaluation_config != {"baseline": baseline, **spec}:
            raise ValueError("prior evaluation configuration differs from registry")
    effective = (baseline_environment_kwargs(config) if baseline == "B0_search_prior" else
                 native_evaluation.resolved_config(config, episodes, seed, "stochastic")["environment_config"])
    streams = dict(environment_innovation_seed="seed + zero_based_episode_index; production seed_innovations before construction and after reset",
        seed=seed, policy_sampling="none; deterministic MADDPG actor" if spec["learning"] else "none",
        scope="same seeds do not guarantee identical random events along different trajectories")
    comparable = dict(common_task_conditions=conditions, selected_content_sha256=selected["selected_content_sha256"],
                      evaluation_seed=seed, episodes=episodes)
    resolved = dict(schema="ch3.baseline_framework.resolved.v1", baseline=baseline, **spec,
        planner_mode=spec["planner"], episodes=episodes, seed=seed, device="cpu", collection_mode="serial",
        training_update=False, optimizer_update_count=0, reference_training_method=reference["method"],
        reference_config_path=str(reference_path), reference_config_sha256=file_sha256(reference_path),
        reference_kind="run_config_reference" if "outputs" in reference_path.parts else "default_config_reference",
        common_task_conditions=conditions, effective_environment_config=effective,
        native_runtime_config=config, checkpoint=checkpoint_identity, random_streams=streams,
        comparable_inputs_sha256=digest(comparable),
        comparison_scope="task/scene/seed inputs only; not a paired-trajectory or equal-training-cost guarantee",
        standby_intervention={"pse_use_standby": False, "standby_mode": "initial_position_hold"} if baseline == "B0_search_prior" else None)
    return dict(spec=spec, config=config, resolved=resolved, selected=selected, sources=sources, output=output)


def evaluate(baseline, manifest, output_dir, *, episodes=100, seed=12729,
             reference_training_config=None, checkpoint=None):
    plan = evaluation_plan(baseline, manifest, output_dir, episodes=episodes, seed=seed,
                          reference_training_config=reference_training_config, checkpoint=checkpoint)
    spec, config, resolved, selected, sources, output = (plan[k] for k in ("spec", "config", "resolved", "selected", "sources", "output"))
    frozen = digest(dict(spec=spec, config=config, resolved=resolved, selected=selected))
    identity = dict(schema="ch3.baseline_framework.identity.v1", method=spec["method"],
        runtime_method=spec["runtime_method"], baseline=baseline, resolved_config_sha256=digest(resolved),
        reference_config_path=resolved["reference_config_path"],
        task_protocol=resolved["common_task_conditions"]["task_protocol"],
        reward_objective=resolved["common_task_conditions"]["reward_objective"],
        sources_before=sources, checkout_before=checkout_identity(),
        manifest_file_sha256=selected["origin"]["original_file_sha256"], selected_content_sha256=selected["selected_content_sha256"],
        comparable_inputs_sha256=resolved["comparable_inputs_sha256"], checkpoint=resolved["checkpoint"])
    identity["start_identity_sha256"] = digest(identity)

    def verify():
        verify_framework_sources(sources)
        if file_sha256(resolved["reference_config_path"]) != resolved["reference_config_sha256"]:
            raise ValueError("reference config changed during evaluation")
        if file_sha256(manifest) != identity["manifest_file_sha256"]:
            raise ValueError("manifest changed during evaluation")
        if resolved["checkpoint"] and file_sha256(resolved["checkpoint"]["path"]) != resolved["checkpoint"]["file_sha256"]:
            raise ValueError("checkpoint changed during evaluation")
        if frozen != digest(dict(spec=spec, config=config, resolved=resolved, selected=selected)):
            raise ValueError("evaluation inputs changed in memory")

    verify()
    validated_output(config, output)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("resolved_config.json", resolved), ("evaluation_manifest.json", selected), ("identity.json", identity)):
        write_json(output/name, value)
    input_hashes = {name: file_sha256(output/name) for name in ("resolved_config.json", "evaluation_manifest.json", "identity.json")}
    rows, diagnostics = [], []
    started, actual_steps = time.perf_counter(), None if spec["learning"] else 0
    runtime, scenario, index = None, None, None
    native = output / "native_evaluation"

    def save(status, finalized=False):
        result = unified_summary(rows, spec, episodes, finalized=finalized,
                                 wall_seconds=time.perf_counter()-started, actual_steps=actual_steps)
        write_json(output/"episodes.json", rows)
        write_json(output/"summary.json", result)
        write_json(output/"evaluation_progress.json", dict(status=status, n_valid_episodes=len(rows),
            n_expected_episodes=episodes, evaluation_complete=result["evaluation_complete"], actual_environment_steps=actual_steps,
            native_progress_path="native_evaluation/evaluation_progress.json" if spec["learning"] else None))
        write_json(output/"controller_diagnostics.json", dict(episodes=diagnostics,
            diagnostic_scope="prior runtimes measured per step" if not spec["learning"] else "native learned evaluator; no invented actor/sampler counters"))
        return result

    def finish_identity():
        try:
            identity["sources_after"] = framework_sources()
        except Exception as exc:
            from chapter3_bser.experiments.hgr.provenance import fresh_source_identity
            from .provenance import baseline_source_identity
            identity["sources_after"] = dict(validation_error=str(exc), production=fresh_source_identity(), baseline=baseline_source_identity())
        identity["checkout_after"] = checkout_identity()
        identity["end_state_sha256"] = digest(identity["sources_after"])
        write_json(output/"identity.json", identity)

    def harvest_native():
        nonlocal actual_steps
        if (native/"episodes.json").exists():
            values = json.loads((native/"episodes.json").read_text(encoding="utf-8"))
            for row in values[len(rows):]:
                rows.append(normalize_row(row, spec))
            actual_steps = sum(r["episode_length"] for r in rows)

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        save("running")
        with (output/"run_console.log").open("x", encoding="utf-8") as log:
            if spec["learning"]:
                learned_evaluation.evaluate(checkpoint, native, episodes=episodes, seed=seed,
                                            policy_mode="deterministic", manifest=manifest)
                verify()
                harvest_native()
                native_summary = json.loads((native/"summary.json").read_text(encoding="utf-8"))
                if not native_summary["evaluation_complete"] or len(rows) != episodes:
                    raise ValueError("native evaluation did not complete the requested scenarios")
            else:
                runtime_class = BasicSearchPriorRuntime if baseline == "B0_search_prior" else BSERPriorRuntime
                for index, scenario in enumerate(selected["scenarios"]):
                    verify()
                    episode_started = time.perf_counter()
                    runtime = runtime_class(config, scenario, seed=seed+index, episode_id=index)
                    while not runtime.terminal:
                        before = runtime.step
                        try:
                            runtime.advance()
                        finally:
                            actual_steps += runtime.step - before
                    # Reuse only the existing task/reward row adapter. Its B0
                    # method label is replaced explicitly for this B1 producer.
                    row = prior_evaluation.episode_row(runtime, config, scenario, index, seed+index, time.perf_counter()-episode_started)
                    row["method"] = spec["runtime_method"]
                    diag = dict(runtime.controller_diagnostics(), scenario_id=scenario["scenario_id"])
                    for field in ("found_step", "handoff_event_step", "handoff_decision_step", "executor_target_received_step",
                                  "capture_contact_step_count", "capture_full_hold_step_count", "first_collision_step", "first_collision_agent_ids", "first_collision_phase"):
                        diag[field] = row.get(field)
                    runtime.close()
                    runtime = None
                    verify()
                    rows.append(normalize_row(row, spec))
                    diagnostics.append(diag)
                    save("running")
                    line = f"[{index+1}/{episodes}] {baseline} scenario={scenario['scenario_id']} outcome={row['failure_stage']} steps={row['episode_length']} wall={row['wall_seconds']:.2f}s"
                    print(line, flush=True)
                    print(line, file=log, flush=True)
            verify()
            if any(file_sha256(output/name) != expected for name, expected in input_hashes.items()):
                raise ValueError("saved framework inputs changed during evaluation")
            finish_identity()
            return save("complete", finalized=True)
    except (Exception, KeyboardInterrupt) as exc:
        failure = dict(evaluation_complete=False, scenario=scenario, evaluation_episode_index=index,
            exception_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(),
            partial_controller_diagnostics=None if runtime is None else runtime.controller_diagnostics())
        if spec["learning"]:
            try:
                harvest_native()
            except Exception as harvesting_error:
                failure["native_harvest_error"] = str(harvesting_error)
            actual_steps = None
            progress_path = native / "evaluation_progress.json"
            if progress_path.exists():
                try:
                    progress = json.loads(progress_path.read_text(encoding="utf-8"))
                    measured = progress.get("actual_environment_steps")
                    if type(measured) is int and measured >= sum(r["episode_length"] for r in rows):
                        actual_steps = measured
                except (ValueError, OSError) as progress_error:
                    failure["native_progress_error"] = str(progress_error)
        if (native/"evaluation_failure.json").exists():
            failure["native_failure"] = json.loads((native/"evaluation_failure.json").read_text(encoding="utf-8"))
        write_json(output/"evaluation_failure.json", failure)
        finish_identity()
        save("failed")
        raise
    finally:
        if runtime is not None:
            runtime.close()
        torch.set_num_threads(previous_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, choices=tuple(CONTRACTS))
    parser.add_argument("--manifest", required=True, help="Existing frozen validation manifest; never generated")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12729)
    parser.add_argument("--output-dir", required=True, help="New/empty directory with a collision_terminal path component")
    parser.add_argument("--reference-training-config", help="Actual HGR run config preferred; omitted uses labeled default_config_reference")
    parser.add_argument("--checkpoint", help="Required for B2/B3, forbidden for B0/B1; native method must match")
    args = parser.parse_args(argv)
    evaluate(**vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
