"""D1: frozen-head prediction and same-state candidate representation audit."""

import copy
import subprocess
from pathlib import Path
import numpy as np

from core.scenarios.ch3_generator_impl import build_scenario_manifests
from . import PROFILE, TRAINING_REFERENCE
from .features import feature_catalog
from .metrics import prediction_summary, window_label
from .provenance import (Progress, atomic_csv, atomic_json, digest, experiment_identity,
                         file_hash, fresh_output, load_checkpoint, overlap_report, read_json,
                         scenarios_from_manifest, training_label_baseline)
from .provenance import ROOT
from .runner import (arguments, historical_inputs, key, make_job, no_op_check,
                     record_exception, run_work)


def training_scenarios(args, payload):
    reasons, source = [], None
    if args.training_manifest:
        source = dict(kind="provided_training_manifest", path=str(args.training_manifest), sha256=file_hash(args.training_manifest))
        scenarios = scenarios_from_manifest(read_json(args.training_manifest))
    else:
        config = read_json(args.training_config)
        source = dict(kind="reconstructed_from_training_config", path=str(args.training_config), sha256=file_hash(args.training_config))
        if config.get("profile") != PROFILE or not all(k in config for k in ("episodes", "seed")):
            return [], dict(source=source, verified=False, reasons=["training_config_does_not_identify_M20_generation"])
        scenarios = build_scenario_manifests(count=int(config["episodes"]), generator_seed=int(config["seed"]),
                                              split="train", profiles=(PROFILE,))[PROFILE]["scenarios"]
        source["generator"] = "core.scenarios.ch3_generator_impl.build_scenario_manifests"
        try:
            differences = subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", "diff", "--name-only", TRAINING_REFERENCE, "--", "core"], cwd=ROOT).decode().splitlines()
            # The registered-method metadata evolution is unrelated to generation.
            differences = [p for p in differences if p != "core/registry/experiment_registry.py"]
            source["core_differences_from_training_reference"] = differences
            if differences:
                reasons.append("generator_dependency_code_differs_from_training_reference")
        except subprocess.CalledProcessError:
            reasons.append("training_reference_generator_code_unavailable")
    completed = int(payload.get("completed_episode", 0))
    rows = payload.get("episode_metrics", [])
    expected = {key(s) for s in scenarios[:completed]}
    actual = {key(r) for r in rows if "scenario_id" in r and "scenario_seed" in r}
    if not completed or len(rows) != completed or len(actual) != completed or actual != expected:
        reasons.append("checkpoint_training_episode_ids_seeds_do_not_fully_crosscheck_manifest_prefix")
    if len(scenarios) < completed:
        reasons.append("training_manifest_shorter_than_checkpoint")
    return scenarios, dict(source=source, verified=not reasons, reasons=reasons,
                           checkpoint_completed_episodes=completed, checked_episode_rows=len(rows),
                           exclusion_scope="all manifest scenes, including configured not-yet-trained tail (conservative)")


def freeze_scenarios(args, histories, payload):
    if args.manifest:
        manifest = read_json(args.manifest)
        scenarios = scenarios_from_manifest(manifest)
        if len(scenarios) != 30 or manifest.get("generator_seed", manifest.get("scenario_seed")) != 51729:
            raise ValueError("D1 requires the frozen 30-scene seed-51729 manifest")
    else:
        manifest = build_scenario_manifests(count=30, generator_seed=51729, split="validation", profiles=(PROFILE,))[PROFILE]
        scenarios = scenarios_from_manifest(manifest)
    if len({key(s) for s in scenarios}) != 30 or any(s.get("scenario_profile") != PROFILE or s.get("max_steps") != 400 for s in scenarios):
        raise ValueError("invalid D1 scene profile, cutoff or duplicate identity")
    training, training_check = training_scenarios(args, payload)
    reports = dict(training=overlap_report(scenarios, training), historical=overlap_report(scenarios, histories["ON"]["scenarios"]))
    if any(v["overlapping"] for v in reports.values()):
        raise ValueError(f"predeclared D1 manifest overlaps exclusion data: {reports}")
    verified = training_check["verified"] and all(v["physical_fields_complete"] for v in reports.values())
    return manifest, scenarios, dict(status="independence_verified" if verified else "independence_unverified",
                                     training_check=training_check, overlap_checks=reports,
                                     reused_scenario_ids_are_not_alone_overlap=True,
                                     exclusion_fingerprints="seed AND separately layout/initial agents/target motion; no selection by outcome")


def run(args):
    output = fresh_output(args.output_dir)
    progress = Progress(output, 0)
    try:
        model, model_identity, payload = load_checkpoint(args.checkpoint)
        histories = historical_inputs(args.historical_off_output, args.historical_on_output, model_identity, payload)
        manifest, scenarios, independence = freeze_scenarios(args, histories, payload)
        selected = scenarios[:2] if args.smoke else scenarios
        config = copy.deepcopy(histories["OFF"]["config"])
        identity = experiment_identity(config, model_identity, manifest, workers=args.workers, mode="smoke" if args.smoke else "diagnostic")
        identity.update(independence=independence, scenario_seed=51729, bootstrap_seed=61729,
                        scenarios=scenarios, selected_scenarios=[key(s) for s in selected], frozen_before_rollouts=True,
                        historical_sources={k: histories[k]["source_hashes"] for k in ("OFF", "ON")},
                        feature_contract=feature_catalog(), checkpoint_history=histories["checkpoint_verification"])
        atomic_json(output/"audit_manifest.json", identity)
        atomic_json(output/"scenario_manifest.json", manifest)
        atomic_json(output/"resolved_audit_config.json", config)
        baseline = training_label_baseline(payload)
        atomic_json(output/"training_label_baseline.json", baseline)
        metadata_payload = {k: payload[k] for k in ("metadata", "completed_episode", "schema")}
        del payload
        work = []
        # Identical OFF runtime: bare, fingerprint-only, and shadow/candidate
        # observer. No guided allocator is created in any of these three runs.
        for index, scenario in enumerate(selected):
            # checkpoint_info is independent of replay; retain only tiny metadata.
            job = make_job(model, model_identity, metadata_payload, config, scenario, index, manifest)
            for branch in ("bare", "control", "shadow"):
                work.append(dict(unit=f"{index}:{branch}", scenario_id=scenario["scenario_id"], branch=branch, job=job,
                                 no_hook=branch == "bare", options=dict(shadow=branch == "shadow", capture=branch == "shadow")))
        progress.total = len(work)
        predictions, consistency, representation, scores, generations, checks, vectors, outputs = [], [], [], [], [], [], {}, {}
        def save(result):
            outputs[result["unit"]] = result
            if result["unit"].endswith(":shadow"):
                audit = result["audit"]
                predictions.extend(dict(r, **window_label(r["step"], audit["found_step"], audit["observed_until"])) for r in audit["predictions"])
                consistency.extend(audit["feature_consistency"])
                representation.extend(audit["candidate_representation"])
                scores.extend(audit["candidate_scores"])
                generations.extend(audit["candidate_generation"])
                vectors.update(audit["feature_vectors"])
                atomic_csv(output/"prediction_rows.csv", sorted(predictions, key=lambda r: (*key(r), r["step"], r["agent_id"])))
                atomic_csv(output/"feature_consistency.csv", sorted(consistency, key=lambda r: (*key(r), r["step"], r["agent_id"])))
                atomic_csv(output/"candidate_representation.csv", sorted(representation, key=lambda r: (*key(r), r["step"], r["agent_id"])))
                atomic_csv(output/"candidate_scores.csv", sorted(scores, key=lambda r: (*key(r), r["step"], r["algorithm"], r["greedy_round"], r["candidate_key"])))
                atomic_csv(output/"candidate_generation.csv", sorted(generations, key=lambda r: (*key(r), r["generation_index"])))
        run_work(work, args.workers, progress, on_result=save)
        for index, scenario in enumerate(selected):
            check = no_op_check(outputs[f"{index}:control"]["audit"], outputs[f"{index}:shadow"]["audit"], outputs[f"{index}:bare"]["payload"])
            checks.append(dict(scenario_id=scenario["scenario_id"], scenario_seed=scenario["scenario_seed"], **check,
                               weights=outputs[f"{index}:shadow"]["audit"]["weights"]))
        atomic_json(output/"no_op_validation.json", dict(passed=all(c["passed"] for c in checks), scenarios=checks))
        if not all(c["passed"] for c in checks):
            raise RuntimeError("shadow no-op validation failed; prediction outputs are invalid")
        summary, bins = prediction_summary(sorted(predictions, key=lambda r: (*key(r), r["step"], r["agent_id"])), baseline)
        summary.update(independence=independence, mode=identity["mode"], planned_scenarios=30, evaluated_scenarios=len(selected))
        atomic_json(output/"prediction_summary.json", summary)
        atomic_csv(output/"calibration_bins.csv", bins)
        if args.save_features:
            ids = sorted(vectors)
            np.savez_compressed(output/"raw_features.npz", feature_ids=np.asarray(ids), features=np.asarray([vectors[k] for k in ids], dtype=np.float32))
        if file_hash(args.checkpoint) != model_identity["checkpoint_sha256"]:
            raise RuntimeError("checkpoint file hash changed")
        progress.write(stage="finished", status="completed")
        return 0
    except Exception:
        return record_exception(output, progress)


def main(argv=None):
    return run(arguments(__doc__, argv, prediction=True))


if __name__ == "__main__":
    raise SystemExit(main())
