"""D2: common-ON-prefix, one-decision A/B/C interventions, then all OFF."""

import copy
from pathlib import Path

from . import MAX_STEPS
from .features import feature_catalog
from .metrics import paired_outcomes, window_label
from .provenance import (Progress, atomic_csv, atomic_json, digest, experiment_identity,
                         file_hash, fresh_output, load_checkpoint)
from .runner import (arguments, execute_work, historical_inputs, key, make_job,
                     no_op_check, record_exception, run_work, task_signature)


def installed_signature(decision):
    return digest([decision["installed_assignment_geometry"], decision["public_guidance_geometry"],
                   decision["effective_guidance_geometry"]])


def root_decision(audit, step):
    return next((r for r in audit["decisions"] if r["step"] == step), None)


def needs_root_probe(decision):
    # ID-only changes are candidates for verification, never automatic delivery.
    return (decision["proposal_accepted"] and decision["step"] < MAX_STEPS and
            (decision["allocation_proposal_changed"] or decision["proposal_candidate_ids_changed"]))


def replay_equal(left, right):
    return (left["status"] == right["status"] and
            left["audit"]["steps"] == right["audit"]["steps"] and
            left["audit"]["terminal_fingerprint"] == right["audit"]["terminal_fingerprint"] and
            left["audit"]["root_fingerprint"] == right["audit"]["root_fingerprint"] and
            left["audit"]["decisions"] == right["audit"]["decisions"] and
            (left["audit"]["payload"] is None or
             task_signature(left["audit"]["payload"]) == task_signature(right["audit"]["payload"])))


def locate_work(work):
    """Full original OFF/ON reproduction, then prefix-only A probes. No outcomes select roots."""
    audits, checks = {}, {}
    for mode in ("OFF", "ON"):
        job = work["jobs"][mode]
        bare = execute_work(dict(unit="bare", job=job, no_hook=True))
        control = execute_work(dict(unit="control", job=job, options=dict(capture=False)))
        observed = execute_work(dict(unit="observed", job=job))
        audits[mode] = observed["audit"]
        check = no_op_check(control["audit"], observed["audit"], bare["payload"])
        historical = work["historical_rows"][mode]
        reproduced = observed["audit"]["payload"]["episode"]
        differences = {k: dict(historical=historical.get(k), reproduced=reproduced.get(k))
                       for k in task_signature(observed["audit"]["payload"])
                       if k in historical and historical[k] != reproduced.get(k)}
        old_count = int(historical.get("search_value_guidance", {}).get("accepted_search_change_count", 0))
        new_count = int(reproduced.get("search_value_guidance", {}).get("accepted_search_change_count", 0))
        check.update(historical_outcome_differences=differences, historical_count=old_count, reproduced_count=new_count,
                     historical_reproduced=not differences and old_count == new_count)
        checks[mode] = check
    on = audits["ON"]
    root, rejected_roots = None, []
    for decision in on["decisions"]:
        if not needs_root_probe(decision):
            continue
        probe = execute_work(dict(unit="probe_A", job=work["jobs"]["ON"],
                                 options=dict(root_step=decision["step"], intervention="A", probe=True)))
        candidate = root_decision(probe["audit"], decision["step"])
        boundary = on["boundaries"][str(decision["step"])]
        state_equal = probe["audit"]["root_fingerprint"] == boundary["fingerprint"]
        prefix_equal = probe["audit"]["prefix_hash"] == boundary["prefix_hash"]
        if not state_equal or not prefix_equal or probe["status"] != "boundary_probe":
            rejected_roots.append(dict(step=decision["step"], reason="common_prefix_or_root_probe_mismatch",
                                       status=probe["status"], state_equal=state_equal, prefix_equal=prefix_equal))
            # Do not hunt a later 'valid-looking' boundary after failed reproduction.
            break
        different = candidate is not None and installed_signature(candidate) != installed_signature(decision)
        if different:
            root = dict(step=decision["step"], original_ON_decision=decision, A_probe_decision=candidate,
                        root_fingerprint=boundary["fingerprint"], prefix_hash=boundary["prefix_hash"],
                        state_equal=state_equal, prefix_equal=prefix_equal)
            break
        rejected_roots.append(dict(step=decision["step"], reason="candidate_ID_or_proposal_change_without_installed_control_change",
                                   state_equal=state_equal, prefix_equal=prefix_equal))
    # The common root uses ON history, not a reset into a whole-episode OFF trajectory.
    step = root["step"] if root else 0
    prefix_fields = ("step", "action_hash", "guidance_hash", "physical_state_hash", "rng_hash")
    prefixes = {mode: [{k: row[k] for k in prefix_fields} for row in audits[mode]["steps"] if row["step"] < step] for mode in ("OFF", "ON")}
    summary = dict(**work["identity"], status="root_located" if root else "mismatch_no_actual_installed_intervention",
                   root=root, checks=checks, rejected_root_candidates=rejected_roots,
                   ON_prefix_matches_historical_OFF=prefixes["OFF"] == prefixes["ON"],
                   prefix_comparison_scope="physical environment, actions, search guidance and RNG; ON controller/scorer histories may differ",
                   decisions=on["decisions"], root_candidate_scores=[r for r in on["candidate_scores"] if root and r["step"] == step])
    return dict(unit=work["unit"], location=summary)


def branch_work(work):
    if work.get("missing_root"):
        return dict(unit=work["unit"], status="mismatch_no_root", branch=work["branch"])
    return execute_work(work)


def branch_outcome(result, location, branch):
    identity = {k: location[k] for k in ("scenario_id", "scenario_seed")}
    row = dict(**identity, branch=branch, status=result["status"], valid_pair=False,
               proposal_accepted=False, treatment_delivered=False)
    root = location["root"]
    if root is None:
        return row
    audit = result["audit"]
    step = root["step"]
    if audit["root_fingerprint"] is None:
        row.update(root_step=step, remaining_steps=MAX_STEPS-step, reason="root_not_reached", weights=audit["weights"])
        return row
    decision = root_decision(audit, step)
    row.update(root_step=step, remaining_steps=MAX_STEPS-step,
               root_state_hash=audit["root_fingerprint"]["sha256"], prefix_hash=audit["prefix_hash"],
               weights=audit["weights"], guided_root_count=audit["guided_root_count"], guided_after_root_count=audit["guided_after_root_count"])
    if decision is None:
        return row
    reference = root["A_probe_decision"]
    previous = decision.get("previous_assignment_geometry", {})
    row.update(candidate_pool_hash=decision["candidate_pool_hash"], proposal_signatures={k: digest(v) if v is not None else None for k, v in decision["proposal_geometries"].items()},
               chosen_proposal_signature=decision["chosen_proposal_signature"], installed_assignment_signature=decision["installed_assignment_signature"],
               installed_guidance_signature=decision["installed_guidance_signature"], proposal_accepted=decision["proposal_accepted"],
               actual_control_differs_from_A=installed_signature(decision) != installed_signature(reference),
               treatment_delivered=decision["proposal_accepted"] and installed_signature(decision) != installed_signature(reference) if branch != "A" else
                   decision["proposal_accepted"] and decision["installed_assignment_geometry"] != previous,
               changed_agent_ids=[int(k) for k in ("0", "1", "2") if any(reference[field].get(k) != decision[field].get(k)
                                  for field in ("installed_assignment_geometry", "public_guidance_geometry", "effective_guidance_geometry"))],
               guidance_duration=audit["guidance_duration"], first_followup_replan=audit["first_followup_replan"],
               proposal_objective=decision["chosen_proposal_objective"], baseline_objective=decision["baseline_objective"])
    if audit["payload"] is not None:
        outcome = audit["payload"]["episode"]
        label = window_label(step, audit["found_step"], audit["observed_until"])
        row.update(found_50=label["label"] if label["main_eligible"] else None, window=label,
                   found=outcome["found"], contact=outcome["contact_episode"], success=outcome["success"], found_step=audit["found_step"],
                   observed_until=audit["observed_until"], searcher_collisions=audit["suffix_collisions"], max_collision_streak=audit["max_collision_streak"],
                   known_ratio_gain=audit["known_ratio_gain"], travel_distance=audit["travel_distance"])
    return row


def validate_branches(location, results):
    if location["root"] is None:
        return dict(passed=False, reason="historical_intervention_not_reproduced")
    root = location["root"]
    checks = dict(A_repeat=replay_equal(results["A"], results["A_repeat"]),
                  root_probe_verified=root["state_equal"] and root["prefix_equal"],
                  order_independent=all(replay_equal(results[b], results[b+"_reverse"]) for b in ("A", "B", "C")),
                  common_root=all(r["audit"]["root_fingerprint"] == root["root_fingerprint"] for r in results.values()),
                  common_prefix=all(r["audit"]["prefix_hash"] == root["prefix_hash"] for r in results.values()),
                  no_op=all(c["passed"] for c in location["checks"].values()),
                  historical_reproduced=all(c["historical_reproduced"] for c in location["checks"].values()),
                  subsequent_OFF=all(r["audit"]["guided_after_root_count"] == 0 and r["audit"]["guided_root_count"] <= 1 for r in results.values()),
                  global_cutoff=all(r["audit"]["observed_until"] <= MAX_STEPS for r in results.values()),
                  immutable_weights=all(r["audit"]["weights"] and r["audit"]["weights"]["actor_before"] == r["audit"]["weights"]["actor_after"] and
                                        r["audit"]["weights"]["head_before"] == r["audit"]["weights"]["head_after"] for r in results.values()))
    delivered = root_decision(results["B"]["audit"], root["step"])
    checks["B_reproduces_ON_delivery"] = (delivered is not None and delivered["proposal_accepted"] and
                                        installed_signature(delivered) == installed_signature(root["original_ON_decision"]) and
                                        installed_signature(delivered) != installed_signature(root["A_probe_decision"]))
    return dict(passed=all(checks.values()), checks=checks,
                interpretation="common-root single intervention; A suffix need not equal whole-episode historical OFF")


def run(args):
    output = fresh_output(args.output_dir)
    progress = Progress(output, 0)
    try:
        model, model_identity, payload = load_checkpoint(args.checkpoint)
        histories = historical_inputs(args.historical_off_output, args.historical_on_output, model_identity, payload)
        all_selected = histories["selected"]
        selected = all_selected[:1] if args.smoke else all_selected
        identity = experiment_identity(histories["ON"]["config"], model_identity, histories["ON"]["manifest"], workers=args.workers,
                                       mode="smoke" if args.smoke else "diagnostic")
        identity.update(all_intervention_scenarios=all_selected, selected_scenarios=selected,
                        historical_intervention_scene_count=len(all_selected), scene_selection="all historical ON accepted_search_change_count>0, manifest order",
                        feature_contract=feature_catalog(), checkpoint_history=histories["checkpoint_verification"],
                        historical_sources={k: histories[k]["source_hashes"] for k in ("OFF", "ON")}, historical_metrics_sha256=histories["metrics_sha256"],
                        training_source=dict(path=str(args.training_config or args.training_manifest), sha256=file_hash(args.training_config or args.training_manifest)),
                        gates="same complete pre-controller root, common prefix, A/A, reverse order, bare OFF/ON, B installed delivery, frozen weights")
        atomic_json(output/"branch_manifest.json", identity)
        atomic_json(output/"resolved_audit_config.json", {k: histories[k]["config"] for k in ("OFF", "ON")})
        locations, outcomes, validations = [], [], []
        all_decisions, all_scores = [], []
        progress.total = 8*len(selected)
        locate_jobs = []
        for index, scenario in enumerate(selected):
            historical_index = next(i for i, s in enumerate(histories["ON"]["scenarios"]) if key(s) == key(scenario))
            jobs = {mode: make_job(model, model_identity, payload, histories[mode]["config"], scenario, historical_index, histories[mode]["manifest"]) for mode in ("OFF", "ON")}
            locate_jobs.append(dict(unit=f"{index}:locate", scenario_id=scenario["scenario_id"], branch="locate", jobs=jobs,
                                    identity=dict(scenario_id=scenario["scenario_id"], scenario_seed=scenario["scenario_seed"]),
                                    historical_rows={mode: histories[mode]["rows"][key(scenario)] for mode in ("OFF", "ON")}))
        del payload
        def save_location(result):
            locations.append(result["location"])
            locations.sort(key=lambda r: key(r))
            atomic_json(output/"historical_reproduction_check.json", dict(scenarios=locations, checkpoint_history=histories["checkpoint_verification"]))
        located = run_work(locate_jobs, args.workers, progress, on_result=save_location, worker=locate_work)
        for index, (work, found) in enumerate(zip(locate_jobs, located)):
            location, results = found["location"], {}
            root = location["root"]
            # Separate groups make the reversed execution order explicit even with
            # two workers. Each job still rebuilds from the original ON seed.
            for names in (("A", "B", "C"), ("A_repeat",), ("C_reverse", "B_reverse", "A_reverse")):
                branches = [dict(unit=f"{index}:{name}", branch=name, scenario_id=work["scenario_id"], job=work["jobs"]["ON"],
                                 missing_root=root is None, options=dict(intervention=name[0], root_step=root["step"] if root else None)) for name in names]
                def save_branch(result):
                    name = result["unit"].split(":", 1)[1]
                    results[name] = result
                    if name in ("A", "B", "C"):
                        outcomes.append(branch_outcome(result, location, name))
                        outcomes.sort(key=lambda r: (*key(r), r["branch"]))
                        atomic_csv(output/"branch_outcomes.csv", sorted(outcomes, key=lambda r: (*key(r), r["branch"])))
                run_work(branches, args.workers, progress, on_result=save_branch, worker=branch_work)
            validation = validate_branches(location, results)
            validations.append(dict(scenario_id=location["scenario_id"], scenario_seed=location["scenario_seed"], **validation))
            for row in outcomes:
                if key(row) == key(location):
                    row["valid_pair"] = validation["passed"] and row["status"] == "completed"
            all_decisions.extend(dict(r, branch="historical_ON_replay") for r in location["decisions"])
            all_scores.extend(dict(r, branch="historical_ON_replay") for r in location["root_candidate_scores"])
            if root:
                for name in ("A", "B", "C"):
                    all_decisions.extend(dict(r, branch=name) for r in results[name]["audit"]["decisions"] if r["step"] == root["step"])
                    all_scores.extend(dict(r, branch=name) for r in results[name]["audit"]["candidate_scores"] if r["step"] == root["step"])
            atomic_csv(output/"decision_audit.csv", all_decisions)
            atomic_csv(output/"candidate_scores.csv", all_scores)
            atomic_csv(output/"branch_outcomes.csv", sorted(outcomes, key=lambda r: (*key(r), r["branch"])))
            atomic_json(output/"replay_validation.json", dict(passed=all(v["passed"] for v in validations), scenarios=validations))
            pairs, summary = paired_outcomes(outcomes)
            atomic_csv(output/"paired_branch_comparison.csv", pairs)
            summary.update(mode=identity["mode"], all_selected_scenes=len(all_selected), completed_scenes=len(validations),
                           mismatches=sum(not v["passed"] for v in validations),
                           rejected_or_undelivered=[r for r in outcomes if not r["treatment_delivered"]],
                           C_unavailable=sum(r["status"] == "C_UNAVAILABLE" for r in outcomes))
            atomic_json(output/"branch_summary.json", summary)
        if not selected:
            raise ValueError("historical metrics select zero intervention scenes; no branch experiment exists")
        if file_hash(args.checkpoint) != model_identity["checkpoint_sha256"]:
            raise RuntimeError("checkpoint file hash changed")
        passed = all(v["passed"] for v in validations)
        progress.write(stage="finished", status="completed" if passed else "completed_with_mismatches")
        return 0 if passed else 2
    except Exception:
        return record_exception(output, progress)


def main(argv=None):
    return run(arguments(__doc__, argv))


if __name__ == "__main__":
    raise SystemExit(main())
