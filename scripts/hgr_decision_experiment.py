"""Bounded HGR research screen; default CLI is a read-only preflight.

This orchestrates the unchanged Chapter 3 runtime/estimators. It never resumes a
Trainer. Each natural suffix update starts from the same real policy-only source;
theta is never updated. Different updated targets are never pooled as repeats.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import traceback

import numpy as np
import torch

from chapter3_bser.experiments.hgr import phase2_gradient_efficiency as phase2
from chapter3_bser.experiments.hgr.provenance import checkout_identity, fresh_source_identity, require_source_match
from chapter3_bser.experiments.hgr.train import load_config as load_runtime, validated_output, write_json
from chapter3_bser.models.hgr.estimator import prefix_losses, gradient_vector
from chapter3_bser.models.hgr.phase1 import (
    STREAM_REVISION, PairNoise, NoUpdateProof, behavior_identity, canonical,
    identical_behavior, isolated_global_rng, named_seed, phase1_options, runtime_contract,
)
from chapter3_bser.models.hgr.policy import weights_hash
from core.scenarios.ch3_generator_impl import build_scenario_manifests
from scripts.hgr_suffix_diagnostics import diagnose_suffix_update
from scripts.hgr_decision_statistics import summarize_pair

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hgr.decision_experiment.v1"
DEFAULT_CONFIG = ROOT / "configs/chapter3/hgr_decision_experiment.json"
SCRIPT_FILES = ("scripts/hgr_decision_experiment.py", "scripts/hgr_suffix_diagnostics.py",
                "scripts/hgr_decision_statistics.py")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def load_config(path=DEFAULT_CONFIG):
    return json.loads(_path(path).read_text(encoding="utf-8"))


def resolve_config(config):
    cfg = copy.deepcopy(dict(config))
    fields = {"schema", "runtime_config", "snapshot_source", "starting_policy", "snapshot_count",
              "update_replicas", "suffix_batch_size", "macro_repeats", "correction_draws", "seed",
              "random_source_revision", "max_environment_steps", "output_dir", "thresholds"}
    if set(cfg) != fields or cfg["schema"] != SCHEMA:
        raise ValueError("invalid decision experiment schema/fields")
    if cfg["starting_policy"] != "new_policy" or cfg["random_source_revision"] != STREAM_REVISION:
        raise ValueError("requires the predeclared new_policy start and Phase1 named streams")
    for name in ("snapshot_count", "update_replicas", "suffix_batch_size", "correction_draws", "max_environment_steps"):
        _int(cfg[name], name)
    _int(cfg["macro_repeats"], "macro_repeats", 2)
    if cfg["snapshot_count"] != 10:
        raise ValueError("snapshot_count must be 10; all frozen scenarios are required")
    _int(cfg["seed"], "seed", 0)
    if cfg["seed"] >= 2**63:
        raise ValueError("seed must be less than 2**63")
    for name in ("runtime_config", "snapshot_source", "output_dir"):
        if not isinstance(cfg[name], str) or not cfg[name].strip():
            raise ValueError(f"{name} must be a path")
        cfg[name] = str(_path(cfg[name]))
    runtime = load_runtime(cfg["runtime_config"])
    options = phase1_options(runtime)
    if (not options or runtime["algorithm"] != "hgr" or options["predictor_mode"] != "zero"
            or options["lambda"] != 0 or options["pairing"] != "policy_crn"
            or not options["zero_update_bypass"]):
        raise ValueError("requires unchanged Phase1 HGR: zero predictor, policy_crn, exact bypass")
    if (cfg["correction_draws"] != runtime["correction_draws_per_cycle"]
            or cfg["suffix_batch_size"] != runtime["suffix_training_episodes_per_cycle"]):
        raise ValueError("suffix batch size and fixed K must match the supplied runtime")
    _finite_positive(runtime["suffix_lr"], "runtime suffix_lr")
    thresholds = cfg["thresholds"]
    if not isinstance(thresholds, dict) or set(thresholds) != {
            "relative_signal_tolerance", "noise_scale_multiplier", "min_paired_queries_per_replica"}:
        raise ValueError("explicit screening thresholds required")
    _finite_positive(thresholds["relative_signal_tolerance"], "relative_signal_tolerance")
    _finite_positive(thresholds["noise_scale_multiplier"], "noise_scale_multiplier")
    _int(thresholds["min_paired_queries_per_replica"], "min_paired_queries_per_replica")
    return cfg, runtime


def budget_plan(cfg, runtime, diagnostics_only=False):
    h, u, b = runtime["max_steps"], cfg["update_replicas"], cfg["suffix_batch_size"]
    suffix = u*b*h
    comparison = 0 if diagnostics_only else u*cfg["macro_repeats"]*(2*cfg["snapshot_count"]+2*cfg["correction_draws"])*h
    return dict(suffix_update_max_steps=suffix, comparison_max_steps=comparison,
                total_max_steps=suffix+comparison, configured_cap=cfg["max_environment_steps"],
                reservation="whole predeclared experiment, worst-case full horizon",
                wall_clock_limit=False, stop_long_branches_early=False)


def _prepare(config, *, diagnostics_only=False, source_override=None, synthetic=False):
    cfg, runtime = resolve_config(config)
    output = validated_output(runtime, cfg["output_dir"])
    if source_override is None:
        source_path = Path(cfg["snapshot_source"])
        if not source_path.is_file():
            raise ValueError(f"snapshot_source missing: {source_path}")
        source = phase2.load_snapshot_source(source_path, runtime)
        source_sha = _sha(source_path)
    else:
        if not synthetic:
            raise ValueError("in-memory sources are only supported by synthetic tests")
        source, source_sha = source_override, None
    phase2._validate_source(source, dict(scope="fresh_main", snapshot_count=cfg["snapshot_count"],
                                       runtime_config=runtime), synthetic=synthetic)
    contract = runtime_contract(runtime)
    if not identical_behavior(behavior_identity(source.old_policy.theta_minus, contract),
                              behavior_identity(source.new_policy.theta_minus, contract)):
        raise ValueError("frozen source theta identities differ")
    budget = budget_plan(cfg, runtime, diagnostics_only)
    if budget["total_max_steps"] > cfg["max_environment_steps"]:
        raise ValueError("max_environment_steps cannot reserve the complete experiment; no rollout started")
    base = source.new_policy
    audit = dict(status="PREFLIGHT_PASS", schema=SCHEMA, source_schema=phase2.SOURCE_SCHEMA,
                 snapshot_source=cfg["snapshot_source"], snapshot_source_sha256=source_sha,
                 source_identity=source.source_identity, checkout=checkout_identity(),
                 script_sha256={p: _sha(ROOT/p) for p in SCRIPT_FILES},
                 runtime_config_sha256=_sha(cfg["runtime_config"]),
                 runtime_contract_sha256=phase2._digest(contract),
                 starting_policy="new_policy", starting_behavior_sha256=behavior_identity(base, contract).sha256,
                 snapshot_count=len(source.cases), update_replicas=cfg["update_replicas"],
                 macro_repeats=cfg["macro_repeats"], budget=budget,
                 diagnostics_only=bool(diagnostics_only), output_dir=str(output),
                 output_state="empty" if output.exists() else "not_created",
                 experiment_started=False, environment_steps=0, trainer_resume=False)
    return cfg, runtime, source, output, audit


def preflight(config=DEFAULT_CONFIG, *, diagnostics_only=False):
    config = load_config(config) if isinstance(config, (str, Path)) else config
    with isolated_global_rng():
        return _prepare(config, diagnostics_only=diagnostics_only)[4]


class DecisionBackend(phase2.RuntimeBackend):
    def training_case(self, runtime, seed, replica, index):
        manifests = build_scenario_manifests(count=1, generator_seed=seed, split="train",
                                             profiles=[runtime["profile"]])
        scenario = manifests[runtime["profile"]]["scenarios"][0]
        return dict(snapshot_id=f"update_{replica:02d}_{index:02d}", scenario=scenario)


def decide(replicas, thresholds, *, diagnostics_only=False):
    """A predeclared research screen, deliberately not a confidence test."""
    base = dict(rule="empirical_screen_not_confidence_bound", performance_claims_supported=False,
                scope="these predeclared natural updates and fixed evaluation scenarios",
                thresholds=copy.deepcopy(thresholds), universal_ineffectiveness_proven=False)
    if not any(r["diagnostics"]["effective_suffix_score_steps"] for r in replicas):
        return dict(base, status="NOT_EXERCISED", reason="no effective natural suffix score records",
                    next_action="pause; do not retry until handoff")
    changed = [r for r in replicas if r["diagnostics"]["behavior"]["phi_identity_changed"]]
    if not changed:
        return dict(base, status="STOP_CURRENT_HGR", reason="all predeclared updates left phi behavior exactly unchanged",
                    next_action="investigate suffix learning; correction target is exactly zero for these updates")
    if diagnostics_only:
        return dict(base, status="INCONCLUSIVE", reason="diagnostics-only: correction not measured",
                    next_action="review suffix diagnostics before commissioning a separate screen")
    assessments = []
    for row in changed:
        summary = row["statistics"]; signal = summary["signal"]
        support = row["paired_query_count"] >= thresholds["min_paired_queries_per_replica"]
        scale = signal["g0_rms_norm"]
        if not support or not scale or not math.isfinite(scale):
            classification = "INSUFFICIENT_SUPPORT"
            upper_scale = lower_scale = None
        else:
            center = signal["mean_delta_norm"] / scale
            noise = thresholds["noise_scale_multiplier"] * signal["mean_standard_error_scale"] / scale
            upper_scale, lower_scale = center+noise, max(0., center-noise)
            tolerance = thresholds["relative_signal_tolerance"]
            classification = "LOW_OBSERVED_SIGNAL" if upper_scale < tolerance else (
                "MATERIAL_OBSERVED_SIGNAL" if lower_scale >= tolerance else "NOISE_UNRESOLVED")
        assessments.append(dict(replica_id=row["replica_id"], classification=classification,
                                upper_screen_scale=upper_scale, lower_screen_scale=lower_scale,
                                these_are_confidence_bounds=False))
    kinds = [r["classification"] for r in assessments]
    if all(x == "LOW_OBSERVED_SIGNAL" for x in kinds):
        return dict(base, status="STOP_CURRENT_HGR", reason="all measured nonzero updates below the predeclared screening tolerance",
                    assessments=assessments, next_action="do not expand current HGR; prioritize suffix-learning diagnosis")
    if any(x == "MATERIAL_OBSERVED_SIGNAL" for x in kinds):
        return dict(base, status="CONTINUE_SIGNAL_STUDY", reason="at least one predeclared update has material observed correction",
                    assessments=assessments,
                    next_action="validate on independent updates and actual matched-cost means; no efficiency advantage established")
    return dict(base, status="INCONCLUSIVE", reason="query coverage or repeat precision insufficient at the fixed budget",
                assessments=assessments, next_action="pause module expansion; review evidence, do not auto-increase budget")


class Experiment:
    def __init__(self, cfg, runtime, source, output, audit, backend):
        self.cfg, self.runtime, self.source, self.output = cfg, runtime, source, output
        self.audit, self.backend = audit, backend
        self.contract = runtime_contract(runtime)
        self.base = copy.deepcopy(source.new_policy)
        self.base_identity = behavior_identity(self.base, self.contract)
        self.costs = {k: 0 for k in ("suffix_training_steps", "hgr_main_steps", "direct_new_mc_steps",
                                    "hgr_correction_old_steps", "hgr_correction_new_steps")}
        self.inflight = None
        self.active_pair = None
        self.pair_ids = set()
        self.report = dict(schema=SCHEMA, status="RUNNING", config=cfg, preflight=audit,
                           replicas=[], costs=self.costs, parameter_updates=0, theta_updates=0,
                           trainer_resume=False, formal_experiment=False, performance_claims_supported=False,
                           source_preparation_steps=source.preparation_environment_steps,
                           budget=audit["budget"], reference_rollouts=0,
                           reference_note="Independent Direct macro means anchor risk diagnostics; no exact gradient truth.")

    def stream(self, replica, purpose, index=0):
        return named_seed(self.cfg["seed"], replica+1, "hgr_decision/"+purpose, index)

    def persist(self):
        self.report["total_new_environment_steps"] = sum(self.costs.values())
        self.report["total_including_historical_preparation_steps"] = sum(self.costs.values())+self.source.preparation_environment_steps
        write_json(self.output/"decision_results.json", self.report)

    def guard(self):
        if not identical_behavior(self.base_identity, behavior_identity(self.base, self.contract)):
            raise RuntimeError("common starting policy was changed")
        if self.active_pair is not None:
            for policy, identity in self.active_pair:
                if not identical_behavior(identity, behavior_identity(policy, self.contract)):
                    raise RuntimeError("frozen comparison policy changed")
        if sum(self.costs.values()) > self.audit["budget"]["total_max_steps"]:
            raise RuntimeError("actual step ledger exceeds reserved experiment bound")

    def arrays(self, name, **arrays):
        path = self.output/(name+".npz")
        for value in arrays.values():
            if not np.isfinite(value).all():
                raise ValueError("nonfinite saved diagnostic vector")
        with path.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        return dict(path=path.name, sha256=_sha(path),
                    arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k,v in arrays.items()})

    def save_pt(self, name, payload):
        path = self.output/name
        with path.open("xb") as handle:
            torch.save(phase2._pack(payload), handle)
        return dict(path=path.name, sha256=_sha(path), schema=payload["schema"])

    def collect(self, replica, case, policy, purpose, index, cost_field):
        stream_id, seed = self.stream(replica, purpose, index)
        self.inflight = dict(operation="collect", replica_id=replica,
                             random_stream_id=stream_id, scenario_id=case["snapshot_id"])
        trajectory = self.backend.collect(self.runtime, copy.deepcopy(case), policy, seed=seed, stream_id=stream_id)
        self.costs[cost_field] += len(trajectory["records"])
        phase2._validate_trajectory(trajectory, policy, self.contract, self.runtime["max_steps"])
        self.guard()
        row = dict(snapshot_id=case["snapshot_id"], scenario_seed=case["scenario"]["scenario_seed"],
                   tau=trajectory["tau"], snapshot_hash=None if trajectory["snapshot"] is None else trajectory["snapshot"].sha256,
                   random_stream_id=stream_id, environment_steps=len(trajectory["records"]),
                   summary=trajectory.get("summary", {}))
        self.inflight = None
        return trajectory, row

    def branch(self, snapshot, policy, noise, cost_field):
        self.inflight = dict(operation="branch", **noise.public())
        result = self.backend.branch(snapshot, policy, noise, keep_records=False)
        _int(result["steps"], "branch steps")
        self.costs[cost_field] += result["steps"]
        if (result["terminal_step"] != snapshot.step+result["steps"]
                or result["terminal_step"] > snapshot.max_steps or not math.isfinite(result["G_plus"])):
            raise ValueError("branch return/clock contract mismatch")
        self.guard()
        self.inflight = None
        return {k:v for k,v in result.items() if k != "records"}

    def compare_block(self, replica, repeat, old, new, replica_row):
        main, main_rows, direct_rows = [], [], []
        before = dict(self.costs)
        direct_sum = None
        row = dict(replica_id=replica, repeat_id=repeat, status="RUNNING",
                   hgr_main=main_rows, direct_main=direct_rows, correction_queries=[])
        replica_row["macros"].append(row)
        self.persist()
        # Same predeclared scenarios; independent method-specific policy streams.
        for index, case in enumerate(self.source.cases):
            trajectory, row = self.collect(replica, case, old, f"macro/{repeat}/hgr/main", index, "hgr_main_steps")
            main.append(trajectory); main_rows.append(row)
            self.persist()
            direct, row = self.collect(replica, case, new, f"macro/{repeat}/direct/main", index, "direct_new_mc_steps")
            direct_rows.append(row)
            self.persist()
            vector = phase2.estimator_vectors(new, direct, self.runtime["rl"]["gamma"])["g_full"]
            direct_sum = vector if direct_sum is None else direct_sum+vector
            del direct
        row = replica_row["macros"][-1]
        n, k = len(main), self.cfg["correction_draws"]
        selection_id, selection_seed = self.stream(replica, f"macro/{repeat}/formal_selection")
        indices = np.random.default_rng(selection_seed).choice(n, size=k, replace=True)
        draws, queries = [], row["correction_queries"]
        for draw, index in enumerate(indices):
            index = int(index); trajectory = main[index]
            noises = {role: PairNoise.make(self.cfg["seed"], replica+1,
                      f"hgr_decision/macro/{repeat}/correction", draw, role, "policy_crn") for role in ("old", "new")}
            pair_id = noises["old"].pair_id
            if pair_id in self.pair_ids:
                raise RuntimeError("formal query pair_id reused")
            self.pair_ids.add(pair_id)
            query = dict(draw=draw, main_index=index, q=1/n, pair_id=pair_id,
                         snapshot_id=main_rows[index]["snapshot_id"], snapshot_hash=main_rows[index]["snapshot_hash"],
                         tau=trajectory["tau"], label=None, no_handoff=trajectory["tau"] is None, status="RUNNING")
            queries.append(query)
            self.persist()
            if trajectory["tau"] is not None:
                query["old"] = self.branch(trajectory["snapshot"], old, noises["old"], "hgr_correction_old_steps")
                self.persist()
                query["new"] = self.branch(trajectory["snapshot"], new, noises["new"], "hgr_correction_new_steps")
                query["label"] = query["new"]["G_plus"]-query["old"]["G_plus"]
            else:
                query["label"] = 0.
            query["status"] = "PASS"
            draws.append((index, 1/n, query["label"]))
            self.persist()
        with torch.enable_grad():
            losses = prefix_losses(new, main, [0.]*n, draws, gamma=self.runtime["rl"]["gamma"], method="hgr")
            params = list(new.theta_minus.parameters())
            g0 = -gradient_vector(losses["old"], params).cpu().numpy().astype(np.float64)
            delta = -(gradient_vector(losses["prediction"], params)+gradient_vector(losses["correction"], params)).cpu().numpy().astype(np.float64)
        vectors = dict(g0=g0, g_delta=delta, g_full=g0+delta, direct_new_mc=direct_sum/n)
        steps = {key: self.costs[key]-before[key] for key in self.costs}
        row.update(status="PASS", main_batch_size=n, fixed_K=k,
                   selection_stream_id=selection_id, selection_seed=selection_seed,
                   hgr_main=main_rows, direct_main=direct_rows, correction_queries=queries,
                   costs=steps, hgr_steps=steps["hgr_main_steps"]+steps["hgr_correction_old_steps"]+steps["hgr_correction_new_steps"],
                   direct_steps=steps["direct_new_mc_steps"],
                   phi0_hash=weights_hash(old.phi), phi1_hash=weights_hash(new.phi),
                   theta_hash=weights_hash(new.theta_minus),
                   vector_norms={key: float(np.linalg.norm(value)) for key,value in vectors.items()})
        row["vectors"] = self.arrays(f"replica_{replica:02d}_macro_{repeat:02d}", **vectors)
        self.guard()
        return vectors, row

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        # Exclusive ownership prevents races or overwriting retained run evidence.
        with (self.output/"run.lock").open("x", encoding="utf-8") as handle:
            handle.write("No automatic resume. Preserve this directory on success or failure.\n")
        self.persist()
        try:
            with isolated_global_rng():
                # All training cases are declared before any outcome is observed.
                planned = []
                for replica in range(self.cfg["update_replicas"]):
                    batch = []
                    for index in range(self.cfg["suffix_batch_size"]):
                        stream_id, seed = self.stream(replica, "suffix_training/scenario", index)
                        case = self.backend.training_case(self.runtime, seed, replica, index)
                        batch.append(dict(case=case, scenario_stream_id=stream_id, generator_seed=seed))
                    planned.append(batch)
                plan = dict(schema=SCHEMA, config=self.cfg, audit=self.audit,
                            evaluation_cases=self.source.cases, training_batches=planned,
                            update_schedule="independent one-step updates from the identical starting policy",
                            parameters_updated="phi only; no theta step; no chained training",
                            all_nonidentical_updates_measured=True, approximate_zero_bypass=False)
                with (self.output/"plan.json").open("x", encoding="utf-8") as handle:
                    json.dump(plan, handle, ensure_ascii=False, indent=2, allow_nan=False)
                for replica, batch in enumerate(planned):
                    self.active_pair = None
                    old = copy.deepcopy(self.base)
                    trajectories, metadata = [], []
                    row = dict(replica_id=replica, suffix_trajectories=metadata, comparison_status="PENDING",
                               macros=[], paired_query_count=0)
                    self.report["replicas"].append(row)
                    self.persist()
                    for index, item in enumerate(batch):
                        trajectory, collected = self.collect(replica, item["case"], old, "suffix_training/main", index, "suffix_training_steps")
                        trajectories.append(trajectory); metadata.append(collected)
                        self.persist()
                    row["training_records"] = self.save_pt(f"replica_{replica:02d}_training_records.pt", dict(
                        schema="hgr.decision.training_records.v1", source_identity=self.source.source_identity,
                        trajectories=[{k:v for k,v in t.items() if k != "snapshot"} for t in trajectories]))
                    self.inflight = dict(operation="one_suffix_update", replica_id=replica)
                    new, diagnostic, arrays = diagnose_suffix_update(old, trajectories,
                        gamma=self.runtime["rl"]["gamma"], lr=self.runtime["suffix_lr"], contract=self.contract)
                    self.report["parameter_updates"] += 1
                    self.inflight = None
                    row["diagnostics"] = diagnostic
                    row["diagnostic_vectors"] = self.arrays(f"replica_{replica:02d}_suffix", **arrays)
                    del trajectories, arrays
                    policies = {name: dict(state=p.state_dict(), training_modes={k:m.training for k,m in p.named_modules()},
                                            behavior_sha256=behavior_identity(p, self.contract).sha256)
                                for name,p in (("old_policy",old),("new_policy",new))}
                    row["policy_pair"] = self.save_pt(f"replica_{replica:02d}_policy_pair.pt", dict(
                        schema="hgr.decision.policy_pair.v1", source_identity=self.source.source_identity,
                        runtime_contract_sha256=phase2._digest(self.contract), replica_id=replica,
                        parent_source_sha256=self.audit["snapshot_source_sha256"],
                        origin="one original suffix SGD update; audit artifact, not a Trainer checkpoint", **policies))
                    self.active_pair = [(p, behavior_identity(p, self.contract)) for p in (old,new)]
                    if not diagnostic["behavior"]["phi_identity_changed"]:
                        NoUpdateProof.create(old.phi, new.phi, self.contract)
                        row.update(comparison_status="EXACT_ZERO_UPDATE", exact_target_correction_zero=True)
                    elif self.audit["diagnostics_only"]:
                        row["comparison_status"] = "DIAGNOSTICS_ONLY"
                    else:
                        data = {key: [] for key in ("g0", "g_delta", "direct_new_mc")}
                        for repeat in range(self.cfg["macro_repeats"]):
                            vectors, macro = self.compare_block(replica, repeat, old, new, row)
                            row["paired_query_count"] += sum(not q["no_handoff"] for q in macro["correction_queries"])
                            for key in data:
                                data[key].append(vectors[key])
                            self.persist()
                        row["statistics"] = summarize_pair(np.stack(data["g0"]), np.stack(data["g_delta"]),
                            np.stack(data["direct_new_mc"]), dict(hgr_steps=[m["hgr_steps"] for m in row["macros"]],
                                                                 direct_steps=[m["direct_steps"] for m in row["macros"]]))
                        row["comparison_status"] = "EXERCISED" if row["paired_query_count"] else "NOT_EXERCISED"
                        del data, vectors
                    self.guard(); self.persist()
                require_source_match(self.audit["source_identity"], fresh_source_identity(), context="decision experiment end")
                if any(_sha(ROOT/p) != sha for p,sha in self.audit["script_sha256"].items()):
                    raise RuntimeError("decision scripts changed during run")
                if self.audit["snapshot_source_sha256"] is not None and _sha(self.cfg["snapshot_source"]) != self.audit["snapshot_source_sha256"]:
                    raise RuntimeError("frozen source file changed during run")
                if _sha(self.cfg["runtime_config"]) != self.audit["runtime_config_sha256"]:
                    raise RuntimeError("runtime config changed during run")
                self.report.update(status="PASS", frozen_theta_check="PASS", original_source_unchanged=True,
                    suffix_sgd_calls=self.report["parameter_updates"],
                    changed_suffix_replicas=sum(r["diagnostics"]["behavior"]["phi_identity_changed"] for r in self.report["replicas"]),
                    decision=decide(self.report["replicas"], self.cfg["thresholds"], diagnostics_only=self.audit["diagnostics_only"]))
                self.persist()
        except BaseException:
            self.report.update(status="FAIL", traceback=traceback.format_exc(), failed_operation=self.inflight,
                               failed_call_steps_unknown=bool(self.inflight and self.inflight["operation"] in ("collect", "branch")))
            self.persist()
            raise
        return self.report


def run_experiment(config, *, diagnostics_only=False, _source=None, _backend=None):
    """Explicit execution API. Test injections cannot enter the production CLI."""
    config = load_config(config) if isinstance(config, (str, Path)) else config
    with isolated_global_rng():
        args = _prepare(config, diagnostics_only=diagnostics_only, source_override=_source, synthetic=_backend is not None)
        return Experiment(*args, _backend or DecisionBackend()).run()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--execute", action="store_true", help="Run bounded natural updates and comparisons; default is preflight only")
    parser.add_argument("--diagnostics-only", action="store_true", help="Skip gradient comparison; do not infer HGR efficiency")
    args = parser.parse_args(argv)
    try:
        if args.execute:
            report = run_experiment(args.config, diagnostics_only=args.diagnostics_only)
            value = dict(status=report["status"], decision=report["decision"],
                         results=str(Path(report["config"]["output_dir"])/"decision_results.json"),
                         total_new_environment_steps=report["total_new_environment_steps"])
        else:
            audit = preflight(args.config, diagnostics_only=args.diagnostics_only)
            value = {k:v for k,v in audit.items() if k not in ("source_identity", "checkout", "script_sha256")}
            value["source_identity_sha256"] = audit["source_identity"]["sha256"]
            value["git_commit"] = audit["checkout"]["head"]
        print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps(dict(status="FAIL" if args.execute else "PREFLIGHT_FAIL", error=str(exc),
                              experiment_requested=args.execute), ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
