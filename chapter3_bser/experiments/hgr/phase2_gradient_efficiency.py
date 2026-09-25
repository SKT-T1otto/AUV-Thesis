"""Frozen-policy gradient diagnostics. No training, optimizer, or policy update.

fresh_main estimates the initial-distribution prefix gradient. fixed_boundary
reports conditional score-estimator statistics given recorded prefixes.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
import torch

from chapter3_bser.models.hgr.estimator import prefix_losses, gradient_vector
from chapter3_bser.models.hgr.phase1 import (
    STREAM_REVISION, NoUpdateProof, PairNoise, behavior_identity, canonical,
    identical_behavior, isolated_global_rng, named_seed, phase1_options, runtime_contract,
)
from chapter3_bser.models.hgr.policy import HandoffPolicy, weights_hash
from .provenance import fresh_source_identity, require_source_match, checkout_identity
from .runtime import DecisionSnapshot, collect_trajectory, continue_branch
from .train import load_config as load_runtime_config, validate_config, validated_output, write_json

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "hgr.phase2.gradient_efficiency.v1"
SOURCE_SCHEMA = "hgr.phase2.frozen_source.v1"
METHODS = ("direct_new_mc", "old_mc", "hgr")
SCOPES = ("fresh_main", "fixed_boundary")


def _positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def load_config(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    for key in ("runtime_config", "snapshot_source", "output_dir"):
        if isinstance(value.get(key), str) and not Path(value[key]).is_absolute():
            # Repository-relative CLI paths, independent of the shell's cwd.
            value[key] = str(ROOT / value[key])
    return value


def resolve_config(config, num_snapshots=None, num_repeats=None, reference_budget=None):
    value = copy.deepcopy(dict(config))
    if value.get("schema") != SCHEMA or value.get("scope") not in SCOPES:
        raise ValueError("unknown Phase2 schema or scope")
    if value.get("random_source_revision") != STREAM_REVISION:
        raise ValueError("Phase2 requires Phase1 named streams")
    for key, override, default, minimum in (
            ("snapshot_count", num_snapshots, 10, 1), ("repeat_count", num_repeats, 20, 2),
            ("reference_rollouts", reference_budget, 1000, 2)):
        value[key] = _positive_integer(value.get(key, default) if override is None else override, key, minimum)
    n, budget = value["snapshot_count"], value["reference_rollouts"]
    if budget % n or budget // n < 2:
        raise ValueError("reference_budget is TOTAL rollouts, divisible by snapshot_count, with >=2 per case")
    if value.get("reference_budget_unit") != "total_rollouts":
        raise ValueError("reference_budget_unit must explicitly be total_rollouts")
    if value.get("vector_storage") not in ("npz", "inline"):
        raise ValueError("vector_storage must be npz or inline")
    seed = value.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("invalid run seed")
    runtime = value.get("runtime_config")
    runtime = load_runtime_config(runtime) if isinstance(runtime, (str, Path)) else validate_config(runtime)
    options = phase1_options(runtime)
    if (not options or runtime["algorithm"] != "hgr"
            or options["predictor_mode"] != "zero" or options["lambda"] != 0):
        raise ValueError("Phase2 v1 evaluates the existing Phase1 zero predictor HGR only")
    value["runtime_config"] = runtime
    return value


@dataclass
class FrozenSnapshotSource:
    """One shared theta, two fixed suffix actors, and predeclared cases.

    fresh_main cases: {snapshot_id, scenario}. fixed_boundary cases:
    {snapshot_id, prefix}, where prefix has records/rewards/tau/snapshot/features.
    A bare DecisionSnapshot lacks the prefix scores and is deliberately rejected.
    """
    old_policy: object
    new_policy: object
    cases: list
    scope: str
    source_identity: dict
    preparation_environment_steps: int = 0


def _pack(value):
    if isinstance(value, DecisionSnapshot):
        return {"__decision_snapshot__": dict(vars(value))}
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {str(k): _pack(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    raise ValueError(f"unsupported frozen-source value {type(value).__name__}")


def _unpack(value):
    if isinstance(value, dict):
        if set(value) == {"__decision_snapshot__"}:
            snapshot = DecisionSnapshot(**value["__decision_snapshot__"])
            snapshot.validate()
            return snapshot
        return {k: _unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unpack(v) for v in value]
    return value


def _policy_from_payload(payload, runtime):
    with isolated_global_rng():
        policy = HandoffPolicy(runtime["policy"])
    policy.load_state_dict(payload["state"], strict=True)
    modules = dict(policy.named_modules())
    if set(payload["training_modes"]) != set(modules):
        raise ValueError("frozen policy mode inventory mismatch")
    for name, training in payload["training_modes"].items():
        if not isinstance(training, bool):
            raise ValueError("invalid policy training mode")
        modules[name].training = training
    identity = behavior_identity(policy, runtime_contract(runtime))
    if identity.sha256 != payload["behavior_sha256"]:
        raise ValueError("frozen policy forward attributes/config/state mismatch")
    return policy


def save_snapshot_source(path, source, runtime_config):
    """Explicit manual export of supplied objects, without collecting or training."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    runtime = validate_config(runtime_config)
    require_source_match(source.source_identity, fresh_source_identity(), context="Phase2 source export")
    policies = []
    for policy in (source.old_policy, source.new_policy):
        payload = dict(state=policy.state_dict(),
                       training_modes={k: v.training for k, v in policy.named_modules()},
                       behavior_sha256=behavior_identity(policy, runtime_contract(runtime)).sha256)
        # Fail before writing if actual forward attributes cannot be reconstructed.
        _policy_from_payload(payload, runtime)
        policies.append(payload)
    payload = dict(schema=SOURCE_SCHEMA, source_identity=source.source_identity,
                   runtime_contract_sha256=_digest(runtime_contract(runtime)), scope=source.scope,
                   preparation_environment_steps=source.preparation_environment_steps,
                   old_policy=policies[0], new_policy=policies[1], cases=_pack(source.cases))
    with path.open("xb") as handle:
        torch.save(payload, handle)


def load_snapshot_source(path, runtime):
    # Only our explicit frozen-source artifact, never a Trainer resume checkpoint.
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    expected = {"schema", "source_identity", "runtime_contract_sha256", "scope",
                "preparation_environment_steps", "old_policy", "new_policy", "cases"}
    if not isinstance(payload, dict) or set(payload) != expected or payload["schema"] != SOURCE_SCHEMA:
        raise ValueError("requires hgr.phase2.frozen_source.v1; training checkpoints/bare snapshots are unsupported")
    require_source_match(payload["source_identity"], fresh_source_identity(), context="Phase2 frozen source")
    if payload["runtime_contract_sha256"] != _digest(runtime_contract(runtime)):
        raise ValueError("frozen source runtime contract mismatch")
    return FrozenSnapshotSource(_policy_from_payload(payload["old_policy"], runtime),
                                _policy_from_payload(payload["new_policy"], runtime),
                                _unpack(payload["cases"]), payload["scope"], payload["source_identity"],
                                payload["preparation_environment_steps"])


class VectorMoments:
    """Float64 Welford accumulation, no stack of hundreds of large gradients."""
    def __init__(self):
        self.count = 0
        self.mean = self.m2 = None

    def add(self, vector):
        vector = np.asarray(vector, dtype=np.float64)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("invalid gradient vector")
        if not self.count:
            self.mean = np.zeros_like(vector); self.m2 = np.zeros_like(vector)
        if vector.shape != self.mean.shape:
            raise ValueError("gradient dimension changed")
        self.count += 1
        delta = vector - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (vector - self.mean)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.m2).all():
            raise ValueError("nonfinite gradient moments")

    @property
    def variance(self):
        if self.count < 2:
            raise ValueError("variance requires at least two independent repeats")
        return self.m2 / (self.count - 1)


def estimator_vectors(policy, trajectory, gamma, *, draws=None, bypass=None):
    """Read-only differentiation of the EXISTING production estimator."""
    method = "stochastic_direct_mc" if draws is None else "hgr"
    with torch.enable_grad():
        losses = prefix_losses(policy, [trajectory], [0.], [] if draws is None else draws,
                               gamma=gamma, method=method, bypass=bypass)
        params = list(policy.theta_minus.parameters())
        g0 = -gradient_vector(losses["old"], params).cpu().numpy().astype(np.float64)
        delta = -(gradient_vector(losses["prediction"], params)
                  + gradient_vector(losses["correction"], params)).cpu().numpy().astype(np.float64)
    result = dict(g0=g0, g_delta=delta, g_full=g0+delta)
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError("nonfinite score estimator")
    return result


class RuntimeBackend:
    """Serial adapter to the unchanged Phase1 simulator."""
    def collect(self, runtime, case, policy, *, seed, stream_id):
        return collect_trajectory(runtime, case["scenario"], policy, seed=seed,
                                  episode_id=seed % (2**31-1))

    def branch(self, snapshot, policy, noise, *, keep_records=False):
        return continue_branch(snapshot, policy, pair_noise=noise, keep_records=keep_records)


def _validate_trajectory(trajectory, policy, contract, horizon):
    records, rewards, tau = trajectory["records"], trajectory["rewards"], trajectory["tau"]
    if (trajectory.get("trajectory_complete") is not True or trajectory.get("consumed", False)
            or not 0 < len(records) <= horizon or len(rewards) != len(records)
            or [r["t"] for r in records] != list(range(len(records)))
            or not all(records[-1]["dones"]) or any(all(r["dones"]) for r in records[:-1])
            or not np.isfinite(np.asarray(rewards)).all()):
        raise ValueError("requires a complete finite unconsumed trajectory")
    if tau is not None and (isinstance(tau, bool) or not isinstance(tau, int) or not 0 <= tau < len(records)):
        raise ValueError("invalid action-before boundary")
    if any(bool(r["suffix"]) != (tau is not None and i >= tau) for i, r in enumerate(records)):
        raise ValueError("prefix/suffix partition mismatch")
    if (trajectory.get("theta_hash") != weights_hash(policy.theta_minus)
            or trajectory.get("theta_behavior_sha256") != behavior_identity(policy.theta_minus, contract).sha256
            or trajectory.get("suffix_behavior_sha256") != behavior_identity(policy.phi, contract).sha256):
        raise ValueError("trajectory was sampled under another frozen behavior")
    if tau is not None and (trajectory["snapshot"] is None or trajectory["snapshot"].step != tau):
        raise ValueError("trajectory snapshot/tau mismatch")


class Evaluation:
    def __init__(self, config, source, output, backend):
        self.config, self.runtime, self.source = config, config["runtime_config"], source
        self.output, self.backend = output, backend
        self.contract = runtime_contract(self.runtime)
        self.old, self.new = copy.deepcopy(source.old_policy), copy.deepcopy(source.new_policy)
        for policy in (self.old, self.new):
            policy.theta_minus.requires_grad_(True)
            policy.phi.requires_grad_(False)
        self.identities = [behavior_identity(p, self.contract) for p in (self.old, self.new)]
        if not identical_behavior(behavior_identity(self.old.theta_minus, self.contract),
                                  behavior_identity(self.new.theta_minus, self.contract)):
            raise ValueError("phi0/phi1 must share exactly the same theta behavior")
        self.equal_suffix = identical_behavior(behavior_identity(self.old.phi, self.contract),
                                               behavior_identity(self.new.phi, self.contract))
        self.phi_hashes = dict(phi0_hash=weights_hash(self.old.phi), phi1_hash=weights_hash(self.new.phi))
        self.dimension = sum(p.numel() for p in self.new.theta_minus.parameters())
        self.costs = dict(source_preparation_steps=source.preparation_environment_steps,
                          reference_steps=0, direct_new_mc_steps=0, old_mc_steps=0, hgr_main_steps=0,
                          hgr_correction_old_steps=0, hgr_correction_new_steps=0)
        self.inflight = None

    def guard(self):
        for policy, identity in zip((self.old, self.new), self.identities):
            if not identical_behavior(identity, behavior_identity(policy, self.contract)):
                raise RuntimeError("frozen policy parameters/buffers/forward attributes changed")

    def stream(self, case_index, purpose, repeat):
        return named_seed(self.config["seed"], case_index+1, "phase2/"+self.config["scope"]+"/"+purpose, repeat)

    def noise(self, case_index, purpose, index, role, mode=None):
        return PairNoise.make(self.config["seed"], case_index+1, "phase2/"+self.config["scope"]+"/"+purpose,
                              index, role, mode or self.runtime["phase1"]["pairing"])

    def branch(self, case_index, snapshot, policy, noise, cost_field, *, records=False):
        self.inflight = dict(case_index=case_index, operation="branch", **noise.public())
        result = self.backend.branch(snapshot, policy, noise, keep_records=records)
        _positive_integer(result["steps"], "actual branch steps")
        self.costs[cost_field] += result["steps"]
        if (result["terminal_step"] != snapshot.step+result["steps"]
                or result["terminal_step"] > snapshot.max_steps or not np.isfinite(result["G_plus"])):
            raise ValueError("branch return/clock contract mismatch")
        self.guard(); self.inflight = None
        return result

    def main(self, case_index, case, policy, method, repeat, cost_field):
        stream_id, seed = self.stream(case_index, method+"/main", repeat)
        self.inflight = dict(case_index=case_index, operation="full_trajectory", random_stream_id=stream_id)
        if self.config["scope"] == "fresh_main":
            trajectory = self.backend.collect(self.runtime, copy.deepcopy(case), policy, seed=seed, stream_id=stream_id)
            self.costs[cost_field] += len(trajectory["records"])
        else:
            prefix = case["prefix"]
            noise = self.noise(case_index, method+"/main", repeat, "new", "independent")
            stream_id = noise.public()["trace_id"]
            suffix = self.branch(case_index, prefix["snapshot"], policy, noise, cost_field, records=True)
            trajectory = dict(prefix, records=prefix["records"]+suffix["records"],
                              rewards=list(prefix["rewards"])+[r["team_reward"] for r in suffix["records"]],
                              trajectory_complete=True, consumed=False,
                              suffix_behavior_sha256=behavior_identity(policy.phi, self.contract).sha256)
        _validate_trajectory(trajectory, policy, self.contract, self.runtime["max_steps"])
        self.guard(); self.inflight = None
        return trajectory, stream_id

    def sample(self, case_index, case, method, repeat):
        before = sum(self.costs.values())
        policy = self.new if method in ("direct_new_mc", "reference") else self.old
        cost_field = "reference_steps" if method == "reference" else "hgr_main_steps" if method == "hgr" else method+"_steps"
        trajectory, stream_id = self.main(case_index, case, policy, method, repeat, cost_field)
        pair_id, _ = self.stream(case_index, "reference_repeat" if method=="reference" else "comparison_repeat", repeat)
        draws, queries, proof = [], [], None
        if method == "hgr":
            if self.equal_suffix and self.runtime["phase1"]["zero_update_bypass"]:
                proof = NoUpdateProof.create(self.old.phi, self.new.phi, self.contract)
            else:
                k = self.runtime["correction_draws_per_cycle"]
                # N=1 per replicate: Phase1 fixed-K with replacement, q=1.
                for draw in range(k):
                    label = 0.
                    old_noise = self.noise(case_index, f"hgr/correction/repeat/{repeat}", draw, "old")
                    new_noise = self.noise(case_index, f"hgr/correction/repeat/{repeat}", draw, "new")
                    query = dict(draw=draw, index=0, q=1., pair_id=old_noise.pair_id, label=0.,
                                 no_handoff=trajectory["tau"] is None)
                    if trajectory["tau"] is not None:
                        new_result = self.branch(case_index, trajectory["snapshot"], self.new, new_noise,
                                                 "hgr_correction_new_steps")
                        old_result = self.branch(case_index, trajectory["snapshot"], self.old, old_noise,
                                                 "hgr_correction_old_steps")
                        label = new_result["G_plus"]-old_result["G_plus"]
                        if not np.isfinite(label):
                            raise ValueError("invalid formal delta label")
                        query.update(label=label, old=old_result, new=new_result)
                    draws.append((0, 1., label)); queries.append(query)
        vectors = estimator_vectors(self.new if method == "hgr" else policy, trajectory,
                                    self.runtime["rl"]["gamma"],
                                    draws=draws if method == "hgr" else None, bypass=proof)
        self.guard()
        snapshot = trajectory.get("snapshot")
        row = dict(snapshot_id=case["snapshot_id"], tau=trajectory["tau"],
                   snapshot_hash=None if snapshot is None else snapshot.sha256,
                   method=method, repeat_id=repeat, pair_id=pair_id, **self.phi_hashes,
                   environment_steps=sum(self.costs.values())-before,
                   gradient_norm=float(np.linalg.norm(vectors["g_full"])), random_stream_id=stream_id,
                   random_source_revision=STREAM_REVISION,
                   actual_trajectory_length=len(trajectory["records"]),
                   K_actual=len(draws), K_requested=self.runtime["correction_draws_per_cycle"] if method=="hgr" else 0,
                   skip_reason="identical_suffix_behavior" if proof else None, correction_queries=queries)
        return vectors, row

    def arrays(self, name, **values):
        if any(not np.isfinite(np.asarray(v)).all() for v in values.values()):
            raise ValueError("nonfinite artifact array")
        if self.config["vector_storage"] == "inline":
            return {key: dict(values=np.asarray(value).tolist(), shape=list(value.shape), dtype=str(value.dtype))
                    for key, value in values.items()}
        relative = Path("vectors")/(name+".npz")
        path = self.output/relative; path.parent.mkdir(exist_ok=True)
        with path.open("xb") as handle:
            np.savez_compressed(handle, **values)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        return {key: dict(path=relative.as_posix(), key=key, shape=list(value.shape),
                         dtype=str(value.dtype), sha256=sha) for key, value in values.items()}


def _validate_source(source, config, synthetic=False):
    if not isinstance(source, FrozenSnapshotSource) or source.scope != config["scope"]:
        raise ValueError("snapshot source scope mismatch")
    require_source_match(source.source_identity, fresh_source_identity(), context="Phase2 frozen source")
    _positive_integer(source.preparation_environment_steps, "preparation_environment_steps", 0)
    if len(source.cases) != config["snapshot_count"]:
        raise ValueError("source must contain exactly the predeclared snapshot_count; no outcome selection")
    if not synthetic and any(not isinstance(p, HandoffPolicy) for p in (source.old_policy, source.new_policy)):
        raise ValueError("production source requires the existing HandoffPolicy architecture")
    names = [case.get("snapshot_id") for case in source.cases]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("missing/duplicate snapshot_id")
    runtime = config["runtime_config"]; contract = runtime_contract(runtime)
    prefix_steps = 0
    for case in source.cases:
        if source.scope == "fresh_main":
            scenario = case.get("scenario", {})
            if (scenario.get("scenario_profile") != runtime["profile"]
                    or scenario.get("max_steps") != runtime["max_steps"]
                    or not scenario.get("scenario_id") or "scenario_seed" not in scenario):
                raise ValueError("fresh_main requires a predeclared full initial scenario, not a handoff snapshot")
            canonical(scenario)
        else:
            prefix = case.get("prefix", {})
            tau = prefix.get("tau")
            if (isinstance(tau, bool) or not isinstance(tau, int) or not 0 <= tau < runtime["max_steps"]
                    or len(prefix.get("records", [])) != tau or len(prefix.get("rewards", [])) != tau
                    or prefix.get("snapshot") is None or prefix["snapshot"].step != tau
                    or prefix["snapshot"].max_steps != runtime["max_steps"]
                    or prefix.get("consumed", False) or prefix.get("trajectory_complete") is not False):
                raise ValueError("fixed_boundary requires a live snapshot AND its entire unconsumed prefix")
            if ([r["t"] for r in prefix["records"]] != list(range(tau))
                    or any(r["suffix"] or all(r["dones"]) for r in prefix["records"])
                    or not np.isfinite(np.asarray(prefix["rewards"])).all()
                    or prefix.get("theta_hash") != weights_hash(source.old_policy.theta_minus)
                    or prefix.get("theta_behavior_sha256") != behavior_identity(source.old_policy.theta_minus, contract).sha256):
                raise ValueError("invalid/stale recorded prefix")
            if not synthetic:
                if not isinstance(prefix["snapshot"], DecisionSnapshot):
                    raise ValueError("requires a real DecisionSnapshot")
                prefix["snapshot"].validate()
            prefix_steps += tau
    if source.preparation_environment_steps < prefix_steps:
        raise ValueError("source preparation cost omits the captured prefix steps")


def run_phase2_gradient_efficiency(config, snapshot_source=None, num_snapshots=None,
                                   num_repeats=None, reference_budget=None, *, _backend=None):
    """Evaluate only. reference_budget counts TOTAL Direct-New reference rollouts."""
    config = resolve_config(config, num_snapshots, num_repeats, reference_budget)
    runtime = config["runtime_config"]
    source = snapshot_source if snapshot_source is not None else config.get("snapshot_source")
    if source is None:
        raise ValueError("supply a frozen snapshot_source artifact or FrozenSnapshotSource; no policies are invented")
    if isinstance(source, (str, Path)):
        source = load_snapshot_source(source, runtime)
    _validate_source(source, config, synthetic=_backend is not None)
    source = replace(source, cases=copy.deepcopy(source.cases))
    output = validated_output(runtime, config["output_dir"])
    config["output_dir"] = str(output)
    if isinstance(config.get("snapshot_source"), Path):
        config["snapshot_source"] = str(config["snapshot_source"])
    # Validate all policy identities before creating output files.
    evaluation = Evaluation(config, source, output, _backend or RuntimeBackend())
    output.mkdir(parents=True, exist_ok=True)
    n, repeats, budget = config["snapshot_count"], config["repeat_count"], config["reference_rollouts"]
    per_reference = budget//n
    k, horizon = runtime["correction_draws_per_cycle"], runtime["max_steps"]
    manifest = [dict(snapshot_id=c["snapshot_id"], scenario=c["scenario"]) if source.scope=="fresh_main"
                else dict(snapshot_id=c["snapshot_id"], tau=c["prefix"]["tau"],
                          snapshot_hash=c["prefix"]["snapshot"].sha256) for c in source.cases]
    report = dict(schema=SCHEMA, status="RUNNING", scope=source.scope,
                  gradient_target="theta_minus only, phi0/phi1 held fixed",
                  estimand="uniform mixture over predeclared initial scenarios" if source.scope=="fresh_main"
                  else "conditional score-estimator moments given fixed recorded prefixes",
                  source_identity=fresh_source_identity(), checkout=checkout_identity(),
                  config=config, snapshot_manifest=manifest, random_source_revision=STREAM_REVISION,
                  frozen_behavior=[identity.public() for identity in evaluation.identities],
                  gradient_parameter_layout=[dict(name=name, shape=list(p.shape), size=p.numel())
                                             for name,p in evaluation.new.theta_minus.named_parameters()],
                  gradient_dimension=evaluation.dimension, gradient_samples_per_method=n*repeats,
                  total_comparison_gradient_samples=3*n*repeats, reference_rollouts=budget,
                  reference_rollouts_per_case=per_reference,
                  worst_case_environment_steps_excluding_source=(budget+n*repeats*(3+2*k))*horizon,
                  reference=[], samples=[], statistics={}, costs=evaluation.costs,
                  policy_updates=0, performance_claims_supported=False, formal_experiment=False)

    def persist():
        report["total_environment_steps"] = sum(evaluation.costs.values())
        write_json(output/"phase2_results.json", report)

    persist()
    try:
        with isolated_global_rng():
            reference_stats = []
            for index, case in enumerate(source.cases):
                stats = VectorMoments(); metadata = []
                for repeat in range(per_reference):
                    vectors, row = evaluation.sample(index, case, "reference", repeat)
                    stats.add(vectors["g_full"]); metadata.append(row)
                reference_stats.append(stats)
                report["reference"].append(dict(snapshot_id=case["snapshot_id"], rollouts=per_reference,
                                               gradient_norm=float(np.linalg.norm(stats.mean)), samples=metadata))
                persist()
            g_ref = sum(s.mean for s in reference_stats)/n
            ref_variance = sum(s.variance/per_reference for s in reference_stats)/(n*n)
            ref_arrays = evaluation.arrays("reference", g_ref=g_ref, reference_mean_variance=ref_variance,
                                           case_reference_means=np.stack([s.mean for s in reference_stats]))
            report.update(reference_gradient_norm=float(np.linalg.norm(g_ref)),
                          reference_gradient=ref_arrays["g_ref"], reference_arrays=ref_arrays,
                          reference_mean_variance_trace=float(ref_variance.sum()),
                          reference_note="Independent finite MC estimate, not exact truth. MSE includes reference uncertainty.")
            statistics = {m: VectorMoments() for m in METHODS}
            strata = [{m: VectorMoments() for m in METHODS} for _ in range(n)]
            for index, case in enumerate(source.cases):
                for repeat in range(repeats):
                    for method in METHODS:
                        vectors, row = evaluation.sample(index, case, method, repeat)
                        full = vectors["g_full"]
                        row["gradient_mse"] = float(np.dot(full-g_ref, full-g_ref))
                        local_ref = reference_stats[index].mean
                        row["case_reference_mse"] = float(np.dot(full-local_ref, full-local_ref))
                        row["cost_normalized_error"] = row["gradient_mse"]*row["environment_steps"]
                        if not np.isfinite([row["gradient_mse"], row["cost_normalized_error"]]).all():
                            raise ValueError("nonfinite error metric")
                        arrays = evaluation.arrays(f"{index:03d}_{repeat:05d}_{method}",
                                                   **(vectors if method=="hgr" else {"gradient_vector":full}))
                        row.update(arrays)
                        if method=="hgr":
                            row["gradient_vector"] = arrays["g_full"]
                        statistics[method].add(full); strata[index][method].add(full)
                        report["samples"].append(row)
                        persist()
            for method, stats in statistics.items():
                rows = [row for row in report["samples"] if row["method"]==method]
                per_case = [s[method] for s in strata]
                arrays = evaluation.arrays("statistics_"+method, mean=stats.mean,
                                           per_dimension_variance=stats.variance,
                                           within_case_variance=np.stack([s.variance for s in per_case]),
                                           case_means=np.stack([s.mean for s in per_case]))
                report["statistics"][method] = dict(
                    samples=stats.count, variance_ddof=1, variance_scope="pooled descriptive; includes case heterogeneity",
                    mean_gradient_norm=float(np.linalg.norm(stats.mean)), variance_trace=float(stats.variance.sum()),
                    mean_gradient_mse=float(np.mean([r["gradient_mse"] for r in rows])),
                    mean_cost_normalized_error=float(np.mean([r["cost_normalized_error"] for r in rows])),
                    total_environment_steps=sum(r["environment_steps"] for r in rows), **arrays)
            evaluation.guard()
            require_source_match(report["source_identity"], fresh_source_identity(), context="Phase2 end check")
            hgr_rows = [r for r in report["samples"] if r["method"]=="hgr"]
            exercised = any(r["tau"] is not None and r["correction_queries"] for r in hgr_rows)
            report.update(status="PASS", frozen_policy_check="PASS",
                          handoff_correction_status="EXERCISED" if exercised else "NOT_EXERCISED",
                          no_handoff_hgr_samples=sum(r["tau"] is None for r in hgr_rows))
            persist()
    except BaseException:
        report.update(status="FAIL", error=traceback.format_exc(), failed_call=evaluation.inflight,
                      failed_call_steps_unknown=evaluation.inflight is not None)
        persist()
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT/"configs/chapter3/hgr_phase2_gradient_efficiency.json"))
    args = parser.parse_args()
    report = run_phase2_gradient_efficiency(load_config(args.config))
    print(json.dumps(dict(status=report["status"], environment_steps=report["total_environment_steps"],
                          results=str(Path(report["config"]["output_dir"])/"phase2_results.json"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
