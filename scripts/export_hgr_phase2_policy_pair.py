"""One-time, audited parameter import from the migrated 3090 HGR cycle 4.

Run from the repository with ``python -m scripts.export_hgr_phase2_policy_pair``.
This is not checkpoint evaluation/resume: no Trainer, optimizer, predictor,
simulator, checkpoint RNG or saved trajectory is restored. Historical source
identity is retained in a separate audit; current policy objects receive exact
learned tensors. Existing source gates and retained files remain unchanged.

The reviewed input and destination source hashes are deliberately pinned. This
is not a general cross-version loader; another source requires another review.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import torch

from chapter3_bser.experiments.hgr import build_phase2_source as builder
from chapter3_bser.experiments.hgr.phase2_gradient_efficiency import _policy_from_payload
from chapter3_bser.experiments.hgr.provenance import (
    checkout_identity, fresh_source_identity, require_source_match, validate_source_identity,
)
from chapter3_bser.experiments.hgr.train import COST_FIELDS, digest
from chapter3_bser.models.hgr.phase1 import behavior_identity, canonical, isolated_global_rng, runtime_contract
from chapter3_bser.models.hgr.policy import HandoffPolicy, weights_hash

ROOT = builder.ROOT
HISTORICAL_COMMIT = "da4a8a64cba26b2adf7d248fd8486e517422628c"
HISTORICAL_SOURCE = "767b1232abb19e3d4a1d9d2d4c173d0cc3417b0b17a893e23514e78d3db2608c"
REVIEWED_CURRENT_SOURCE = "100f369e2c1958a065905e379bd13ca63cdc7f76000352f2ee8fa267ff9aabca"
INPUTS = (
    ("hgr_main_000048_cycle_000003.pt", "650ef72e866458e4184dc7146a1fdc4491a52f936983c9e239c9e4e7f10dae55"),
    ("hgr_main_000064_cycle_000004.pt", "bd3085fd9839f2a5ae76f91626cd46a5ab25c6862530d44e7ae57d1871b9897c"),
)
DEFAULT_INPUT = ROOT.parent / "3090结果/collision_terminal/HGR-train100_seed2729_v1/checkpoints"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_checkpoint(path, expected_sha):
    raw = Path(path).read_bytes()
    if sha(raw) != expected_sha:
        raise ValueError("checkpoint bytes differ from the reviewed 3090 input")
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)


def verify_sources(historical):
    validate_source_identity(historical)
    if historical["sha256"] != HISTORICAL_SOURCE:
        raise ValueError("unreviewed historical source")
    archive = subprocess.run(
        ["git", "-c", "safe.directory=" + ROOT.as_posix(), "archive", HISTORICAL_COMMIT,
         "core", "chapter3_bser"], cwd=ROOT, check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        inventory = {m.name: sha(tree.extractfile(m).read()) for m in tree.getmembers()
                     if m.isfile() and m.name.endswith(".py")}
    if inventory != historical["files"]:
        raise ValueError("historical checkpoint source is not the audited Git tree")
    current = fresh_source_identity()
    if current["sha256"] != REVIEWED_CURRENT_SOURCE:
        raise ValueError("current production source changed; a new compatibility review is required")
    return current


def reconstruct_pair(before, after, runtime):
    """Recover the recorded within-cycle pair BEFORE the prefix update.

    The after checkpoint holds theta_after, which is deliberately NOT exported.
    Its phi is unchanged by the prefix update, as checked in the audited Trainer.
    The cycle's three hashes prove the shared theta, phi0 and phi1 association.
    """
    for checkpoint in (before, after):
        if (checkpoint.get("schema") != "hgr.complete_cycle.v1"
                or checkpoint.get("architecture_version") != "hgr.isolated_tanh_gaussian.v1"
                or checkpoint.get("cycle_complete") is not True
                or checkpoint.get("prefixes_valid") is not False):
            raise ValueError("requires a complete legacy HGR cycle checkpoint")
        if checkpoint["config_hash"] != digest(checkpoint["config"]):
            raise ValueError("checkpoint config hash mismatch")
        if checkpoint["config"]["policy"] != runtime["policy"]:
            raise ValueError("policy architecture/forward configuration mismatch")
        source_contract = runtime_contract(checkpoint["config"])
        destination_contract = runtime_contract(runtime)
        differences = {k for k in source_contract.keys() | destination_contract.keys()
                       if source_contract.get(k) != destination_contract.get(k)}
        if (differences != {"checkpoint_schema", "phase1_bypass_authorized"}
                or source_contract["checkpoint_schema"] != "hgr.complete_cycle.v1"
                or destination_contract["checkpoint_schema"] != "hgr.complete_cycle.phase1.v1"
                or source_contract["phase1_bypass_authorized"] is not False
                or destination_contract["phase1_bypass_authorized"] is not True):
            raise ValueError("runtime contract differs beyond the reviewed Phase1 metadata")
        if not checkpoint["cycles"] or checkpoint["cycles"][-1]["complete"] is not True:
            raise ValueError("missing completed cycle evidence")
    require_source_match(before["source_identity"], after["source_identity"], context="historical pair")
    if before["config"] != after["config"] or after["cycles"][:-1] != before["cycles"]:
        raise ValueError("checkpoints do not have the same uninterrupted history/config")
    row = after["cycles"][-1]
    if (after["cycle"] != before["cycle"] + 1 or row["cycle"] != after["cycle"]
            or after["completed_main"] != before["completed_main"] + row["N"]
            or row["completed_main_trajectories"] != after["completed_main"]):
        raise ValueError("checkpoints are not adjacent completed cycles")
    with isolated_global_rng():
        old = HandoffPolicy(runtime["policy"])
        final = HandoffPolicy(runtime["policy"])
        old.load_state_dict(before["policy"], strict=True)
        final.load_state_dict(after["policy"], strict=True)
        for policy, checkpoint in ((old, before), (final, after)):
            if any(not bool(torch.isfinite(v).all()) for v in policy.state_dict().values()):
                raise ValueError("nonfinite learned weights")
            if (weights_hash(policy.theta_minus) != checkpoint["theta_hash"]
                    or weights_hash(policy.phi) != checkpoint["phi_hash"]):
                raise ValueError("stored weight hash mismatch")
        if (row["theta_hash"] != before["theta_hash"]
                or row["theta_after_hash"] != after["theta_hash"]
                or row["phi0_hash"] != before["phi_hash"]
                or row["phi1_hash"] != after["phi_hash"]):
            raise ValueError("cycle evidence does not prove the requested within-cycle pair")
        new = copy.deepcopy(old)
        new.phi.load_state_dict(final.phi.state_dict(), strict=True)
    identities = builder.validate_policy_pair(old, new, runtime)
    # All modes are the historical constructor defaults. The pinned Trainer
    # never changes modes, and the actor contains no Dropout/BatchNorm.
    contract = runtime_contract(runtime)
    payloads = []
    for policy in (old, new):
        payload = dict(state={k: v.detach().clone() for k, v in policy.state_dict().items()},
                       training_modes={k: v.training for k, v in policy.named_modules()},
                       behavior_sha256=behavior_identity(policy, contract).sha256)
        _policy_from_payload(payload, runtime)  # current strict factory, no fallback
        payloads.append(payload)
    delta = torch.cat([(new.phi.state_dict()[k].double() - v.double()).reshape(-1)
                       for k, v in old.phi.state_dict().items()])
    with torch.no_grad():
        observations = torch.linspace(-1, 1, 32 * 28, dtype=torch.float32).reshape(32, 28)
        mu0, std0 = old.phi.distribution_parameters(observations)
        mu1, std1 = new.phi.distribution_parameters(observations)
    diagnostic = dict(phi_delta_l2=float(delta.norm()), phi_delta_max=float(delta.abs().max()),
                      changed_elements=int((delta != 0).sum()), total_elements=delta.numel(),
                      probe="float32 linspace(-1,1,896).reshape(32,28); synthetic, not environment states",
                      probe_mean_max_change=float((mu1 - mu0).abs().max()),
                      probe_std_max_change=float((std1 - std0).abs().max()),
                      probe_action_max_change=float((mu1.tanh() - mu0.tanh()).abs().max()))
    # Include actual historical preparation, not just this export's zero steps.
    cost = sum(after["costs"][k] for k in COST_FIELDS)
    if type(cost) is not int or cost <= 0:
        raise ValueError("missing historical environment cost")
    return payloads, identities, diagnostic, cost


def export_pair(checkpoint_dir=DEFAULT_INPUT, builder_config=builder.DEFAULT_CONFIG):
    config, runtime = builder._resolve_config(builder.load_config(builder_config))
    output = config["policy_source_path"]
    if output is None:
        raise ValueError("policy_source_path must be configured")
    audit_path = output.with_suffix(".audit.json")
    for path in (output, audit_path):
        if path.exists():
            raise FileExistsError(f"retained file will not be overwritten: {path}")
    checkpoints, sources = [], []
    for name, checksum in INPUTS:
        path = Path(checkpoint_dir).resolve() / name
        checkpoints.append(read_checkpoint(path, checksum))
        sources.append(dict(path=str(path), sha256=checksum, size_bytes=path.stat().st_size))
    before, after = checkpoints
    current = verify_sources(before["source_identity"])
    policies, identities, diagnostic, cost = reconstruct_pair(before, after, runtime)
    report = dict(
        schema="hgr.phase2.reviewed_parameter_import.v1", status="POLICY_PAIR_READY",
        historical_commit=HISTORICAL_COMMIT, historical_source_identity=before["source_identity"],
        destination_source_identity=current, destination_checkout=checkout_identity(),
        exporter_sha256=sha(Path(__file__).read_bytes()), inputs=sources,
        historical_cycle=after["cycle"], old_completed_main=before["completed_main"],
        new_completed_main=after["completed_main"], shared_theta_hash=before["theta_hash"],
        phi0_hash=before["phi_hash"], phi1_hash=after["phi_hash"],
        discarded_post_update_theta_hash=after["theta_hash"], **identities,
        selection="Latest saved adjacent cycle with a nonzero float32 distribution difference on the fixed probe; no rollout or efficiency outcomes used. Cycle 7 probe change was zero; cycles 5/6 phi were identical.",
        compatibility="Explicit import of learned tensors into current policy objects, not a historical checkpoint resume/evaluation. Historical inventory exactly matches the pinned Git tree. Runtime contracts differ only in checkpoint_schema and Phase1 bypass authorization. Source gates remain strict.",
        mode_provenance="Constructor training=True; pinned historical Trainer never calls train/eval; actor has no Dropout/BatchNorm.",
        diagnostic=diagnostic, limitation="Very small policy difference. Probe is not an on-policy effect estimate or evidence of gradient efficiency. No claim of task performance or statistical power.",
        preparation_environment_steps=cost, export_environment_steps=0, parameter_updates=0,
        trainer_resumed=False, frozen_source_generated=False, phase2_experiment="NOT_RUN",
        policy_pair_output=str(output))
    audit_bytes = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    origin = dict(kind="real_policy_export",
                  description=json.dumps(dict(operation="reviewed_within_cycle_parameter_import",
                       audit_path=str(audit_path), audit_sha256=sha(audit_bytes), inputs=sources,
                       historical_source_sha256=HISTORICAL_SOURCE, historical_cycle=after["cycle"]), ensure_ascii=False),
                  theta_id=f"{sources[0]['sha256']}:theta_minus:{before['theta_hash']}",
                  phi0_id=f"{sources[0]['sha256']}:phi:{before['phi_hash']}",
                  phi1_id=f"{sources[1]['sha256']}:phi:{after['phi_hash']}")
    payload = dict(schema=builder.POLICY_PAIR_SCHEMA, source_identity=current,
                   runtime_contract_sha256=sha(canonical(runtime_contract(runtime))),
                   old_policy=policies[0], new_policy=policies[1],
                   preparation_environment_steps=cost, origin=origin)
    require_source_match(current, fresh_source_identity(), context="export end check")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation, no overwrites, no failure cleanup of retained evidence.
    with audit_path.open("xb") as stream:
        stream.write(audit_bytes)
    with output.open("xb") as stream:
        torch.save(payload, stream)
    _, loaded_audit = builder.load_policy_pair(output, runtime)
    return dict(status="POLICY_PAIR_READY", output=str(output), sha256=loaded_audit["sha256"],
                audit=str(audit_path), preparation_environment_steps=cost,
                export_environment_steps=0, diagnostic=diagnostic, frozen_source_generated=False,
                phase2_experiment="NOT_RUN")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--builder-config", type=Path, default=builder.DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    print(json.dumps(export_pair(args.checkpoint_dir, args.builder_config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
