"""Synthetic import invariants; never train, restore a Trainer, or run a scene."""
import copy
import json
from unittest.mock import patch

import pytest
import torch

from scripts import export_hgr_phase2_policy_pair as export
from chapter3_bser.experiments.hgr import train


@pytest.fixture
def pair():
    torch.set_num_threads(1)
    config, runtime = export.builder._resolve_config(export.builder.load_config())
    runtime["policy"]["actor"].update(hidden_dim=8, expert_hidden_dim=8)
    legacy = copy.deepcopy(runtime)
    legacy.pop("phase1")
    legacy["checkpoint_schema"] = "hgr.complete_cycle.v1"
    with export.isolated_global_rng():
        old = export.HandoffPolicy(legacy["policy"])
        new = copy.deepcopy(old)
        with torch.no_grad():
            new.phi.log_std.add_(.01)
            new.theta_minus[0].log_std.add_(.02)
    history = [dict(cycle=3, complete=True)]
    row = dict(cycle=4, complete=True, N=16, completed_main_trajectories=64,
               theta_hash=export.weights_hash(old.theta_minus),
               theta_after_hash=export.weights_hash(new.theta_minus),
               phi0_hash=export.weights_hash(old.phi), phi1_hash=export.weights_hash(new.phi))
    result = []
    for policy, cycle, completed, cycles in ((old, 3, 48, history), (new, 4, 64, history + [row])):
        result.append(dict(schema="hgr.complete_cycle.v1",
            architecture_version="hgr.isolated_tanh_gaussian.v1", cycle_complete=True,
            prefixes_valid=False, config=copy.deepcopy(legacy), config_hash=export.digest(legacy),
            source_identity=export.fresh_source_identity(), policy=copy.deepcopy(policy.state_dict()),
            theta_hash=export.weights_hash(policy.theta_minus), phi_hash=export.weights_hash(policy.phi),
            cycle=cycle, completed_main=completed, cycles=copy.deepcopy(cycles),
            costs={key: 10 for key in export.COST_FIELDS}))
    return result[0], result[1], runtime


def test_reconstructs_recorded_pair_not_post_update_theta_and_preserves_rng(pair):
    before, after, runtime = pair
    states = [copy.deepcopy(cp["policy"]) for cp in (before, after)]
    rng = torch.get_rng_state().clone()
    with patch.object(train.Trainer, "__init__", side_effect=AssertionError("no Trainer")), \
            patch.object(train, "update_suffix", side_effect=AssertionError("no updates")):
        policies, ids, diagnostic, cost = export.reconstruct_pair(before, after, runtime)
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in before["policy"].items():
        assert torch.equal(policies[0]["state"][key], value)
        expected = after["policy"][key] if key.startswith("phi.") else value
        assert torch.equal(policies[1]["state"][key], expected)
        assert policies[0]["state"][key].data_ptr() != value.data_ptr()
        assert torch.equal(before["policy"][key], states[0][key])
        assert torch.equal(after["policy"][key], states[1][key])
    assert ids["phi0_behavior_sha256"] != ids["phi1_behavior_sha256"]
    assert diagnostic["probe_std_max_change"] > 0
    assert cost == 10 * len(export.COST_FIELDS)


@pytest.mark.parametrize("failure", ["lineage", "theta", "phi0", "phi1", "tensor", "nonfinite", "config"])
def test_rejects_inconsistent_historical_evidence(pair, failure):
    before, after, runtime = pair
    row = after["cycles"][-1]
    if failure == "lineage":
        after["cycle"] += 1
    elif failure in ("theta", "phi0", "phi1"):
        row[{"theta": "theta_hash", "phi0": "phi0_hash", "phi1": "phi1_hash"}[failure]] = "0" * 64
    elif failure in ("tensor", "nonfinite"):
        after["policy"]["phi.log_std"][0] = float("nan") if failure == "nonfinite" else 1.
    else:
        for cp in (before, after):
            cp["config"]["max_steps"] = 399
            cp["config_hash"] = export.digest(cp["config"])
    with pytest.raises(ValueError):
        export.reconstruct_pair(before, after, runtime)


def test_rejects_identical_phi_even_with_changed_theta(pair):
    before, after, runtime = pair
    for key, value in before["policy"].items():
        if key.startswith("phi."):
            after["policy"][key] = value.clone()
    after["phi_hash"] = after["cycles"][-1]["phi1_hash"] = before["phi_hash"]
    with pytest.raises(export.builder.PolicySourceUnavailable, match="different behavior"):
        export.reconstruct_pair(before, after, runtime)


def test_bad_file_hash_rejected_before_deserialization(tmp_path):
    path = tmp_path / "not_a_checkpoint.pt"
    path.write_bytes(b"bad")
    with patch.object(torch, "load", side_effect=AssertionError("must reject bytes first")):
        with pytest.raises(ValueError, match="reviewed 3090"):
            export.read_checkpoint(path, "0" * 64)


def test_unreviewed_historical_source_rejected_before_git():
    with patch.object(export.subprocess, "run", side_effect=AssertionError("unreviewed source")):
        with pytest.raises(ValueError, match="historical source"):
            export.verify_sources(export.fresh_source_identity())


def test_policy_only_roundtrip_and_exclusive_output(pair, tmp_path, monkeypatch):
    before, after, runtime = pair
    inputs = []
    for index, cp in enumerate((before, after)):
        path = tmp_path / f"synthetic_{index}.pt"
        torch.save(cp, path)
        inputs.append((path.name, export.sha(path.read_bytes())))
    monkeypatch.setattr(export, "INPUTS", inputs)
    monkeypatch.setattr(export, "verify_sources", lambda _: export.fresh_source_identity())
    config = dict(policy_source_path=tmp_path / "policy_only.pt")
    monkeypatch.setattr(export.builder, "_resolve_config", lambda _: (config, runtime))
    monkeypatch.setattr(export.builder, "build_cases", lambda *a: pytest.fail("no scenarios"))
    result = export.export_pair(tmp_path)
    assert result["status"] == "POLICY_PAIR_READY"
    assert result["export_environment_steps"] == 0
    assert result["phase2_experiment"] == "NOT_RUN"
    source, audit = export.builder.load_policy_pair(config["policy_source_path"], runtime)
    declaration = json.loads(audit["origin"]["description"])
    assert declaration["audit_sha256"] == export.sha((tmp_path / "policy_only.audit.json").read_bytes())
    assert not source.cases
    assert source.preparation_environment_steps == 10 * len(export.COST_FIELDS)
    raw = config["policy_source_path"].read_bytes()
    with pytest.raises(FileExistsError):
        export.export_pair(tmp_path)
    assert config["policy_source_path"].read_bytes() == raw
