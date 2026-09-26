"""Source preparation unit tests; all policy/scene fixtures are synthetic.

No real simulator, training, user checkpoint, or Phase2 experiment is invoked.
"""
import copy
import hashlib
import json
from pathlib import Path
import random
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from chapter3_bser.experiments.hgr import build_phase2_source as builder
from chapter3_bser.experiments.hgr import phase2_gradient_efficiency as phase2
from chapter3_bser.models.hgr.phase1 import behavior_identity, canonical, runtime_contract
from chapter3_bser.models.hgr.policy import HandoffPolicy


@pytest.fixture(autouse=True)
def no_experiments(monkeypatch):
    """Fail the tests if source preparation crosses any execution boundary."""
    from chapter3_bser.experiments.hgr import runtime, train
    guards = [
        (runtime.MissionRuntime, "__init__"), (runtime.MissionRuntime, "restore"),
        (train.Trainer, "__init__"), (torch.optim.SGD, "step"), (torch.optim.Adam, "step"),
        (phase2, "run_phase2_gradient_efficiency"),
    ]
    for target, name in guards:
        monkeypatch.setattr(target, name, Mock(side_effect=AssertionError("experiment/training forbidden")))


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config(tmp_path):
    cfg = builder.load_config()
    evaluation = phase2.load_config(builder.ROOT/"configs/chapter3/hgr_phase2_gradient_efficiency.json")
    runtime = phase2.resolve_config(evaluation)["runtime_config"]
    runtime["policy"]["actor"].update(hidden_dim=8, expert_hidden_dim=8)
    evaluation["runtime_config"] = runtime
    cfg.update(evaluation_config=evaluation, source_output_path=str(tmp_path/"new"/"frozen_phase2_source.pt"),
               policy_source_path=str(tmp_path/"synthetic_policy_pair.pt"))
    return cfg


def policy_payload(policy, runtime):
    return dict(state={k: v.detach().cpu().clone() for k, v in policy.state_dict().items()},
                training_modes={k: v.training for k, v in policy.named_modules()},
                behavior_sha256=behavior_identity(policy, runtime_contract(runtime)).sha256)


def fixture_payload(config, *, same_phi=False, different_theta=False, different_theta_mode=False):
    _, runtime = builder._resolve_config(config)
    # Synthetic deterministic tensors only, never exported as experiment data.
    old = HandoffPolicy(runtime["policy"])
    with torch.no_grad():
        for i, parameter in enumerate(old.parameters()):
            parameter.fill_((i % 5 + 1)*.01)
    new = copy.deepcopy(old)
    if not same_phi:
        with torch.no_grad():
            new.phi.log_std.add_(.01)
    if different_theta:
        with torch.no_grad():
            next(new.theta_minus.parameters()).add_(.01)
    if different_theta_mode:
        new.theta_minus.eval()
    return dict(schema=builder.POLICY_PAIR_SCHEMA, source_identity=builder.fresh_source_identity(),
                runtime_contract_sha256=builder._sha(runtime_contract(runtime)),
                old_policy=policy_payload(old, runtime), new_policy=policy_payload(new, runtime),
                preparation_environment_steps=7,
                origin=dict(kind="real_policy_export", description="Synthetic unit fixture, not experiment data",
                            theta_id="unit_theta", phi0_id="unit_phi0", phi1_id="unit_phi1"))


def write_policy(config, payload=None, **options):
    if payload is None:
        payload = fixture_payload(config, **options)
    torch.save(payload, config["policy_source_path"])
    return payload


def fake_manifest(*, count, generator_seed, split, profiles):
    assert (count, split, profiles) == (1, "validation", [builder.PROFILE])
    obstacles = [dict(center=[10.1234567, 10., 4.], size=[1., 1., 2.]),
                 dict(center=[15., 15., 4.], size=[1., 1., 2.])]
    scenario = dict(
        scenario_id="generator_fixture_0001", scenario_seed=generator_seed,
        scenario_profile=builder.PROFILE, max_steps=400, planner_seed=generator_seed,
        initial_agent_positions=[[1., 1., 2.], [4., 1., 2.], [1., 4., 2.], [4., 4., 2.]],
        initial_executor_wait_point=[7., 2., 2.], target_position=[2.1234567, 2., 4.],
        target_initial_position=[2.1234567, 2., 4.], target_initial_velocity=[.4, 0., 0.],
        target_motion_mode="constant_velocity_reflect_v1", target_state_schema="moving_target_state_v1",
        obstacle_layout_id="custom_aabb_v1", obstacle_knowledge_mode="online_unknown",
        obstacles=obstacles, obstacle_layout_sha256=builder._sha(obstacles),
        target_trajectory_sha256="generator_hash_preserved", target_obstacle_clearance=.2)
    return {builder.PROFILE: {"scenarios": [scenario]}}


@pytest.fixture
def generator(monkeypatch):
    mock = Mock(side_effect=fake_manifest)
    monkeypatch.setattr(builder, "build_scenario_manifests", mock)
    return mock


def test_source_schema_theta_phi_and_consumer_roundtrip(config, generator):
    supplied = write_policy(config)
    original_bytes = Path(config["policy_source_path"]).read_bytes()
    _, runtime = builder._resolve_config(config)
    result = builder.build_phase2_source(config)
    assert result["status"] == "READY"
    assert result["schema"] == "hgr.phase2.frozen_source.v1"
    assert result["environment_steps"] == 0
    assert result["preparation_environment_steps"] == 7
    output = Path(config["source_output_path"])
    payload = torch.load(output, weights_only=True, map_location="cpu")
    assert set(payload) == {"schema", "source_identity", "runtime_contract_sha256", "scope",
                            "preparation_environment_steps", "old_policy", "new_policy", "cases"}
    theta_keys = [k for k in payload["old_policy"]["state"] if k.startswith("theta_minus.")]
    assert theta_keys
    for key in theta_keys:
        assert torch.equal(payload["old_policy"]["state"][key], payload["new_policy"]["state"][key])
    for role in ("old_policy", "new_policy"):
        for key, value in supplied[role]["state"].items():
            assert torch.equal(value, payload[role]["state"][key])
    source = phase2.load_snapshot_source(output, runtime)
    ids = builder.validate_policy_pair(source.old_policy, source.new_policy, runtime)
    assert ids["theta_behavior_sha256"] == result["theta_behavior_sha256"]
    assert ids["phi0_behavior_sha256"] != ids["phi1_behavior_sha256"]
    evaluation = phase2.resolve_config(config["evaluation_config"])
    phase2._validate_source(source, evaluation)
    assert Path(config["policy_source_path"]).read_bytes() == original_bytes
    assert len(source.cases) == 10
    assert generator.call_count == 10
    audit = source.cases[0]["policy_source_audit"]
    assert audit["sha256"] == hashlib.sha256(original_bytes).hexdigest()


@pytest.mark.parametrize("options", [dict(different_theta=True), dict(different_theta_mode=True)])
def test_theta_exact_identity_required_before_scenario_generation(config, generator, options):
    write_policy(config, **options)
    result = builder.build_phase2_source(config)
    assert result["status"] == "SOURCE_NOT_AVAILABLE"
    assert "theta behavior identity mismatch" in result["reason"]
    generator.assert_not_called()
    assert not Path(config["source_output_path"]).exists()


def test_identical_phi_is_rejected_without_perturbation(config, generator):
    write_policy(config, same_phi=True)
    before = Path(config["policy_source_path"]).read_bytes()
    result = builder.build_phase2_source(config)
    assert result["status"] == "SOURCE_NOT_AVAILABLE"
    assert "different behavior identities" in result["reason"]
    assert Path(config["policy_source_path"]).read_bytes() == before
    generator.assert_not_called()


def test_ten_fixed_scenarios_complete_audit_and_rng_isolation(config, generator):
    _, runtime = builder._resolve_config(config)
    saved = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    cases = builder.build_cases(config["scenario_seeds"], runtime)
    assert len(cases) == 10
    assert [c["snapshot_id"] for c in cases] == [f"scene_{i:02d}" for i in range(10)]
    assert [c["scenario"]["scenario_seed"] for c in cases] == config["scenario_seeds"]
    for i, case in enumerate(cases):
        scenario, audit = case["scenario"], case["scenario_audit"]
        assert audit["scenario_id"] == scenario["scenario_id"] == f"scene_{i:02d}"
        assert audit["scenario_seed"] == scenario["scenario_seed"]
        assert audit["runtime_profile"] == builder.PROFILE
        assert audit["max_steps"] == 400
        assert audit["initial_agent_positions"] == scenario["initial_agent_positions"]
        assert audit["obstacle_layout_hash"] == builder._sha(scenario["obstacles"])
        trajectory = audit["target_trajectory"]
        assert trajectory["sample_steps"] == list(range(401))
        assert len(trajectory["positions"]) == len(trajectory["velocities"]) == 401
        assert trajectory["positions"][0] == np.asarray(scenario["target_initial_position"], dtype=np.float32).tolist()
        assert trajectory["velocities"][0] == scenario["target_initial_velocity"]
        assert audit["target_trajectory_hash"] == builder._sha(trajectory)
        assert scenario["target_trajectory_sha256"] == "generator_hash_preserved"
        assert trajectory["kind"] == "deterministic_runtime_projection"
    assert random.getstate() == saved[0]
    np.testing.assert_array_equal(np.random.get_state()[1], saved[1][1])
    assert np.random.get_state()[2:] == saved[1][2:]
    assert torch.equal(torch.get_rng_state(), saved[2])
    assert canonical(builder.build_cases(config["scenario_seeds"], runtime)) == canonical(cases)


@pytest.mark.parametrize("schema", ["hgr.complete_cycle.phase1.v1", "hgr.complete_cycle.v1",
                                    "bser.phase1c.training_state.v2", builder.POLICY_PAIR_SCHEMA])
def test_trainer_checkpoints_forbidden_even_with_renamed_schema(config, generator, monkeypatch, schema):
    payload = fixture_payload(config)
    payload.update(schema=schema, optimizer={"state": {}}, episode=100)
    write_policy(config, payload)
    factory = Mock(side_effect=AssertionError("must reject before policy construction"))
    monkeypatch.setattr(builder, "_policy_from_payload", factory)
    result = builder.build_phase2_source(config)
    assert result["status"] == "SOURCE_NOT_AVAILABLE"
    assert "Trainer checkpoints" in result["reason"]
    factory.assert_not_called()
    generator.assert_not_called()


@pytest.mark.parametrize("path_value", [None, "does_not_exist.pt"])
def test_missing_real_policy_source_never_constructs_random_policy(config, generator, monkeypatch, path_value):
    config["policy_source_path"] = path_value
    factory = Mock(side_effect=AssertionError("no random fallback permitted"))
    monkeypatch.setattr(builder, "_policy_from_payload", factory)
    result = builder.build_phase2_source(config)
    assert result["status"] == "SOURCE_NOT_AVAILABLE"
    assert result["generated"] is False
    assert result["environment_steps"] == 0
    assert not Path(config["source_output_path"]).parent.exists()
    factory.assert_not_called()
    generator.assert_not_called()


@pytest.mark.parametrize("corruption", ["source_identity", "runtime_contract", "tensor"])
def test_stale_or_corrupt_exports_are_not_repaired(config, generator, corruption):
    payload = fixture_payload(config)
    if corruption == "source_identity":
        payload["source_identity"]["sha256"] = "0"*64
    elif corruption == "runtime_contract":
        payload["runtime_contract_sha256"] = "0"*64
    else:
        payload["old_policy"]["state"]["phi.log_std"].add_(.5)
    write_policy(config, payload)
    result = builder.build_phase2_source(config)
    assert result["status"] == "SOURCE_NOT_AVAILABLE"
    generator.assert_not_called()
    assert not Path(config["source_output_path"]).exists()


@pytest.mark.parametrize("seeds", [[1]*10, list(range(9)), [True]+list(range(1,10))])
def test_invalid_seed_declarations_fail_closed(config, generator, seeds):
    config["scenario_seeds"] = seeds
    with pytest.raises(ValueError, match="10 distinct"):
        builder.build_phase2_source(config)
    generator.assert_not_called()


def test_existing_output_is_preserved(config, generator):
    output = Path(config["source_output_path"])
    output.parent.mkdir()
    output.write_bytes(b"user-retained-source")
    with pytest.raises(FileExistsError):
        builder.build_phase2_source(config)
    assert output.read_bytes() == b"user-retained-source"
    generator.assert_not_called()


def test_default_missing_source_status_and_cli(config, generator, tmp_path, capsys):
    defaults, runtime = builder._resolve_config(builder.load_config())
    assert defaults["policy_source_path"] == builder.ROOT/"outputs/chapter3/hgr_phase2/policy_pair_source.pt"
    assert defaults["source_output_path"] == builder.ROOT/"outputs/chapter3/hgr_phase2/frozen_phase2_source.pt"
    evaluation = phase2.load_config(builder.ROOT/"configs/chapter3/hgr_phase2_gradient_efficiency.json")
    assert Path(evaluation["snapshot_source"]) == defaults["source_output_path"]
    assert runtime["profile"] == builder.PROFILE and runtime["max_steps"] == 400
    config["policy_source_path"] = str(tmp_path/"awaiting_real_policy_pair.pt")
    config_path = tmp_path/"config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert builder.main(["--config", str(config_path)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "SOURCE_NOT_AVAILABLE"
    assert "missing" in result["reason"]
    assert not Path(config["source_output_path"]).parent.exists()
    generator.assert_not_called()
