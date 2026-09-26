"""Phase2 launch checks; all rollout/training entry points are forbidden here."""
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from chapter3_bser.experiments.hgr import phase2_gradient_efficiency as phase2
from scripts import check_hgr_phase2_config as preflight


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    from chapter3_bser.experiments.hgr import runtime, train
    guards = [(phase2, "collect_trajectory"), (phase2, "continue_branch"),
              (phase2.Evaluation, "sample"), (runtime.MissionRuntime, "__init__"),
              (runtime.MissionRuntime, "restore"), (train.Trainer, "__init__"),
              (torch.optim.SGD, "step"), (torch.optim.Adam, "step")]
    for target, name in guards:
        monkeypatch.setattr(target, name, Mock(side_effect=AssertionError("rollout/training forbidden")))


@pytest.fixture
def setup(tmp_path):
    config = phase2.load_config(phase2.ROOT / "configs/chapter3/hgr_phase2_debug.json")
    runtime = phase2.resolve_config(config)["runtime_config"]
    runtime["policy"]["actor"].update(hidden_dim=8, expert_hidden_dim=8)
    with phase2.isolated_global_rng():
        old = phase2.HandoffPolicy(runtime["policy"])
        new = copy.deepcopy(old)
        with torch.no_grad():
            new.phi.log_std.add_(.01)
    cases = [dict(snapshot_id=f"scene_{i:02d}", scenario=dict(
        scenario_id=f"scene_{i:02d}", scenario_seed=200 + i,
        scenario_profile="M20_MOVING_UNKNOWN_MULTI", max_steps=400)) for i in range(10)]
    # Synthetic unit fixture only, under pytest's temporary directory.
    source = phase2.FrozenSnapshotSource(old, new, cases, "fresh_main", phase2.fresh_source_identity())
    source_path = tmp_path / "synthetic_frozen.pt"
    phase2.save_snapshot_source(source_path, source, runtime)
    config.update(runtime_config=runtime, snapshot_source=str(source_path),
                  output_dir=str(tmp_path / "collision_terminal" / "debug"))
    return config, tmp_path / "config.json"


def save_config(setup):
    config, path = setup
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


@pytest.mark.parametrize("directory", ["debug", "collision_terminal_debug"])
def test_illegal_output_rejected_before_rollout(setup, tmp_path, directory):
    config, _ = setup
    config["output_dir"] = str(tmp_path / directory)
    path = save_config(setup)
    with pytest.raises(preflight.Phase2ConfigError, match="output_dir.*collision_terminal"):
        preflight.validate_phase2_config(path)
    # The unchanged official Python entry also rejects this before collecting.
    with pytest.raises(ValueError, match="collision_terminal"):
        phase2.run_phase2_gradient_efficiency(phase2.load_config(path))
    assert not Path(config["output_dir"]).exists()


@pytest.mark.parametrize("existing_empty", [False, True])
def test_valid_collision_terminal_output_passes_without_writes(setup, existing_empty):
    config, _ = setup
    output = Path(config["output_dir"])
    if existing_empty:
        output.mkdir(parents=True)
    source_path = Path(config["snapshot_source"])
    before = source_path.read_bytes()
    result = preflight.validate_phase2_config(save_config(setup))
    assert result["status"] == "PREFLIGHT_PASS"
    assert result["snapshot_count"] == result["source_scenario_count"] == 10
    assert result["output_state"] == ("empty" if existing_empty else "not_created")
    assert result["experiment_started"] is False
    assert result["environment_steps"] == 0
    assert result["snapshot_source_sha256"] == hashlib.sha256(before).hexdigest()
    assert source_path.read_bytes() == before
    assert output.exists() == existing_empty
    if existing_empty:
        assert list(output.iterdir()) == []


def test_snapshot_count_two_rejected_for_ten_case_source(setup):
    config, _ = setup
    config["snapshot_count"] = 2
    path = save_config(setup)
    with pytest.raises(preflight.Phase2ConfigError, match="snapshot_count=2.*scenario count=10"):
        preflight.validate_phase2_config(path)
    with pytest.raises(ValueError, match="snapshot_count"):
        phase2.run_phase2_gradient_efficiency(phase2.load_config(path))
    assert not Path(config["output_dir"]).exists()


def test_missing_source_rejected_before_rollout(setup, tmp_path):
    config, _ = setup
    config["snapshot_source"] = str(tmp_path / "absent.pt")
    path = save_config(setup)
    with pytest.raises(preflight.Phase2ConfigError, match="snapshot_source file does not exist.*absent.pt"):
        preflight.validate_phase2_config(path)
    with pytest.raises(FileNotFoundError, match="absent.pt"):
        phase2.run_phase2_gradient_efficiency(phase2.load_config(path))
    assert not Path(config["output_dir"]).exists()


def test_wrong_frozen_schema_rejected(setup):
    config, _ = setup
    torch.save(dict(schema="not.a.frozen.source"), config["snapshot_source"])
    path = save_config(setup)
    with pytest.raises(preflight.Phase2ConfigError, match="hgr.phase2.frozen_source.v1"):
        preflight.validate_phase2_config(path)
    with pytest.raises(ValueError, match="hgr.phase2.frozen_source.v1"):
        phase2.run_phase2_gradient_efficiency(phase2.load_config(path))


def test_nonempty_output_preserved_and_resume_rejected(setup):
    config, _ = setup
    output = Path(config["output_dir"])
    output.mkdir(parents=True)
    retained = output / "retained.txt"
    retained.write_bytes(b"retained evidence")
    path = save_config(setup)
    with pytest.raises(preflight.Phase2ConfigError, match="resuming or appending"):
        preflight.validate_phase2_config(path)
    with pytest.raises(FileExistsError):
        phase2.run_phase2_gradient_efficiency(phase2.load_config(path))
    assert retained.read_bytes() == b"retained evidence"
    assert list(output.iterdir()) == [retained]


def test_invalid_runtime_rejected_before_rollout(setup):
    config, _ = setup
    config["runtime_config"]["rl"]["gamma"] = 2.
    path = save_config(setup)
    with pytest.raises(preflight.Phase2ConfigError, match="runtime_config invalid"):
        preflight.validate_phase2_config(path)
    with pytest.raises(ValueError, match="gamma"):
        phase2.run_phase2_gradient_efficiency(phase2.load_config(path))
    assert not Path(config["output_dir"]).exists()


def test_cli_reports_actionable_failure_and_success(setup, tmp_path, capsys):
    config, _ = setup
    path = save_config(setup)
    assert preflight.main(["--config", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PREFLIGHT_PASS"
    config["snapshot_source"] = str(tmp_path / "missing.pt")
    save_config(setup)
    assert preflight.main(["--config", str(path)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PREFLIGHT_FAIL"
    assert "snapshot_source" in result["error"]
    assert result["experiment_started"] is False
    assert not Path(config["output_dir"]).exists()


@pytest.mark.parametrize("name,repeats,budget,run_name", [
    ("hgr_phase2_debug.json", 2, 20, "debug_gradient_efficiency_10scene"),
    ("hgr_phase2_gradient_efficiency.json", 20, 1000, "gradient_efficiency_01"),
])
def test_supplied_configs_and_existing_frozen_source_read_only(name, repeats, budget, run_name, tmp_path):
    path = phase2.ROOT / "configs/chapter3" / name
    config = phase2.load_config(path)
    assert config["scope"] == "fresh_main"
    assert (config["snapshot_count"], config["repeat_count"], config["reference_rollouts"]) == (10, repeats, budget)
    output = phase2.ROOT / "outputs/chapter3/hgr_phase2/collision_terminal" / run_name
    assert Path(config["output_dir"]) == output
    frozen = Path(config["snapshot_source"])
    if not frozen.is_file():
        pytest.skip("real frozen source is a user-supplied, untracked artifact")
    before = frozen.read_bytes()
    # A completed user run must remain protected. Source validity must not
    # depend on whether that retained experiment has already been executed.
    if output.exists() and any(output.iterdir()):
        with pytest.raises(preflight.Phase2ConfigError, match="new or empty"):
            preflight.validate_phase2_config(path)
    probe_output = tmp_path / "collision_terminal" / run_name
    probe_config = dict(config, output_dir=str(probe_output))
    probe_path = tmp_path / name
    probe_path.write_text(json.dumps(probe_config), encoding="utf-8")
    result = preflight.validate_phase2_config(probe_path)
    assert result["status"] == "PREFLIGHT_PASS"
    assert result["source_scenario_count"] == 10
    assert result["snapshot_source_sha256"] == hashlib.sha256(before).hexdigest()
    assert frozen.read_bytes() == before
    assert not probe_output.exists()


def test_windows_launcher_preflights_before_official_entry():
    script = (phase2.ROOT / "scripts/run_hgr_phase2_gradient_efficiency.bat").read_text(encoding="utf-8")
    assert script.index("-m scripts.check_hgr_phase2_config") < script.index(
        "-m chapter3_bser.experiments.hgr.phase2_gradient_efficiency")
    assert "if errorlevel 1 exit /b %ERRORLEVEL%" in script
