"""Synthetic orchestration checks; real MissionRuntime/Trainer are forbidden."""
import copy
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from chapter3_bser.experiments.hgr import phase2_gradient_efficiency as phase2
from chapter3_bser.models.hgr.phase1 import behavior_identity, runtime_contract
from chapter3_bser.models.hgr.policy import weights_hash
from scripts import hgr_decision_experiment as experiment
from tests.test_hgr_suffix_diagnostics import TinyPolicy


class DecisionPolicy(TinyPolicy):
    def score_log_prob(self, record, *, suffix=False):
        if suffix:
            return super().score_log_prob(record, suffix=True)
        return self.theta_minus.weight.sum() * (0. if record["suffix"] else 1.)


class Backend:
    def __init__(self, no_handoff=False, zero_update=False, bad_clock=False):
        self.no_handoff, self.zero_update, self.bad_clock = no_handoff, zero_update, bad_clock
        self.collect_calls, self.branch_calls, self.train_starts = [], [], []

    def training_case(self, runtime, seed, replica, index):
        return make_case(f"train_{replica}_{index}", seed)

    def collect(self, runtime, case, policy, *, seed, stream_id):
        self.collect_calls.append(stream_id)
        training = case["snapshot_id"].startswith("train_")
        if training:
            self.train_starts.append(weights_hash(policy.phi))
        tau = None if self.no_handoff else 1
        suffix_return = 0. if training and self.zero_update else 1.+float(policy.phi.mean.weight.detach().sum())
        records = []
        for t in range(2):
            suffix = tau is not None and t >= tau
            records.append(dict(t=t, suffix=suffix, dones=[t==1]*4,
                                observations=np.ones((4, 1)),
                                latents=[None, None, None, np.ones(3) if suffix else None],
                                team_reward=1. if t==0 else suffix_return))
        snapshot = None if tau is None else SimpleNamespace(step=1, max_steps=400,
                         sha256=hashlib.sha256(stream_id.encode()).hexdigest())
        contract = runtime_contract(runtime)
        return dict(records=records, rewards=[r["team_reward"] for r in records],
                    tau=tau, snapshot=snapshot, features=None if tau is None else np.zeros(153),
                    consumed=False, trajectory_complete=True,
                    theta_hash=weights_hash(policy.theta_minus),
                    theta_behavior_sha256=behavior_identity(policy.theta_minus, contract).sha256,
                    suffix_behavior_sha256=behavior_identity(policy.phi, contract).sha256)

    def branch(self, snapshot, policy, noise, *, keep_records=False):
        self.branch_calls.append(noise)
        return dict(G_plus=1.+float(policy.phi.mean.weight.detach().sum()), steps=1,
                    terminal_step=3 if self.bad_clock else 2, termination_reason="synthetic_terminal",
                    records=[], snapshot_hash=snapshot.sha256, **noise.public())


def make_case(name, seed):
    return dict(snapshot_id=name, scenario=dict(scenario_profile="M20_MOVING_UNKNOWN_MULTI",
                max_steps=400, scenario_id=name, scenario_seed=seed))


def source():
    old = DecisionPolicy()
    new = copy.deepcopy(old)
    with torch.no_grad():
        new.phi.mean.weight.fill_(.1)
    return phase2.FrozenSnapshotSource(old, new, [make_case(f"scene_{i:02d}", 100+i) for i in range(10)],
                                      "fresh_main", phase2.fresh_source_identity(), 17)


def config(output):
    cfg = experiment.load_config()
    cfg["output_dir"] = str(output)
    return cfg


@pytest.fixture(autouse=True)
def no_real_runtime():
    torch.set_num_threads(1)
    with patch("chapter3_bser.experiments.hgr.runtime.MissionRuntime.__init__", side_effect=AssertionError("no real env")), \
         patch("chapter3_bser.experiments.hgr.runtime.MissionRuntime.restore", side_effect=AssertionError("no real restore")), \
         patch("chapter3_bser.experiments.hgr.train.Trainer.__init__", side_effect=AssertionError("no Trainer")):
        yield


def test_default_cli_only_preflight():
    audit = dict(status="PREFLIGHT_PASS", source_identity={"sha256":"synthetic"}, checkout={"head":"synthetic"})
    with patch.object(experiment, "preflight", return_value=audit) as check, \
         patch.object(experiment, "run_experiment", side_effect=AssertionError("default executed")):
        assert experiment.main([]) == 0
        check.assert_called_once()


def test_budget_and_config_refuse_changes_before_any_rollout(tmp_path):
    cfg = config(tmp_path/"collision_terminal"/"bad")
    resolved, runtime = experiment.resolve_config(cfg)
    assert experiment.budget_plan(resolved, runtime)["total_max_steps"] == 60800
    assert experiment.budget_plan(resolved, runtime, True)["total_max_steps"] == 3200
    for key, value, message in (("snapshot_count", 2, "snapshot_count"),
                                ("correction_draws", 7, "must match"),
                                ("suffix_batch_size", 8, "must match"),
                                ("macro_repeats", 1, "macro_repeats")):
        candidate = dict(cfg, **{key:value})
        with pytest.raises(ValueError, match=message):
            experiment.resolve_config(candidate)
    cfg["max_environment_steps"] = 60799
    backend = Backend()
    with pytest.raises(ValueError, match="reserve"):
        experiment.run_experiment(cfg, _source=source(), _backend=backend)
    assert not backend.collect_calls
    assert not Path(cfg["output_dir"]).exists()


def test_missing_source_wrong_schema_and_output_rejected(tmp_path):
    cfg = config(tmp_path/"collision_terminal"/"new")
    cfg["snapshot_source"] = str(tmp_path/"absent.pt")
    with pytest.raises(ValueError, match="missing"):
        experiment.preflight(cfg)
    bad = tmp_path/"trainer.pt"
    torch.save(dict(schema="hgr.complete_cycle.v1"), bad)
    cfg["snapshot_source"] = str(bad)
    with pytest.raises(ValueError, match="frozen_source"):
        experiment.preflight(cfg)
    cfg["output_dir"] = str(tmp_path/"wrong_protocol")
    with pytest.raises(ValueError, match="collision_terminal"):
        experiment.preflight(cfg)
    assert not (tmp_path/"collision_terminal"/"new").exists()


def test_source_theta_and_case_count_gate(tmp_path):
    cfg = config(tmp_path/"collision_terminal"/"gate")
    frozen = source()
    frozen.cases.pop()
    with pytest.raises(ValueError, match="exactly"):
        experiment.run_experiment(cfg, _source=frozen, _backend=Backend())
    frozen = source()
    with torch.no_grad():
        frozen.new_policy.theta_minus.weight.add_(.1)
    with pytest.raises(ValueError, match="theta"):
        experiment.run_experiment(cfg, _source=frozen, _backend=Backend())


def test_synthetic_whole_protocol_original_estimator_costs_and_saved_arrays(tmp_path):
    output = tmp_path/"collision_terminal"/"screen"
    frozen, backend = source(), Backend()
    before = {k:v.clone() for k,v in frozen.new_policy.state_dict().items()}
    report = experiment.run_experiment(config(output), _source=frozen, _backend=backend)
    assert report["status"] == "PASS"
    assert report["parameter_updates"] == 2 and report["theta_updates"] == 0
    assert report["total_new_environment_steps"] == 2*(4*2 + 2*(20*2+16))
    assert report["total_including_historical_preparation_steps"] == report["total_new_environment_steps"]+17
    assert len(set(backend.train_starts)) == 1  # Replicas do not chain updates.
    assert len(set(backend.collect_calls)) == len(backend.collect_calls)
    assert len(set(n.pair_id for n in backend.branch_calls)) == 32
    for old_noise, new_noise in zip(backend.branch_calls[::2], backend.branch_calls[1::2]):
        assert old_noise.pair_id == new_noise.pair_id
        assert old_noise.policy_seed == new_noise.policy_seed
        assert old_noise.environment_seed != new_noise.environment_seed
    assert report["decision"]["status"] == "CONTINUE_SIGNAL_STUDY"
    for replica in report["replicas"]:
        assert replica["diagnostics"]["behavior"]["theta_identity_unchanged"]
        assert replica["statistics"]["repeat_count"] == 2
        assert not replica["statistics"]["cost_proxy"]["actual_equal_budget_comparison"]
        for macro in replica["macros"]:
            assert macro["main_batch_size"] == 10 and macro["fixed_K"] == 8
            assert len(macro["hgr_main"]) == len(macro["direct_main"]) == 10
            artifact = macro["vectors"]
            assert hashlib.sha256((output/artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
            with np.load(output/artifact["path"], allow_pickle=False) as vectors:
                np.testing.assert_allclose(vectors["g_full"], vectors["direct_new_mc"], atol=1e-12)
                np.testing.assert_array_equal(vectors["g_full"], vectors["g0"]+vectors["g_delta"])
    assert all(torch.equal(value, frozen.new_policy.state_dict()[k]) for k,value in before.items())
    assert json.loads((output/"decision_results.json").read_text(encoding="utf-8"))["status"] == "PASS"
    assert (output/"plan.json").is_file()
    # Retained evidence must not be overwritten on rerun.
    with pytest.raises(FileExistsError):
        experiment.run_experiment(config(output), _source=frozen, _backend=Backend())


@pytest.mark.parametrize("no_handoff,zero_update,expected", [(True,False,"NOT_EXERCISED"), (False,True,"STOP_CURRENT_HGR")])
def test_exact_zero_skip_is_not_near_zero_skip(tmp_path, no_handoff, zero_update, expected):
    backend = Backend(no_handoff=no_handoff, zero_update=zero_update)
    report = experiment.run_experiment(config(tmp_path/"collision_terminal"/"zero"), _source=source(), _backend=backend)
    assert len(backend.collect_calls) == 8
    assert not backend.branch_calls
    assert all(r["comparison_status"] == "EXACT_ZERO_UPDATE" for r in report["replicas"])
    assert report["decision"]["status"] == expected


def test_diagnostics_only_never_calls_branch_or_macro(tmp_path):
    backend = Backend()
    report = experiment.run_experiment(config(tmp_path/"collision_terminal"/"diag"), diagnostics_only=True,
                                       _source=source(), _backend=backend)
    assert len(backend.collect_calls) == 8 and not backend.branch_calls
    assert report["budget"]["total_max_steps"] == 3200
    assert report["decision"]["status"] == "INCONCLUSIVE"


def test_failure_keeps_traceback_partial_cost_and_evidence(tmp_path):
    output = tmp_path/"collision_terminal"/"fail"
    with pytest.raises(ValueError, match="clock"):
        experiment.run_experiment(config(output), _source=source(), _backend=Backend(bad_clock=True))
    report = json.loads((output/"decision_results.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL" and "Traceback" in report["traceback"]
    assert report["costs"]["suffix_training_steps"] == 8
    assert report["costs"]["hgr_correction_old_steps"] == 1
    assert report["failed_operation"]["operation"] == "branch"
    assert (output/"plan.json").exists()
    assert (output/"replica_00_policy_pair.pt").exists()


def test_late_collect_failure_retains_completed_trajectory_metadata(tmp_path):
    class LateFailure(Backend):
        def collect(self, *args, **kwargs):
            if len(self.collect_calls) == 3:
                raise RuntimeError("fourth suffix task failed")
            return super().collect(*args, **kwargs)
    output = tmp_path/"collision_terminal"/"late_collect"
    with pytest.raises(RuntimeError, match="fourth"):
        experiment.run_experiment(config(output), _source=source(), _backend=LateFailure())
    report = json.loads((output/"decision_results.json").read_text(encoding="utf-8"))
    assert len(report["replicas"][0]["suffix_trajectories"]) == 3
    assert report["costs"]["suffix_training_steps"] == 6
    assert report["failed_call_steps_unknown"]


def test_second_side_failure_retains_first_side_and_main_evidence(tmp_path):
    class NewFailure(Backend):
        def branch(self, snapshot, policy, noise, **kwargs):
            if noise.role == "new":
                raise RuntimeError("new side failed")
            return super().branch(snapshot, policy, noise, **kwargs)
    output = tmp_path/"collision_terminal"/"new_failed"
    with pytest.raises(RuntimeError, match="new side"):
        experiment.run_experiment(config(output), _source=source(), _backend=NewFailure())
    report = json.loads((output/"decision_results.json").read_text(encoding="utf-8"))
    macro = report["replicas"][0]["macros"][0]
    assert len(macro["hgr_main"]) == len(macro["direct_main"]) == 10
    query = macro["correction_queries"][0]
    assert "old" in query and "new" not in query
    assert query["label"] is None and query["status"] == "RUNNING"
    assert report["costs"]["hgr_correction_old_steps"] == 1
    assert report["costs"]["hgr_correction_new_steps"] == 0


def test_real_policy_architecture_on_synthetic_records_matches_original_update():
    from chapter3_bser.models.hgr.policy import HandoffPolicy
    from chapter3_bser.models.hgr.estimator import update_suffix
    from scripts.hgr_suffix_diagnostics import diagnose_suffix_update
    runtime = experiment.resolve_config(experiment.load_config())[1]
    policy = HandoffPolicy(runtime["policy"])
    observations = np.zeros((4, 28), dtype=np.float32)
    with torch.no_grad():
        mu, std = policy.phi.distribution_parameters(observations[3])
        latent = (mu+.5*std).reshape(3).numpy().copy()
    records = [dict(t=0, suffix=True, observations=observations,
                    latents=[None,None,None,latent], dones=[True]*4)]
    trajectory = dict(records=records, rewards=[1.], tau=0, consumed=False, trajectory_complete=True)
    expected = copy.deepcopy(policy)
    update_suffix(expected, torch.optim.SGD(expected.phi.parameters(), lr=runtime["suffix_lr"]),
                  [trajectory], runtime["rl"]["gamma"])
    new, diagnostics, _ = diagnose_suffix_update(policy, [trajectory], gamma=runtime["rl"]["gamma"],
                                                 lr=runtime["suffix_lr"], contract=runtime_contract(runtime))
    assert diagnostics["sgd_update"]["native_step_bitwise_match"]
    assert diagnostics["behavior"]["theta_identity_unchanged"]
    assert all(torch.equal(value,new.state_dict()[k]) for k,value in expected.state_dict().items())


def test_reproducibility_and_process_rng_isolation(tmp_path):
    frozen = source()
    random.seed(12); np.random.seed(12); torch.manual_seed(12)
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    reports = [experiment.run_experiment(config(tmp_path/"collision_terminal"/name), _source=frozen, _backend=Backend())
               for name in ("a", "b")]
    assert random.getstate() == state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], state[1][1])
    assert np.random.get_state()[2:] == state[1][2:]
    assert torch.equal(torch.get_rng_state(), state[2])
    # NPZ bytes need not have identical ZIP timestamps; stored numerical arrays must.
    for left,right in zip(reports[0]["replicas"], reports[1]["replicas"]):
        assert left["diagnostics"] == right["diagnostics"]
        assert left["statistics"] == right["statistics"]
        for a,b in zip(left["macros"],right["macros"]):
            assert a["correction_queries"] == b["correction_queries"]


def test_research_decision_preserves_uncertainty_and_all_replicas():
    thresholds = experiment.load_config()["thresholds"]
    def replica(center, noise, queries=2, changed=True):
        return dict(replica_id=0, paired_query_count=queries,
                    diagnostics=dict(effective_suffix_score_steps=1, behavior=dict(phi_identity_changed=changed)),
                    statistics=dict(signal=dict(g0_rms_norm=1., mean_delta_norm=center, mean_standard_error_scale=noise)))
    assert experiment.decide([replica(1e-9,1e-10)], thresholds)["status"] == "STOP_CURRENT_HGR"
    assert experiment.decide([replica(.1,.001)], thresholds)["status"] == "CONTINUE_SIGNAL_STUDY"
    assert experiment.decide([replica(0.,0.,queries=0)], thresholds)["status"] == "INCONCLUSIVE"
    assert experiment.decide([replica(1e-9,1e-10),replica(0.,0.,queries=0)], thresholds)["status"] == "INCONCLUSIVE"
    result = experiment.decide([replica(.001,.1)], thresholds)
    assert result["status"] == "INCONCLUSIVE"
    assert not result["assessments"][0]["these_are_confidence_bounds"]
