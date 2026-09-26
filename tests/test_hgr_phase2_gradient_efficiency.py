"""Synthetic Phase2 tests: never create/restore a real runtime or train."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from chapter3_bser.experiments.hgr import phase2_gradient_efficiency as phase2
from chapter3_bser.models.hgr.phase1 import behavior_identity, runtime_contract
from chapter3_bser.models.hgr.policy import HandoffPolicy, weights_hash

ROOT = Path(__file__).resolve().parents[1]


class SyntheticPolicy(nn.Module):
    def __init__(self, suffix_value):
        super().__init__()
        self.theta_minus = nn.Linear(2, 1, bias=False).double()
        self.phi = nn.Linear(1, 1, bias=False).double()
        with torch.no_grad():
            self.theta_minus.weight.copy_(torch.tensor([[.2, .3]], dtype=torch.float64))
            self.phi.weight.fill_(suffix_value)

    def score_log_prob(self, record):
        if record["suffix"]:
            return self.theta_minus.weight.sum()*0
        return (self.theta_minus.weight.flatten()*torch.tensor(record["score"], dtype=torch.float64)).sum()


class SyntheticBackend:
    def __init__(self, *, random_labels=False, no_handoff=False):
        self.random_labels, self.no_handoff = random_labels, no_handoff
        self.collect_calls, self.branch_calls = [], []

    def collect(self, runtime, case, policy, *, seed, stream_id):
        self.collect_calls.append(stream_id)
        suffix = float(policy.phi.weight.detach())
        noise = np.random.default_rng(seed).normal() if self.random_labels else 0.
        tau = None if self.no_handoff else 1
        records = [dict(t=0, score=[1., 2.], suffix=False, dones=[False]*4, team_reward=1.),
                   dict(t=1, score=[0., 0.], suffix=tau is not None, dones=[True]*4, team_reward=suffix+noise)]
        snapshot = None if tau is None else SimpleNamespace(step=1, max_steps=400,
                         sha256=hashlib.sha256(stream_id.encode()).hexdigest())
        contract = runtime_contract(runtime)
        return dict(records=records, rewards=[1., suffix+noise], tau=tau, snapshot=snapshot,
                    features=None if tau is None else np.zeros(153), consumed=False, trajectory_complete=True,
                    theta_hash=weights_hash(policy.theta_minus),
                    theta_behavior_sha256=behavior_identity(policy.theta_minus, contract).sha256,
                    suffix_behavior_sha256=behavior_identity(policy.phi, contract).sha256)

    def branch(self, snapshot, policy, noise, *, keep_records=False):
        self.branch_calls.append((noise.pair_id, noise.role, noise.policy_seed, noise.environment_seed))
        value = float(policy.phi.weight.detach())
        if self.random_labels:
            value += float(np.random.default_rng(noise.environment_seed).normal())
        record = dict(t=snapshot.step, score=[0., 0.], suffix=True, dones=[True]*4, team_reward=value)
        return dict(G_plus=value, steps=1, terminal_step=snapshot.step+1,
                    termination_reason="synthetic_terminal", records=[record] if keep_records else [],
                    random_stream_id=noise.public()["trace_id"], **noise.public())


def config(output, count=2, repeats=3, budget=8, storage="inline"):
    runtime = json.loads((ROOT/"configs/chapter3/hgr_phase1_zero.json").read_text(encoding="utf-8"))
    runtime["correction_draws_per_cycle"] = 2
    return dict(schema=phase2.SCHEMA, scope="fresh_main", runtime_config=runtime,
                snapshot_source=None, snapshot_count=count, repeat_count=repeats,
                reference_rollouts=budget, reference_budget_unit="total_rollouts",
                random_source_revision=phase2.STREAM_REVISION, seed=1829,
                vector_storage=storage, output_dir=str(output))


def source(count=2, same=False):
    old = SyntheticPolicy(2.)
    new = copy.deepcopy(old)
    with torch.no_grad():
        new.phi.weight.fill_(2. if same else 5.)
    cases = [dict(snapshot_id=f"initial_{i}", scenario=dict(scenario_profile="M20_MOVING_UNKNOWN_MULTI",
             max_steps=400, scenario_id=f"scenario_{i}", scenario_seed=17+i)) for i in range(count)]
    return phase2.FrozenSnapshotSource(old, new, cases, "fresh_main", phase2.fresh_source_identity())


class Phase2GradientEfficiencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.guards = [
            patch("chapter3_bser.experiments.hgr.runtime.MissionRuntime.__init__",
                  side_effect=AssertionError("real simulator forbidden")),
            patch("chapter3_bser.experiments.hgr.runtime.MissionRuntime.restore",
                  side_effect=AssertionError("real snapshot restore forbidden")),
            patch("torch.optim.SGD.step", side_effect=AssertionError("optimizer forbidden")),
            patch("torch.optim.Adam.step", side_effect=AssertionError("optimizer forbidden")),
        ]
        for guard in cls.guards:
            guard.start()

    @classmethod
    def tearDownClass(cls):
        for guard in reversed(cls.guards):
            guard.stop()

    def test_synthetic_estimator_sanity_no_update_and_all_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(Path(directory)/"collision_terminal"/"sanity")
            frozen = source(); backend = SyntheticBackend()
            before = [behavior_identity(p, runtime_contract(cfg["runtime_config"])).exact
                      for p in (frozen.old_policy, frozen.new_policy)]
            report = phase2.run_phase2_gradient_efficiency(cfg, frozen, _backend=backend)
            expected = (1+.95*5)*np.array([1., 2.])
            np.testing.assert_allclose(report["reference_gradient"]["values"], expected, atol=1e-12, rtol=0)
            self.assertAlmostEqual(report["reference_gradient_norm"],np.linalg.norm(expected))
            self.assertEqual(report["policy_updates"],0)
            self.assertEqual(report["gradient_samples_per_method"],6)
            self.assertEqual(len(report["samples"]),18)
            # 8 reference full rollouts (16) + 12 MC full rollouts (24)
            # + 6 HGR main (12) + 6*2 pairs, each of length 1 (24).
            self.assertEqual(report["total_environment_steps"],76)
            for row in report["samples"]:
                gradient = np.array(row["gradient_vector"]["values"])
                if row["method"] in ("hgr", "direct_new_mc"):
                    np.testing.assert_allclose(gradient, expected, atol=1e-12, rtol=0)
                    self.assertLess(row["gradient_mse"],1e-22)
                else:
                    self.assertAlmostEqual(row["gradient_mse"],(.95*3)**2*5,places=10)
                self.assertEqual(row["cost_normalized_error"],row["gradient_mse"]*row["environment_steps"])
                if row["method"]=="hgr":
                    np.testing.assert_allclose(np.array(row["g0"]["values"])+row["g_delta"]["values"],
                                               row["g_full"]["values"],atol=1e-12,rtol=0)
                    self.assertEqual(row["environment_steps"],6)
            for policy, identity in zip((frozen.old_policy,frozen.new_policy),before):
                self.assertEqual(behavior_identity(policy,runtime_contract(cfg["runtime_config"])).exact,identity)
                self.assertTrue(all(p.grad is None for p in policy.parameters()))
            self.assertEqual(report["handoff_correction_status"],"EXERCISED")

    def test_same_source_reproducibility_named_stream_separation_and_rng_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"collision_terminal"
            cfg=config(root/"one"); frozen=source()
            random.seed(43); np.random.seed(43); torch.manual_seed(43)
            saved=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
            a_backend=SyntheticBackend(random_labels=True)
            a=phase2.run_phase2_gradient_efficiency(cfg,frozen,_backend=a_backend)
            self.assertEqual(random.getstate(),saved[0])
            np.testing.assert_array_equal(np.random.get_state()[1],saved[1][1])
            self.assertEqual(np.random.get_state()[2:],saved[1][2:])
            self.assertTrue(torch.equal(torch.get_rng_state(),saved[2]))
            cfg["output_dir"]=str(root/"two")
            b=phase2.run_phase2_gradient_efficiency(cfg,frozen,_backend=SyntheticBackend(random_labels=True))
            self.assertEqual(a["samples"],b["samples"])
            self.assertEqual(a["reference_gradient"],b["reference_gradient"])
            self.assertEqual(a["statistics"],b["statistics"])
            streams=[r["random_stream_id"] for r in a["samples"]]
            ref_streams=[r["random_stream_id"] for case in a["reference"] for r in case["samples"]]
            self.assertEqual(len(set(streams)),len(streams))
            self.assertFalse(set(streams)&set(ref_streams))
            comparison_pairs = {row["pair_id"] for row in a["samples"]}
            reference_pairs = {row["pair_id"] for case in a["reference"] for row in case["samples"]}
            self.assertEqual(len(comparison_pairs),6)
            self.assertFalse(comparison_pairs & reference_pairs)
            hgr=[r for r in a["samples"] if r["method"]=="hgr"]
            pair_ids=[q["pair_id"] for r in hgr for q in r["correction_queries"]]
            self.assertEqual(len(set(pair_ids)),len(pair_ids))
            # A larger reference budget cannot move comparison streams.
            cfg["output_dir"]=str(root/"three")
            c=phase2.run_phase2_gradient_efficiency(cfg,frozen,reference_budget=12,
                                                   _backend=SyntheticBackend(random_labels=True))
            for left,right in zip(a["samples"],c["samples"]):
                self.assertEqual(left["random_stream_id"],right["random_stream_id"])
                self.assertEqual(left["gradient_vector"],right["gradient_vector"])

    def test_method_schema_npz_vectors_and_variance_formula(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"collision_terminal"/"arrays"
            report=phase2.run_phase2_gradient_efficiency(config(output,storage="npz"),source(),
                                                        _backend=SyntheticBackend(random_labels=True))
            persisted=json.loads((output/"phase2_results.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["status"],"PASS")
            required={"snapshot_id","tau","phi0_hash","phi1_hash","snapshot_hash","method",
                      "repeat_id","environment_steps","gradient_norm","gradient_mse","random_stream_id"}
            for method in phase2.METHODS:
                rows=[r for r in report["samples"] if r["method"]==method]
                vectors=[]
                for row in rows:
                    self.assertTrue(required<=set(row))
                    pointer=row["gradient_vector"]; path=output/pointer["path"]
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),pointer["sha256"])
                    with np.load(path,allow_pickle=False) as arrays:
                        vectors.append(arrays[pointer["key"]])
                pointer=report["statistics"][method]["per_dimension_variance"]
                with np.load(output/pointer["path"],allow_pickle=False) as arrays:
                    np.testing.assert_allclose(arrays[pointer["key"]],np.var(vectors,axis=0,ddof=1),
                                               atol=1e-12,rtol=0)
            self.assertGreater(report["reference_mean_variance_trace"],0.)

    def test_zero_update_and_no_handoff_keep_denominator_and_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=config(Path(directory)/"collision_terminal"/"same")
            backend=SyntheticBackend(random_labels=True)
            report=phase2.run_phase2_gradient_efficiency(cfg,source(same=True),_backend=backend)
            self.assertEqual(backend.branch_calls,[])
            for row in report["samples"]:
                if row["method"]=="hgr":
                    self.assertEqual(row["K_actual"],0)
                    self.assertEqual(row["skip_reason"],"identical_suffix_behavior")
                    self.assertEqual(row["g_delta"]["values"],[0.,0.])
            cfg["output_dir"]=str(Path(directory)/"collision_terminal"/"no_handoff")
            backend=SyntheticBackend(no_handoff=True)
            report=phase2.run_phase2_gradient_efficiency(cfg,source(),_backend=backend)
            self.assertEqual(backend.branch_calls,[])
            self.assertEqual(report["no_handoff_hgr_samples"],6)
            self.assertEqual(report["handoff_correction_status"],"NOT_EXERCISED")
            for row in report["samples"]:
                if row["method"]=="hgr":
                    self.assertEqual(row["K_actual"],2)
                    self.assertEqual(row["environment_steps"],2)
                    self.assertIsNone(row["snapshot_hash"])

    def test_fixed_boundary_repeats_are_explicitly_conditional(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=config(Path(directory)/"collision_terminal"/"conditional",count=1,budget=4)
            cfg["scope"]="fixed_boundary"; frozen=source(1)
            full=SyntheticBackend().collect(cfg["runtime_config"],frozen.cases[0],frozen.old_policy,
                                           seed=1,stream_id="synthetic_prefix")
            prefix=dict(full,records=full["records"][:1],rewards=full["rewards"][:1],trajectory_complete=False)
            frozen.scope="fixed_boundary"; frozen.cases=[dict(snapshot_id="boundary_0",prefix=prefix)]
            frozen.preparation_environment_steps=1
            backend=SyntheticBackend(random_labels=True)
            report=phase2.run_phase2_gradient_efficiency(cfg,frozen,_backend=backend)
            self.assertEqual(backend.collect_calls,[])
            self.assertIn("conditional",report["estimand"])
            self.assertEqual({r["snapshot_hash"] for r in report["samples"]},{prefix["snapshot"].sha256})
            self.assertEqual(report["costs"]["source_preparation_steps"],1)
            self.assertTrue(all(row["random_stream_id"].endswith("/new") for row in report["samples"]))

    def test_default_protocol_and_runtime_adapter_delegate_without_simulation(self):
        cfg=phase2.resolve_config(phase2.load_config(ROOT/"configs/chapter3/hgr_phase2_gradient_efficiency.json"))
        self.assertEqual((cfg["scope"],cfg["snapshot_count"],cfg["repeat_count"],cfg["reference_rollouts"]),
                         ("fresh_main",10,20,1000))
        self.assertEqual(cfg["random_source_revision"],"hgr.phase1.named_streams.v1")
        builder_config=json.loads((ROOT/"configs/chapter3/hgr_phase2_source_builder.json").read_text(encoding="utf-8"))
        self.assertEqual(Path(cfg["snapshot_source"]), (ROOT/builder_config["source_output_path"]).resolve())
        backend=phase2.RuntimeBackend(); case=source(1).cases[0]; policy=source(1).old_policy
        noise=phase2.PairNoise.make(23,1,"unit_test",0,"new","policy_crn")
        with patch.object(phase2,"collect_trajectory",return_value="full") as collect:
            self.assertEqual(backend.collect(cfg["runtime_config"],case,policy,seed=41,stream_id="named"),"full")
            collect.assert_called_once_with(cfg["runtime_config"],case["scenario"],policy,seed=41,episode_id=41)
        with patch.object(phase2,"continue_branch",return_value="suffix") as branch:
            self.assertEqual(backend.branch("snapshot",policy,noise,keep_records=True),"suffix")
            branch.assert_called_once_with("snapshot",policy,pair_noise=noise,keep_records=True)

    def test_invalid_source_config_or_output_fails_before_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"collision_terminal"/"invalid"; cfg=config(output)
            backend=SyntheticBackend()
            with self.assertRaisesRegex(ValueError,"snapshot_source"):
                phase2.run_phase2_gradient_efficiency(cfg,_backend=backend)
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError,"divisible"):
                phase2.run_phase2_gradient_efficiency(cfg,source(),reference_budget=7,_backend=backend)
            bad=source()
            with torch.no_grad(): bad.new_policy.theta_minus.weight.add_(1.)
            with self.assertRaisesRegex(ValueError,"same theta"):
                phase2.run_phase2_gradient_efficiency(cfg,bad,_backend=backend)
            bad=source(); bad.source_identity["sha256"]="0"*64
            with self.assertRaises(ValueError):
                phase2.run_phase2_gradient_efficiency(cfg,bad,_backend=backend)
            self.assertEqual(backend.collect_calls,[])
            output.mkdir(parents=True); (output/"retained.txt").write_text("keep",encoding="utf-8")
            with self.assertRaises(FileExistsError):
                phase2.run_phase2_gradient_efficiency(cfg,source(),_backend=backend)
            self.assertEqual((output/"retained.txt").read_text(encoding="utf-8"),"keep")

    def test_mutation_and_nan_fail_with_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=config(Path(directory)/"collision_terminal"/"mutation")
            class MutatingBackend(SyntheticBackend):
                def collect(self,runtime,case,policy,**kwargs):
                    trajectory=super().collect(runtime,case,policy,**kwargs)
                    with torch.no_grad(): policy.phi.weight.add_(1.)
                    return trajectory
            with self.assertRaisesRegex(ValueError,"another frozen behavior"):
                phase2.run_phase2_gradient_efficiency(cfg,source(),_backend=MutatingBackend())
            report=json.loads((Path(cfg["output_dir"])/"phase2_results.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"],"FAIL")
            self.assertIn("another frozen behavior",report["error"])
            self.assertEqual(report["costs"]["reference_steps"],2)
            cfg["output_dir"]=str(Path(directory)/"collision_terminal"/"nan")
            class NaNBackend(SyntheticBackend):
                def branch(self,*args,**kwargs):
                    return dict(super().branch(*args,**kwargs),G_plus=float("nan"))
            with self.assertRaisesRegex(ValueError,"branch return"):
                phase2.run_phase2_gradient_efficiency(cfg,source(),_backend=NaNBackend())
            report=json.loads((Path(cfg["output_dir"])/"phase2_results.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"],"FAIL")

    def test_frozen_source_artifact_roundtrip_no_training_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=config(Path(directory)/"collision_terminal"/"unused",count=1,budget=4)
            runtime=cfg["runtime_config"]
            runtime["policy"]["actor"].update(hidden_dim=8,expert_hidden_dim=8)
            old=HandoffPolicy(runtime["policy"]); new=copy.deepcopy(old)
            with torch.no_grad(): new.phi.log_std.add_(.01)
            cases=source(1).cases
            frozen=phase2.FrozenSnapshotSource(old,new,cases,"fresh_main",phase2.fresh_source_identity())
            path=Path(directory)/"frozen_source.pt"
            phase2.save_snapshot_source(path,frozen,runtime)
            restored=phase2.load_snapshot_source(path,runtime)
            contract=runtime_contract(runtime)
            for a,b in ((old,restored.old_policy),(new,restored.new_policy)):
                self.assertEqual(behavior_identity(a,contract).exact,behavior_identity(b,contract).exact)
            with self.assertRaises(FileExistsError): phase2.save_snapshot_source(path,frozen,runtime)
            checkpoint=Path(directory)/"synthetic_wrong_schema.pt"
            torch.save(dict(schema="hgr.complete_cycle.phase1.v1"),checkpoint)
            with self.assertRaisesRegex(ValueError,"training checkpoints"):
                phase2.load_snapshot_source(checkpoint,runtime)


if __name__=="__main__":
    unittest.main()
