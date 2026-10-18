"""Bounded causal checks for reward timing, execution masks and learner failures."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from core.algorithms.maddpg import MADDPG
from chapter3_bser.experiments.baselines.common.model import build_model as old_model
from chapter3_bser.experiments.baselines.common.runtime import BaselineMissionRuntime
from chapter3_bser.experiments.d2_repair_v1.config import VERSION, validate
from chapter3_bser.experiments.d2_repair_v1.model import build_model
from chapter3_bser.experiments.d2_repair_v1.replay import ExecutionSample, ExecutionReplay
from chapter3_bser.experiments.d2_repair_v1.runtime import RepairedMissionRuntime, execution_gates
from chapter3_bser.experiments.d2_repair_v1.diagnostics import paired_ordering

ROOT = Path(__file__).resolve().parents[1]


def config(label="b2"):
    c = json.loads((ROOT/f"configs/chapter3/d2_repair_v1/{label}_train.json").read_text())
    c["rl"].update(hidden_dim=16, batch_size=4, replay_size=32, warmup_steps=0, update_frequency=1, updates_per_train=1)
    return validate(c)


def sample(n=8, finished=False):
    obs = torch.randn(n, 4, 28)
    obs[..., 20] = 0
    obs[..., 22:26] = torch.eye(4)
    obs[..., 26:28] = torch.tensor([1., 0.])
    if finished:
        obs[:, :3, 20] = 1
    enabled = torch.ones(n, 4)
    if finished:
        enabled[:, :3] = 0
    values = ([obs[:,i].clone() for i in range(4)], [torch.zeros(n,3) for _ in range(4)],
              [torch.full((n,), .25) for _ in range(4)], [obs[:,i].clone() for i in range(4)],
              [torch.zeros(n) for _ in range(4)], torch.ones(n), torch.arange(n), torch.zeros(n, dtype=torch.bool))
    return ExecutionSample(values, enabled.clone(), enabled.clone())


def cloned(net):
    return {k:v.detach().clone() for k,v in net.state_dict().items()}


def delta(a, net):
    return max(float((a[k]-v).abs().max()) for k,v in net.state_dict().items())


class Constant(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, values):
        return torch.full((len(values),1), self.value)


class Zero:
    def step(self, observations, explore=False):
        return [torch.zeros(1,3) for _ in range(4)]


class Repairs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(2729)

    def test_contracts_and_rejected_reweighting(self):
        for label in ("b2", "b3"):
            c = config(label)
            m = build_model(c)
            self.assertEqual((m.gamma, m.tau, m.lr_actor, m.lr_critic), (.95,.005,.001,.0005))
            a = m.agents[0].policy
            a = a.actor if label == "b3" else a
            self.assertEqual((a.fc1.in_features,a.fc3.out_features,m.agents[0].critic1.fc1.in_features), (28,3,124))
            c["rl"]["gamma"] = .998
            with self.assertRaises(ValueError): validate(c)
            c = config(label)
            c["reward"]["contact_entry_bonus"] = 1
            with self.assertRaises(ValueError): validate(c)
            c = config(label)
            c["collision_terminal_reward"] = -1.
            with self.assertRaises(ValueError): validate(c)

    def test_frozen_actor_does_not_move_even_with_adam_momentum(self):
        for label in ("b2", "b3"):
            m = build_model(config(label))
            m.update(sample(),0)  # Populate Adam momentum first.
            before = cloned(m.agents[0].policy)
            critic = cloned(m.agents[0].critic1)
            opt = copy.deepcopy(m.agents[0].policy_optimizer.state_dict())
            m.update(sample(finished=True),0)
            self.assertFalse(m.last_actor_updated)
            self.assertEqual(delta(before,m.agents[0].policy), 0)
            self.assertGreater(delta(critic,m.agents[0].critic1),0)
            from chapter3_bser.experiments.baselines.common.checkpoint import state_digest
            self.assertEqual(state_digest(opt),state_digest(m.agents[0].policy_optimizer.state_dict()))
            self.assertEqual(m.actor_updates[0],1)

    def test_mixed_actor_loss_equals_only_active_rows(self):
        for label in ("b2", "b3"):
            m = build_model(config(label))
            other = build_model(config(label))
            other.load_training_state_dict(m.training_state_dict())
            for model in (m,other):
                for agent in model.agents:
                    for opt in (agent.critic1_optimizer,agent.critic2_optimizer):
                        opt.param_groups[0]["lr"] = 0
            s = sample()
            s.enabled[4:,0] = 0
            short = ExecutionSample(tuple([v[:4] for v in x] if isinstance(x,list) else x[:4] for x in s),
                                    s.enabled[:4],s.next_enabled[:4])
            m.update(s,0)
            other.update(short,0)
            self.assertLess(delta(cloned(m.agents[0].policy),other.agents[0].policy), 2e-7)

    def test_gates_in_online_and_target_inputs(self):
        for label in ("b2", "b3"):
            m = build_model(config(label))
            s = sample()
            s.enabled[:,1] = 0
            s.next_enabled[:,2] = 0
            b = m._prepare_sample(s)
            self.assertTrue(bool((b["next_acs"][2] == 0).all()))
            seen = []
            hook = m.agents[0].critic1.register_forward_pre_hook(lambda net,args: seen.append(args[0].detach().clone()))
            try: m.update(s,0)
            finally: hook.remove()
            self.assertEqual(seen[-1].shape[-1],124)
            self.assertTrue(bool((seen[-1][:,115:118] == 0).all()))

    def test_ungated_legacy_learning_matches_before_source(self):
        # Compare a valid batch to the exact parent implementation captured at
        # task start, not a second invocation of the repaired implementation.
        path = ROOT/"tests/fixtures/maddpg_before_repair.py.txt"
        import hashlib
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "a574285b33d9a82971cf8016ce93e79f1c90b07cca5236edf5deb7fcb9b6d907")
        namespace = {}
        exec(compile(path.read_text(encoding="utf-8"),str(path),"exec"),namespace)
        c = config()
        p = [dict(num_in_pol=28,num_out_pol=3,num_in_critic=124)]*4
        kwargs = dict(hidden_dim=16,lr_actor=.001,lr_critic=.0005,residual_action_reg=0.)
        torch.manual_seed(75)
        before = namespace["MADDPG"](p,["MADDPG"]*4,**kwargs)
        torch.manual_seed(75)
        after = MADDPG(p,["MADDPG"]*4,**kwargs)
        s = sample()
        for i in range(4):
            a = before.update(s,i)
            b = after.update(s,i)
            self.assertEqual(a[:2],b[:2])
        before.update_all_targets()
        after.update_all_targets()
        for x,y in zip(before.agents,after.agents):
            for name in ("policy","critic1","critic2","target_policy","target_critic1","target_critic2"):
                self.assertEqual(delta(cloned(getattr(x,name)),getattr(y,name)),0)

    def test_all_repairs_disabled_preserve_b2_b3_learning(self):
        for label in ("b2", "b3"):
            c = config(label)
            c["repairs"] = dict(waypoint_arrival=False,inactive_actor=False,hold_residual=False)
            c.pop("waypoint_reward_timing")
            torch.manual_seed(182)
            legacy = old_model(c)
            torch.manual_seed(182)
            repaired = build_model(c)
            s = sample()
            for i in range(4):
                legacy.update(s,i)
                repaired.update(s,i)
            for a,b in zip(legacy.agents,repaired.agents):
                self.assertLessEqual(delta(cloned(a.policy),b.policy),1e-7)
                self.assertEqual(delta(cloned(a.critic1),b.critic1),0)

    def test_arrival_repair_cannot_run_after_terminal_collision(self):
        from tests.test_collision_terminal_protocol import collision_env
        env = collision_env()
        try:
            rt = env.unwrapped
            rt.waypoint_reward_timing = "arrival_before_reward_v1"
            rt._search_waypoints[0].copy_(rt._agent_pos[0])
            with patch.object(rt,"_capture_search_arrival_rewards",side_effect=AssertionError("arrival after collision")):
                obs,rewards,dones = env.step(torch.zeros(4,3))
            self.assertEqual(dones,[True]*4)
            self.assertEqual(torch.as_tensor(rewards).tolist(),[-2.]*4)
        finally: env.close()

    def test_local_ddpg_uses_local_inputs_in_all_branches(self):
        m = MADDPG([dict(num_in_pol=28,num_out_pol=3,num_in_critic=31)]*4,["DDPG"]*4,hidden_dim=16)
        self.assertTrue(np.isfinite(m.update(sample(),0)[:2]).all())
        self.assertTrue(np.isfinite(m.update_critic_only(sample(),0)[0]))

    def test_terminal_nan_is_ignored_but_live_nan_raises_before_step(self):
        m = build_model(config())
        a = m.agents[0]
        a.target_critic1 = Constant(float("nan"))
        a.target_critic2 = Constant(float("inf"))
        target = m._compute_target_q(a,torch.zeros(1,124),torch.tensor([-2.]),torch.ones(1))
        self.assertEqual(target.item(),-2.)
        before = cloned(a.critic1)
        with self.assertRaises(FloatingPointError): m.update(sample(),0)
        self.assertEqual(delta(before,a.critic1),0)
        self.assertFalse(a.critic1_optimizer.state)

    def test_nonfinite_gradient_cannot_partially_step_twin_critics(self):
        m = build_model(config())
        a = m.agents[0]
        before = [cloned(a.critic1),cloned(a.critic2)]
        hook = next(a.critic2.parameters()).register_hook(lambda grad: grad*float("nan"))
        try:
            with self.assertRaises(RuntimeError): m.update(sample(),0)
        finally: hook.remove()
        for state,net in zip(before,(a.critic1,a.critic2)): self.assertEqual(delta(state,net),0)

    def test_mutated_sample_rebuilds_targets(self):
        m = build_model(config())
        s = sample()
        before = m._prepare_sample(s)["target_vf_in"].clone()
        s[3][0].add_(5)
        self.assertFalse(torch.equal(before,m._prepare_sample(s)["target_vf_in"]))
        s[2][0][0] = float("inf")
        with self.assertRaises(FloatingPointError): m.update(s,0)

    def test_replay_wrap_keeps_gates_with_transitions(self):
        replay = ExecutionReplay(5)
        for k in range(9):
            gate = np.array([bool(k%2), True, True, True])
            actions = np.zeros((4,3))
            replay.push_transition(dict(observations=[np.full(28,k)]*4,actions=actions,rewards=[k]*4,
                next_observations=[np.full(28,k+1)]*4,dones=[False]*4,
                action_enabled=gate,next_action_enabled=~gate))
        s = replay.sample(5)
        self.assertEqual(sorted(s[0][0][:,0].tolist()),[4,5,6,7,8])
        self.assertTrue(torch.equal(s.enabled[:,0],s[0][0][:,0].remainder(2).bool()))
        self.assertTrue(torch.equal(s.next_enabled,~s.enabled))
        with self.assertRaises(ValueError): build_model(config()).update(tuple(s),0)
        with self.assertRaises(ValueError):
            replay.push_transition(dict(observations=[np.zeros(28)]*4,actions=np.ones((4,3))*2,
                rewards=[0]*4,next_observations=[np.zeros(28)]*4,dones=[False]*4,
                action_enabled=np.ones(4),next_action_enabled=np.ones(4)))

    def test_hold_and_unreachable_requests_override_residual(self):
        obs = np.zeros((4,28))
        ctx = SimpleNamespace(agent_assignments=tuple(SimpleNamespace(agent_id=i,hold_state=i==0,reachable=i!=1) for i in range(4)))
        np.testing.assert_array_equal(execution_gates(obs,ctx,mask_hold=True),[False,False,True,True])

    def test_real_waypoint_bonus_once_with_unchanged_other_components(self):
        scene = json.loads((ROOT/"configs/scenarios/e0_equivalence/M20_MOVING_UNKNOWN_MULTI.json").read_text())["scenarios"][0]
        traces = []
        for fixed in (False, True):
            c = config()
            c["repairs"] = dict(waypoint_arrival=fixed,inactive_actor=False,hold_residual=False)
            c.pop("waypoint_reward_timing")
            rt = RepairedMissionRuntime(c,copy.deepcopy(scene),seed=12729)
            try:
                base = rt.env.unwrapped
                base._search_waypoints[0].copy_(base._agent_pos[0])
                rows = []
                for _ in range(2):
                    row = rt.advance(Zero(),explore=False)
                    rows.append(dict(obs=row["next_observations"],count=int(base.waypoint_reached_counts[0]),
                                     components={k:v.clone() for k,v in base.last_reward_components.items()}))
                traces.append(rows)
                self.assertEqual(rows[0]["count"],1)
                self.assertEqual(rows[1]["count"],1)
                bonus = float(base._waypoint_bonus[0])
            finally: rt.close()
        for t in range(2):
            np.testing.assert_array_equal(traces[0][t]["obs"],traces[1][t]["obs"])
            for key,value in traces[0][t]["components"].items():
                other = traces[1][t]["components"][key]
                if key == "reward_coverage" and t == 0:
                    torch.testing.assert_close(other-value,torch.tensor([bonus,0,0,0]))
                else: torch.testing.assert_close(value,other,rtol=0,atol=0)

    def test_real_frozen_transition_and_hold_action_logging(self):
        scene = copy.deepcopy(json.loads((ROOT/"configs/scenarios/e0_equivalence/M20_MOVING_UNKNOWN_MULTI.json").read_text())["scenarios"][0])
        point = (np.array(scene["initial_agent_positions"][0])+[.4,.4,.2]).tolist()
        scene.update(target_position=point,target_initial_position=point,target_initial_velocity=[.05,0,0])
        rt = RepairedMissionRuntime(config(),scene,seed=12729)
        replay = ExecutionReplay(8)
        try:
            for _ in range(4): row = rt.advance(Zero(),explore=False)
            self.assertEqual([o[20] for o in row["observations"]],[1,1,1,0])
            replay.push_transition(row)
            for label in ("b2","b3"):
                m = build_model(config(label))
                before = cloned(m.agents[0].policy)
                m.update(replay.sample(1),0)
                self.assertEqual(delta(before,m.agents[0].policy),0)
                self.assertFalse(m.last_actor_updated)
            class Ones:
                def step(self, observations, explore=False):
                    return [torch.ones(1,3)*2 for _ in range(4)]
            applied = rt.advance(Ones(),explore=False)
            np.testing.assert_array_equal(applied["actions"][:3],np.zeros((3,3)))
            np.testing.assert_array_equal(applied["proposed_actions"],np.ones((4,3))*2)
            self.assertTrue(bool((np.abs(applied["actions"]) <= 1).all()))
            row = rt.summary()
            self.assertFalse(row["observation_risks"]["markov_sufficiency_established"])
        finally: rt.close()

    def test_ordering_detects_return_metric_disagreement(self):
        base = dict(scenario_id="s",scenario_seed=1,scenario_sha256="scenario_fingerprint",environment_innovation_seed=2,gamma=.95,
                    task_protocol="collision_terminal_v1",reward_objective="team_mean_v1",source_reward_revision="v1",
                    waypoint_reward_timing="arrival_before_reward_v1",max_steps=400,trajectory_complete=True)
        rows = [dict(base,policy_id="a",team_discounted_return=2.,found_step=None,pre_found_collision=True,found_penalized_steps=400),
                dict(base,policy_id="b",team_discounted_return=1.,found_step=100,pre_found_collision=False,found_penalized_steps=100)]
        report = paired_ordering(rows)
        self.assertTrue(all(r["discordant_pairs"] == 1 for r in report["metrics"].values()))
        self.assertFalse(report["reward_metric_equivalence_established"])
        rows[1]["environment_innovation_seed"] = 3
        self.assertEqual(paired_ordering(rows)["paired_scenario_groups"],0)
        rows[1]["trajectory_complete"] = False
        with self.assertRaises(ValueError): paired_ordering(rows)

    def test_short_training_checkpoint_and_held_out_evaluation(self):
        from chapter3_bser.experiments.d2_repair_v1.train import Trainer, load_for_evaluation
        from chapter3_bser.experiments.d2_repair_v1.evaluate import evaluate
        scene = json.loads((ROOT/"tests/fixtures/hgr/handoff_manifest.json").read_text())["scenarios"][0]
        scene = copy.deepcopy(scene)
        scene.update(max_steps=4,scenario_id="repair_four_step_train",scenario_split="train",scenario_role="train")
        for label in ("b2", "b3"):
            with tempfile.TemporaryDirectory(prefix="d2-repair-",dir=ROOT/"tmp") as directory:
                c = config(label)
                c.update(max_steps=4,total_main_trajectories=1,checkpoint_interval=1)
                trainer = Trainer(c,Path(directory)/"collision_terminal/train")
                trainer.scenarios = [copy.deepcopy(scene)]
                summary = trainer.run()
                self.assertEqual(summary["environment_steps"],4)
                self.assertEqual(summary["optimizer_updates"],1)
                m,loaded,payload = load_for_evaluation(summary["latest_checkpoint"])
                self.assertEqual(m.actor_updates,summary["actor_optimizer_steps"])
                self.assertEqual(loaded["rl"]["gamma"],.95)
                self.assertFalse(payload["resume_supported"])
                corrupted = copy.deepcopy(payload)
                optimizer = corrupted["model_state"]["agent_params"][0]["critic1_opt"]
                next(iter(optimizer["state"].values()))["exp_avg"].view(-1)[0] = float("nan")
                from chapter3_bser.experiments.baselines.common.checkpoint import state_digest
                corrupted["model_state_sha256"] = state_digest(corrupted["model_state"])
                bad = Path(directory)/"bad_optimizer.pt"
                torch.save(corrupted,bad)
                with self.assertRaises(ValueError): load_for_evaluation(bad)
                held = copy.deepcopy(scene)
                held.update(scenario_id="repair_four_step_eval",scenario_split="eval",scenario_role="eval")
                manifest = Path(directory)/"manifest.json"
                manifest.write_text(json.dumps(dict(scenarios=[held])),encoding="utf-8")
                with self.assertRaises(ValueError):
                    evaluate(summary["latest_checkpoint"],manifest,Path(directory)/"collision_terminal/same_seed")
                held["scenario_seed"] += 100000
                manifest.write_text(json.dumps(dict(scenarios=[held])),encoding="utf-8")
                with self.assertRaises(ValueError):
                    evaluate(summary["latest_checkpoint"],manifest,Path(directory)/"collision_terminal/renamed_copy")
                held["target_initial_velocity"][0] += .01
                manifest.write_text(json.dumps(dict(scenarios=[held])),encoding="utf-8")
                output = Path(directory)/"collision_terminal/eval"
                evaluate(summary["latest_checkpoint"],manifest,output)
                rows = json.loads((output/"episodes.json").read_text())
                self.assertEqual(len(rows),1)
                self.assertTrue(rows[0]["trajectory_complete"])
                paired_ordering(rows)  # All required metric identities are present.
                held["scenario_id"] = scene["scenario_id"]
                manifest.write_text(json.dumps(dict(scenarios=[held])),encoding="utf-8")
                with self.assertRaises(ValueError):
                    evaluate(summary["latest_checkpoint"],manifest,Path(directory)/"collision_terminal/invalid")


if __name__ == "__main__":
    unittest.main()
