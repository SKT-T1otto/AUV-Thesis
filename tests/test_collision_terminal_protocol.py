"""Regression acceptance for the production strict mission protocol, no long runs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from core.env.task_protocol import STRICT, LEGACY, closed_segment_aabb_first_hit as hit, protocol_identity
from chapter3_bser.experiments.phase1c_prrac import train_phase1c_prrac as trainer
from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.experiments.phase1c_prrac.task_metrics import aggregate_task_outcomes
from chapter3_bser.experiments.phase1c_prrac.checkpoint_transfer import import_actors
from tests.prrac_evaluation_support import ARCHITECTURE, LOSS, checkpoint_payload, worker_jobs, write_checkpoint

ROOT = Path(__file__).resolve().parents[1]
_ENV_TEMPLATES = {}


def strict_config():
    return trainer._load_config(ROOT / "configs/chapter3/bser_phase1c_prrac_collision_terminal_train.json")


def collision_env(agent=0, phase="Search", enabled=True, protocol=STRICT):
    config = strict_config()
    config["profile"] = "S00_STATIC_CLEAR"
    config.update(protocol_identity({"task_protocol": protocol}))
    config["max_steps"] = 2
    config["reward"]["enabled"] = enabled
    key = (protocol, enabled)
    if key not in _ENV_TEMPLATES:
        template = evaluator._make_env(config, config["reward"])
        template.reset()
        _ENV_TEMPLATES[key] = template.env.env.env
    # Copy the public core facade; transparent wrappers do not support deepcopy.
    base = copy.deepcopy(_ENV_TEMPLATES[key])
    env = evaluator.PRRACTrainingEnv(evaluator.Phase1CV2TrainingEnv(
        evaluator.GuidedEnv(base, enabled=True), reward_config=config["reward"]))
    env.env._last_task = env.get_task_state()
    env.env.diagnostics.reset(env, episode_id=0, episode_index=0,
        scenario_id=None, scenario_seed=None, max_steps=config["max_steps"])
    rt = env.unwrapped
    rt._agent_pos.copy_(torch.tensor([[2., 2., 2.], [6., 2., 2.], [10., 2., 2.], [14., 2., 2.]]))
    rt._agent_vel.zero_()
    rt.use_residual_prior = False
    rt._flow_at = lambda positions: torch.zeros_like(positions)
    if phase != "Search":
        rt.task_found = True
        rt.found_step = 0
        rt.agent_finished[:3] = True
        rt.executor_target_assigned = True
        rt._agent_task_known[:] = True
        rt._capture_hold_counter = 2 if phase == "Hold" else 0
        # Existing frozen Searchers stay frozen; inject a physical contact for them.
    start = rt._agent_pos[agent].numpy().copy()
    center = start + np.array([0.025, 0, 0])
    if agent < 3 and phase != "Search":
        center = start.copy()
    rt.obstacles = [{"center": center.tolist(), "size": [0.01, 0.04, 0.04]}]
    rt.ground_truth_obstacles = copy.deepcopy(rt.obstacles)
    rt._build_obstacle_tensors()
    rt._agent_vel[agent, 0] = 1.0
    # Preserve wrapper task_before as well as runtime historical state.
    env.env._last_task = env.get_task_state()
    return env


class CollisionGeometryTests(unittest.TestCase):
    def test_closed_geometry_cases(self):
        lower, upper = [0, 0, 0], [1, 1, 1]
        cases = [([-1,.5,.5],[2,.5,.5],True), ([-1,1,.5],[2,1,.5],True),
                 ([-1,.5,.5],[0,.5,.5],True), ([0,.5,.5],[0,.5,.5],True),
                 ([.5,.5,.5],[.5,.5,.5],True), ([-1,2,.5],[2,2,.5],False),
                 ([-1,1+5e-10,.5],[2,1+5e-10,.5],True),
                 ([-1,1+1e-7,.5],[2,1+1e-7,.5],False)]
        for start, end, expected in cases:
            with self.subTest(start=start, end=end):
                self.assertEqual(hit(start,end,lower,upper) is not None, expected)

    def test_invalid_geometry_rejected(self):
        with self.assertRaises(ValueError):
            hit([float("nan"),0,0],[1,1,1],[0,0,0],[1,1,1])


class CollisionEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_every_agent_and_phase_ends_team_with_final_reward(self):
        for agent in range(4):
            for phase in ("Search", "Intercept", "Hold"):
                with self.subTest(agent=agent, phase=phase):
                    env = collision_env(agent, phase)
                    rt = env.unwrapped
                    with patch.object(rt, "_maybe_detect_swept", side_effect=AssertionError("postcollision Found")), patch.object(rt, "_update_capture", side_effect=AssertionError("postcollision success")):
                        obs, rewards, dones = env.step(torch.zeros(4,3))
                    self.assertEqual(dones, [True]*4)
                    self.assertEqual(torch.as_tensor(rewards).tolist(), [-2.]*4)
                    result = rt.get_episode_result()
                    self.assertEqual(result["termination_reason"], "obstacle_collision")
                    self.assertFalse(result["success"])
                    self.assertEqual(result["first_collision_agent_ids"], [agent])
                    self.assertEqual(result["first_collision_phase"], phase)
                    self.assertEqual(bool(rt.task_found), phase != "Search")
                    self.assertEqual(env.last_reward_breakdown["final_reward_by_agent"], [-2.]*4)
                    self.assertEqual(env.reward_adapter.terminal_bonus_count, 0)
                    self.assertTrue(all(tuple(o.shape)==(28,) and torch.isfinite(torch.as_tensor(o)).all() for o in obs))
                    with self.assertRaisesRegex(RuntimeError, "reset"):
                        env.step(torch.zeros(4,3))
                    env.close()

    def test_segment_tunnelling_and_legacy_endpoint_behavior(self):
        for protocol in (STRICT, LEGACY):
            env=collision_env(protocol=protocol)
            _, _, dones=env.step(torch.zeros(4,3))
            self.assertEqual(bool(env.unwrapped._collision_flags[0]), protocol==STRICT)
            if protocol == STRICT:
                record=env.unwrapped.last_collision_records[0]
                self.assertGreater(record["candidate_position"][0], record["first_hit_point"][0])
                self.assertLess(record["hit_fraction"],1.)
            else:
                self.assertFalse(all(dones))

    def test_adapter_disabled_still_overrides(self):
        env=collision_env(3,"Hold",False)
        self.assertEqual(torch.as_tensor(env.step(torch.zeros(4,3))[1]).tolist(),[-2.]*4)

    def test_frozen_agent_has_no_phantom_collision(self):
        env=collision_env(0)
        env.unwrapped.agent_finished[0]=True
        before=env.unwrapped._agent_pos[0].clone()
        env.step(torch.zeros(4,3))
        self.assertFalse(env.unwrapped._collision_flags.any())
        self.assertTrue(torch.equal(before,env.unwrapped._agent_pos[0]))

    def test_collision_beats_deadline(self):
        env=collision_env(3)
        env.unwrapped.max_steps=1
        env.step(torch.zeros(4,3))
        self.assertEqual(env.unwrapped.get_episode_result()["termination_reason"],"obstacle_collision")

    def test_simultaneous_agents_one_team_failure_and_no_success_event(self):
        env=collision_env(0)
        rt=env.unwrapped
        rt.obstacles.append({"center":rt._agent_pos[3].tolist(),"size":[.01]*3})
        rt._build_obstacle_tensors()
        with patch.object(rt,"_update_capture",side_effect=AssertionError("collision must precede success check")):
            env.step(torch.zeros(4,3))
        self.assertEqual(rt.get_episode_result()["first_collision_agent_ids"],[0,3])
        self.assertFalse(rt.mission_complete)

    def test_target_contact_and_separation_are_not_obstacle_collision(self):
        env=collision_env(3)
        rt=env.unwrapped
        rt.obstacles=[]
        rt.ground_truth_obstacles=[]
        rt._build_obstacle_tensors()
        rt.task_found=True
        rt.executor_target_assigned=True
        rt.target_capture_hold_steps=1
        rt.target_state.motion_mode="static"
        rt._task_target.copy_(rt._agent_pos[3])
        rt._agent_vel.zero_()
        rt._agent_pos[2].copy_(rt._agent_pos[3])
        rt._capture_hold_counter=0
        env.env._last_task=env.get_task_state()
        env.step(torch.zeros(4,3))
        self.assertFalse(rt._collision_flags.any())
        self.assertEqual(rt.get_episode_result()["termination_reason"],"success")
        self.assertEqual(env.reward_adapter.terminal_bonus_count,1)
        with self.assertRaisesRegex(RuntimeError,"reset"):
            env.step(torch.zeros(4,3))
        self.assertEqual(env.reward_adapter.terminal_bonus_count,1)

    def test_terminal_reward_configuration_is_validated(self):
        from core.env.task_protocol import validate_task_config
        for value in (0.,1.,float("inf"),float("nan")):
            with self.assertRaises(ValueError):
                validate_task_config({"task_protocol":STRICT,"collision_terminal_reward":value})
        with self.assertRaisesRegex(ValueError,"reward_clip"):
            validate_task_config({"task_protocol":STRICT,"reward":{"reward_clip":1.}})
        config = strict_config()
        config["profile"] = "S00_STATIC_CLEAR"
        config["collision_terminal_reward"] = -4.0
        config["reward"]["reward_clip"] = 5.0
        validate_task_config(config)
        env = evaluator._make_env(config, config["reward"])
        self.assertEqual(env.unwrapped.collision_terminal_reward, -4.0)
        env.close()

    def test_deadline_success_and_timeout(self):
        for success in (False,True):
            env=collision_env(3)
            rt=env.unwrapped
            rt.obstacles=[]
            rt.ground_truth_obstacles=[]
            rt._build_obstacle_tensors()
            rt.max_steps=1
            def capture(*args):
                if success:
                    rt.task_found=True
                    rt.found_step=1
                    rt.mission_complete=True
                    rt._mission_complete_event=True
                    rt.success_step=1
            with patch.object(rt,"_update_capture",side_effect=capture):
                env.step(torch.zeros(4,3))
            result=rt.get_episode_result()
            self.assertEqual(result["termination_reason"],"success" if success else "timeout")
            self.assertEqual(result["bootstrap_mask"],0.)

    def test_invalid_reset_has_explicit_diagnostic(self):
        config=strict_config()
        scenarios,_=evaluator._build_evaluation_manifest({**config,"evaluation_episodes":1,"scenario_seed":1729,"split":"validation"})
        scenario=copy.deepcopy(scenarios[0])
        obstacle=scenario["obstacles"][0]
        scenario["initial_agent_positions"][0]=obstacle["center"]
        env=evaluator._make_env(config,config["reward"])
        with self.assertRaisesRegex(ValueError,"invalid initial state"):
            env.reset(scenario=scenario)
        self.assertIsNotNone(env.unwrapped.invalid_initial_state)


class CollisionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_critic_terminal_nan_next_state_never_evaluated(self):
        learner=trainer.PRRACMADDPG(architecture=ARCHITECTURE,loss=LOSS)
        data={"rewards":[torch.full((2,1),-2.)]*4,"dones":[torch.ones(2,1)]*4,
              "next_obs":[torch.full((2,28),float('nan'))]*4,"stage_after":torch.zeros(2,dtype=torch.long)}
        with patch.object(learner.agents[0].target_actor,"forward",side_effect=AssertionError("terminal bootstrap")):
            self.assertTrue(torch.equal(learner._target_q(learner.agents[0],data,0),torch.full((2,1),-2.)))

    def test_collector_preserves_terminal_transition_and_does_not_replan(self):
        config=strict_config()
        config["architecture"]=ARCHITECTURE
        config["loss"]=LOSS
        learner=trainer.PRRACMADDPG(architecture=ARCHITECTURE,loss=LOSS)
        scenarios,_=evaluator._build_evaluation_manifest({**config,"evaluation_episodes":1,"scenario_seed":1729,"split":"validation"})
        job={**config,"episode_index":0,"scenario":scenarios[0],"policy_snapshot":learner.policy_snapshot(),"search_value_snapshot":None}
        original=trainer.PRRACTrainingEnv.step
        returned=[]
        def step(env, actions):
            rt=env.unwrapped
            rt.obstacles=[{"center":rt._agent_pos[0].tolist(),"size":[.001]*3}]
            rt._build_obstacle_tensors()
            result=original(env,actions)
            returned.append(result)
            return result
        with patch.object(trainer.PRRACTrainingEnv,"step",step), patch("chapter3_bser.online.controller.OnlineBSERController.step",side_effect=AssertionError("terminal replan")):
            metrics,transitions,*_=trainer._collect_episode(job)
        self.assertEqual(len(transitions),1)
        self.assertEqual(list(transitions[0][2]),[-2.]*4)
        self.assertEqual(transitions[0][4],(True,)*4)
        self.assertEqual(transitions[0][5],(False,)*4)
        for actual,expected in zip(transitions[0][3],returned[0][0]):
            np.testing.assert_array_equal(actual,np.asarray(expected))
        replay=trainer.PRRACReplayAdapter(8,task_config=config)
        trainer._apply_transitions(learner,replay,transitions,{"episode_id":0,"success":False},
            {"warmup_steps":100,"update_frequency":1,"batch_size":4},global_step=0,update_step=0,device="cpu")
        self.assertEqual(len(replay),1)
        self.assertEqual(replay.phase_counts()["success_tail"],0)
        with self.assertRaisesRegex(ValueError,"protocol"):
            trainer.PRRACReplayAdapter(8).load_state_dict(replay.state_dict())

    def test_transfer_actor_only_and_resume_rejected(self):
        config=strict_config()
        config["architecture"],config["loss"]=ARCHITECTURE,LOSS
        payload=checkpoint_payload()
        source=copy.deepcopy(config)
        for field in ("task_protocol","collision_detection_revision","terminal_reward_revision","collision_terminal_reward"):
            source.pop(field,None)
        payload["resolved_training_config"]=source
        payload["metadata"]["config_hash"]=trainer._config_hash(source)
        with tempfile.TemporaryDirectory() as directory:
            path=write_checkpoint(Path(directory)/"source.pt",payload)
            learner=trainer.PRRACMADDPG(architecture=ARCHITECTURE,loss=LOSS)
            before=copy.deepcopy(learner.agents[0].critic1.state_dict())
            with self.assertRaisesRegex(ValueError,"protocol"):
                trainer._load_checkpoint(path,learner,trainer.PRRACReplayAdapter(8,task_config=config),config)
            with self.assertRaisesRegex(ValueError,"protocol"):
                evaluator.load_prrac_checkpoint(path,config=config)
            evaluator.load_prrac_checkpoint(path,config={**config,"allow_protocol_transfer":True})
            lineage=import_actors(path,learner,config)
            self.assertEqual(lineage["source_training_protocol"],LEGACY)
            self.assertIsNone(lineage["source_wall_seconds"])
            self.assertEqual(learner.niter,0)
            self.assertEqual(len(learner.agents[0].actor_optimizer.state),0)
            for key,value in before.items():
                self.assertTrue(torch.equal(value,learner.agents[0].critic1.state_dict()[key]))
            for agent,stored in zip(learner.agents,payload["prrac_training_state"]["agents"]):
                for key,value in stored["actor"].items():
                    self.assertTrue(torch.equal(value,agent.actor.state_dict()[key]))
                    self.assertTrue(torch.equal(value,agent.target_actor.state_dict()[key]))

    def test_cache_and_metric_missingness(self):
        info={"checkpoint":"synthetic.pt","checkpoint_config_hash":"hash","checkpoint_episode":1,"evaluation_mode":"full_prrac",
              "checkpoint_runtime_revision":"dynamic_public_intercept_v2_1","runtime_integration_mode":"legacy"}
        self.assertNotEqual(evaluator._combo_key(info,"manifest"),evaluator._combo_key({**info,**protocol_identity({"task_protocol":STRICT})},"manifest"))
        row={**protocol_identity({"task_protocol":STRICT}),"terminated":True,"truncated":False,
             "termination_reason":"obstacle_collision","success":False,"safe_success":False,"found":False,
             "collision_episode":True,"first_collision_agent_ids":[0,1,3],"first_collision_phase":"Search"}
        result=aggregate_task_outcomes([row])
        self.assertIsNone(result["success_if_found_rate"])
        self.assertEqual(result["n_collision_failure"],1)
        self.assertEqual(result["first_collision_by_role"],{"Searcher":1,"Executor":1})
        missing=aggregate_task_outcomes([row],2)
        self.assertFalse(missing["evaluation_complete"])
        self.assertIsNone(missing["safe_success_rate"])

    def test_critic_warmup_freezes_actors_and_updates_critics(self):
        from tests.prrac_evaluation_support import _transitions
        learner=trainer.PRRACMADDPG(architecture=ARCHITECTURE,loss=LOSS)
        before=copy.deepcopy(learner.agents[0].actor.state_dict())
        critic=copy.deepcopy(learner.agents[0].critic1.state_dict())
        replay=trainer.PRRACReplayAdapter(32)
        result=trainer._apply_transitions(learner,replay,_transitions(),{"episode_id":0,"success":False},
            {"warmup_steps":4,"update_frequency":1,"batch_size":4,"updates_per_train":1,
             "policy_delay":1,"critic_only_warmup_updates":256},global_step=0,update_step=0,device="cpu")
        self.assertEqual(result["actor_update_count"],0)
        self.assertGreater(result["optimizer_update_count"],0)
        self.assertTrue(all(torch.equal(v,learner.agents[0].actor.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(any(not torch.equal(v,learner.agents[0].critic1.state_dict()[k]) for k,v in critic.items()))

    def test_100_pair_plan_only_and_protocol_provenance(self):
        from chapter3_bser.experiments.phase1c_prrac.paired_evaluation import prepare_pair
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"opaque_source.pt"
            path.write_bytes(b"plan fixture only")
            root,plan=prepare_pair(checkpoint=path,
                config_path=ROOT/"configs/chapter3/bser_phase1c_prrac_collision_terminal_eval.json",
                episodes=100,workers=4,output_root=Path(directory)/"collision_terminal",allow_protocol_transfer=True)
            self.assertEqual(plan["total_planned_episodes"],200)
            self.assertFalse(plan["execution_started"])
            config=json.loads(Path(plan["shared_config"]).read_text())
            self.assertEqual(config["task_protocol"],STRICT)
            self.assertTrue(config["allow_protocol_transfer"])

    def test_strict_checkpoint_roundtrip_preserves_identity_and_replay(self):
        config=strict_config()
        config["architecture"],config["loss"]=ARCHITECTURE,LOSS
        config["rl"]["replay_size"]=8
        learner,replay=trainer._build_learner(config)
        with tempfile.TemporaryDirectory() as directory:
            path=trainer._save_checkpoint(learner,replay,Path(directory),config,1,
                global_step=0,update_step=0,replay_sample_count=0,optimizer_update_count=0,
                episode_rows=[],execution_rows=[],prrac_rows=[])
            self.assertTrue(trainer._verify_checkpoint_roundtrip(path,config))
            payload=torch.load(path,weights_only=True)
            self.assertEqual(payload["metadata"]["task_protocol"],STRICT)
            self.assertEqual(payload["prrac_replay_state"]["task_protocol"],STRICT)
            self.assertEqual(payload["resolved_training_config"],config)

    def test_real_warmstart_training_summary_and_same_protocol_resume(self):
        from tests.prrac_evaluation_support import _ImmediateExecutor
        class RealImmediateExecutor(_ImmediateExecutor):
            def map(self, function, jobs):
                return [function(job) for job in jobs]
        config = strict_config()
        config.update(architecture=ARCHITECTURE, loss=LOSS, episodes=2,
                      max_steps=1, workers=1, checkpoint_interval=1)
        config["rl"].update(replay_size=8, batch_size=1, warmup_steps=0,
                            update_frequency=1, updates_per_train=1)
        source = copy.deepcopy(config)
        for field in ("task_protocol", "collision_detection_revision", "terminal_reward_revision", "collision_terminal_reward"):
            source.pop(field, None)
        payload = checkpoint_payload()
        payload["resolved_training_config"] = source
        payload["metadata"]["config_hash"] = trainer._config_hash(source)
        original = trainer.PRRACTrainingEnv.step
        def collide(env, actions):
            rt = env.unwrapped
            rt.obstacles = [{"center": rt._agent_pos[0].tolist(), "size": [.001]*3}]
            rt._build_obstacle_tensors()
            return original(env, actions)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = write_checkpoint(root / "source.pt", payload)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "collision_terminal" / "warmstart"
            with patch.object(trainer, "ProcessPoolExecutor", RealImmediateExecutor), patch.object(trainer.PRRACTrainingEnv, "step", collide):
                summary = trainer.run_training(config_path=config_path, output_dir=output,
                    init_actors_from=source_path, critic_warmup_updates=256)
                self.assertEqual(summary["global_step"], 2)
                self.assertEqual(summary["learner_update_rounds"], 2)
                self.assertEqual(summary["critic_only_warmup_updates_executed"], 2)
                self.assertEqual(summary["n_collision_failure"], 2)
                self.assertIsNone(summary["performance_passed"])
                final_path = output / "checkpoints" / "phase1c_prrac_episode_0002.pt"
                final = torch.load(final_path, weights_only=True)
                self.assertEqual(final["completed_episode"], 2)
                for agent, stored in zip(final["prrac_training_state"]["agents"], payload["prrac_training_state"]["agents"]):
                    for key, value in stored["actor"].items():
                        self.assertTrue(torch.equal(agent["actor"][key], value))
                with self.assertRaises(FileExistsError):
                    trainer.run_training(config_path=config_path, output_dir=output)
                # Retain the completed run: resuming its earlier checkpoint must
                # use a new directory rather than overwrite episode 2 weights.
                resumed = trainer.run_training(config_path=config_path, output_dir=root / "collision_terminal" / "resumed",
                    resume=output / "checkpoints" / "phase1c_prrac_episode_0001.pt")
            self.assertEqual(resumed["global_step"], 2)
            self.assertEqual(resumed["learner_update_rounds"], 2)
            self.assertEqual(resumed["n_valid_episodes"], 2)
            self.assertEqual(resumed["initialization"]["source_training_protocol"], LEGACY)

    def test_mixed_batch_only_nonterminal_rows_bootstrap(self):
        learner=trainer.PRRACMADDPG(architecture=ARCHITECTURE,loss=LOSS)
        next_obs=[torch.zeros(2,28) for _ in range(4)]
        for value in next_obs:
            value[0]=float("nan")
        data={"rewards":[torch.full((2,1),-2.)]*4,"dones":[torch.tensor([[1.],[0.]])]*4,
              "next_obs":next_obs,"stage_after":torch.zeros(2,dtype=torch.long)}
        target=learner._target_q(learner.agents[0],data,0)
        self.assertTrue(torch.isfinite(target).all())
        self.assertEqual(float(target[0]),-2.)

    def test_strict_evaluation_writes_complete_report_and_resumes_cache(self):
        from tests.prrac_evaluation_support import _ImmediateExecutor
        class RealImmediateExecutor(_ImmediateExecutor):
            def map(self, function, jobs):
                return [function(job) for job in jobs]
        config=evaluator._load_config(ROOT/"configs/chapter3/bser_phase1c_prrac_collision_terminal_eval.json")
        config.update(evaluation_episodes=1,max_steps=1,workers=1,allow_protocol_transfer=True)
        source=strict_config()
        source["architecture"],source["loss"]=ARCHITECTURE,LOSS
        for field in ("task_protocol","collision_detection_revision","terminal_reward_revision","collision_terminal_reward"):
            source.pop(field,None)
        payload=checkpoint_payload()
        payload["metadata"]["config_hash"]=trainer._config_hash(source)
        payload["resolved_training_config"]=source
        original=evaluator.PRRACTrainingEnv.step
        def step(env,actions):
            rt=env.unwrapped
            rt.obstacles=[{"center":rt._agent_pos[0].tolist(),"size":[.001]*3}]
            rt._build_obstacle_tensors()
            return original(env,actions)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            ckpt=write_checkpoint(root/"source.pt",payload)
            config_file=root/"evaluation.json"
            config_file.write_text(json.dumps(config),encoding="utf-8")
            with patch.object(evaluator,"ProcessPoolExecutor",RealImmediateExecutor), patch.object(evaluator.PRRACTrainingEnv,"step",step):
                summary=evaluator.run_evaluation(config_path=config_file,checkpoints=[ckpt],
                    output_dir=root/"collision_terminal"/"evaluation")
            outcome=summary["outcome_summaries"][0]
            self.assertEqual(outcome["n_valid_episodes"],1)
            self.assertEqual(outcome["collision_failure_rate"],1.)
            self.assertEqual(outcome["checkpoint_task_protocol"],LEGACY)
            self.assertEqual(outcome["evaluation_task_protocol"],STRICT)
            with patch.object(evaluator,"_evaluate_episode_job",side_effect=AssertionError("cached episode rerun")):
                evaluator.run_evaluation(config_path=config_file,checkpoints=[ckpt],
                    output_dir=root/"collision_terminal"/"evaluation",resume_evaluation=True)
            cache = root/"collision_terminal"/"evaluation"/"episode_evaluation.csv"
            rows = evaluator._read_csv(cache)
            for reason in (None, "unknown_cached_reason"):
                evaluator._write_csv(cache, [dict(row, termination_reason=reason) for row in rows])
                with patch.object(evaluator,"_evaluate_episode_job",side_effect=AssertionError("invalid cache rerun")):
                    with self.assertRaises(ValueError):
                        evaluator.run_evaluation(config_path=config_file,checkpoints=[ckpt],
                            output_dir=root/"collision_terminal"/"evaluation",resume_evaluation=True)

    def test_existing_strict_failure_and_plot_artifacts_are_protected(self):
        for name in ("evaluation_failure.json", "incomplete_episode_evaluation.csv", "collision_terminal_outcomes.png"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "collision_terminal" / "evaluation"
                output.mkdir(parents=True)
                retained = output / name
                retained.write_bytes(b"retained artifact")
                with patch.object(evaluator, "_resolve_checkpoints", side_effect=AssertionError("must reject before checkpoint load")):
                    with self.assertRaises(FileExistsError):
                        evaluator.run_evaluation(
                            config_path=ROOT / "configs/chapter3/bser_phase1c_prrac_collision_terminal_eval.json",
                            checkpoints=[root / "source.pt"], output_dir=output)
                self.assertEqual(retained.read_bytes(), b"retained artifact")

    def test_strict_evaluator_terminal_never_calls_actor_for_prior_only(self):
        with tempfile.TemporaryDirectory() as directory:
            job=worker_jobs(write_checkpoint(Path(directory)/"source.pt"),1)[0]
            job["config"].update(protocol_identity({"task_protocol":STRICT}))
            job["config"]["collision_terminal_reward"]=-2.
            job["config"]["controller"]="prior_only"
            original=evaluator.PRRACTrainingEnv.step
            def step(env,actions):
                self.assertTrue(torch.equal(actions,torch.zeros(4,3)))
                rt=env.unwrapped
                rt.obstacles=[{"center":rt._agent_pos[0].tolist(),"size":[.001]*3}]
                rt._build_obstacle_tensors()
                return original(env,actions)
            with patch.object(evaluator.PRRACTrainingEnv,"step",step), patch.object(evaluator,"_policy_outputs",side_effect=AssertionError("Prior-only actor")), patch("chapter3_bser.online.controller.OnlineBSERController.step",side_effect=AssertionError("terminal replan")):
                row=evaluator._evaluate_episode_job(job)["episode"]
            self.assertEqual(row["task_protocol"],STRICT)
            self.assertEqual(row["termination_reason"],"obstacle_collision")
            self.assertFalse(row["success"])
            self.assertTrue(row["protocol_transfer_evaluation"])


if __name__=="__main__":
    unittest.main()
