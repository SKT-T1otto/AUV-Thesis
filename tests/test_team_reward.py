"""Final reward ordering and objective identities across production PRRAC paths."""
import copy
import json
from pathlib import Path
import unittest
import tempfile

import numpy as np
import torch

from chapter3_bser.experiments.reward_objective import RewardAccounting, TEAM, INDIVIDUAL
from chapter3_bser.experiments.phase1c_prrac import train_phase1c_prrac as train
from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluate
from chapter3_bser.experiments.phase1c_prrac.evaluation_provenance import validate_resume_config
from chapter3_bser.experiments.phase1c_prrac.task_metrics import validated_rows
from tests.prrac_evaluation_support import ARCHITECTURE, LOSS, checkpoint_payload


ROOT = Path(__file__).resolve().parents[1]


class TeamRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_final_source_mean_broadcast_and_discount_accounting(self):
        account = RewardAccounting(dict(reward_objective=TEAM), gamma=.95)
        np.testing.assert_array_equal(account.apply(np.array([0.,0.,0.,2.],dtype=np.float32),{}),[.5]*4)
        self.assertEqual(account.last["source_reward_by_agent"],[0.,0.,0.,2.])
        self.assertEqual(account.last["reward_transform_applied_count"],1)
        with self.assertRaisesRegex(RuntimeError,"already applied"):
            account.apply(torch.ones(4),account.last)
        torch.testing.assert_close(account.apply(torch.full((4,),-2.),{}),torch.full((4,),-2.))
        summary=account.summary()
        self.assertAlmostEqual(summary["team_undiscounted_return"],-1.5)
        self.assertAlmostEqual(summary["team_discounted_return"],.5-.95*2)
        self.assertEqual(summary["source_individual_returns"],[-2.,-2.,-2.,0.])
        individual=RewardAccounting()
        torch.testing.assert_close(individual.apply(torch.tensor([1.,2.,3.,4.]),{}),torch.tensor([1.,2.,3.,4.]))
        self.assertEqual(individual.identity["reward_objective"],INDIVIDUAL)

    def test_team_config_changes_only_objective_and_output(self):
        old=json.loads((ROOT/'configs/chapter3/bser_phase1c_prrac_collision_terminal_train.json').read_text())
        new=train._load_config(ROOT/'configs/chapter3/bser_phase1c_prrac_team_train.json')
        for key in ('rl','architecture','loss','reward','replay','execution_runtime'):
            self.assertEqual(new[key],old[key])
        self.assertNotEqual(train._config_hash(new),train._config_hash(old))
        for mode in ('train', 'eval'):
            individual=json.loads((ROOT/f'configs/chapter3/bser_phase1c_prrac_individual_{mode}.json').read_text(encoding='utf-8'))
            team=json.loads((ROOT/f'configs/chapter3/bser_phase1c_prrac_team_{mode}.json').read_text(encoding='utf-8'))
            self.assertEqual(individual.pop('reward_objective'),INDIVIDUAL)
            self.assertEqual(team.pop('reward_objective'),TEAM)
            individual.pop('output_dir');team.pop('output_dir')
            self.assertEqual(individual,team)

    def test_objective_transfer_is_explicit_and_evaluation_cache_rejects_mix(self):
        payload=checkpoint_payload()
        config=copy.deepcopy(evaluate._load_config(evaluate.DEFAULT_CONFIG))
        config["reward_objective"]=TEAM
        with self.assertRaisesRegex(ValueError,"reward objective"):
            evaluate._validate_checkpoint_payload(payload,config)
        config["allow_objective_transfer"]=True
        evaluate._validate_checkpoint_payload(payload,config)
        with self.assertRaisesRegex(ValueError,"reward objective"):
            validate_resume_config({},dict(reward_objective=TEAM))
        with self.assertRaisesRegex(ValueError,"reward objectives"):
            validated_rows([{},dict(reward_objective=TEAM)])

    def test_real_prrac_collector_replay_and_critic_use_final_team_signal(self):
        from chapter3_bser.models.hgr.policy import weights_hash
        config=train._load_config(ROOT/'configs/chapter3/bser_phase1c_prrac_team_train.json')
        config.update(architecture=copy.deepcopy(ARCHITECTURE),loss=copy.deepcopy(LOSS),max_steps=8)
        config['rl'].update(batch_size=2,replay_size=32,warmup_steps=0,update_frequency=1,updates_per_train=1)
        learner,replay=train._build_learner(config)
        before=weights_hash(learner.agents[0].critic1)
        scenario=json.loads((ROOT/'tests/fixtures/hgr/handoff_manifest.json').read_text(encoding='utf-8'))['scenarios'][0]
        job={**config,'scenario':scenario,'episode_index':0,'policy_snapshot':learner.policy_snapshot()}
        metrics,transitions,execution,_=train._collect_episode(job)
        self.assertTrue(metrics['found'])
        self.assertEqual(metrics['reward_objective'],TEAM)
        for transition in transitions:
            np.testing.assert_array_equal(transition[2],[transition[2][0]]*4)
            self.assertEqual(transition[6]['reward_objective'],TEAM)
        self.assertTrue(any(float(row[2][0]) != 0 for row in transitions[3:]))
        self.assertTrue(all(not any(row[4]) for row in transitions[:-1]))
        info=train._apply_transitions(learner,replay,transitions,metrics,config['rl'],global_step=0,update_step=0,device='cpu')
        self.assertGreater(info['optimizer_update_count'],0)
        self.assertNotEqual(before,weights_hash(learner.agents[0].critic1))
        batch=replay.sample(2,norm_rews=False,device='cpu')
        for rewards in batch.rewards[1:]:
            torch.testing.assert_close(rewards,batch.rewards[0],rtol=0,atol=0)
        _,roundtrip=train._build_learner(config)
        roundtrip.load_state_dict(replay.state_dict())
        wrong=copy.deepcopy(config);wrong.pop('reward_objective');wrong.pop('source_reward_revision')
        _,individual=train._build_learner(wrong)
        with self.assertRaisesRegex(ValueError,'reward objective'):
            individual.load_state_dict(replay.state_dict())
        self.assertAlmostEqual(metrics['reward'],metrics['team_undiscounted_return'])
        self.assertAlmostEqual(execution['adjusted_episode_reward'],metrics['team_undiscounted_return'])

    def test_real_checkpoint_actor_warmstart_and_hgr_mean_initialization(self):
        from chapter3_bser.experiments.phase1c_prrac.checkpoint_transfer import import_actors
        from chapter3_bser.experiments.hgr.train import Trainer, load_config
        from chapter3_bser.models.hgr.policy import weights_hash
        config=train._load_config(ROOT/'configs/chapter3/bser_phase1c_prrac_collision_terminal_train.json')
        config.update(architecture=copy.deepcopy(ARCHITECTURE),loss=copy.deepcopy(LOSS))
        config['rl']['replay_size']=8
        source,replay=train._build_learner(config)
        with tempfile.TemporaryDirectory(prefix='team-mean-import-') as temporary:
            directory=Path(temporary)/'collision_terminal'
            checkpoint=train._save_checkpoint(source,replay,directory/'source/checkpoints',config,1,
                global_step=0,update_step=0,replay_sample_count=0,optimizer_update_count=0,
                episode_rows=[],execution_rows=[],prrac_rows=[])
            team=copy.deepcopy(config); team['reward_objective']=TEAM
            target,target_replay=train._build_learner(team)
            critic_hash=weights_hash(target.agents[0].critic1)
            imported=import_actors(checkpoint,target,team)
            self.assertEqual(imported['source_reward_objective'],INDIVIDUAL)
            self.assertEqual(imported['target_reward_objective'],TEAM)
            self.assertEqual(critic_hash,weights_hash(target.agents[0].critic1))
            self.assertEqual(len(target_replay),0)
            with self.assertRaisesRegex(ValueError,'reward objective'):
                train._load_checkpoint(checkpoint,target,target_replay,team)
            hgr_config=load_config(ROOT/'tests/fixtures/hgr/integration_config.json')
            hgr=Trainer(hgr_config,directory/'hgr_mean',mean_initialization=checkpoint)
            self.assertEqual(hgr.initialization['mode'],'cross_architecture_mean_initialization')
            for i,agent in enumerate(source.agents):
                self.assertEqual(weights_hash(agent.actor),weights_hash(hgr.policy.theta_minus[i].mean_actor))
            self.assertEqual(weights_hash(source.agents[3].actor),weights_hash(hgr.policy.phi.mean_actor))
            hgr.policy.assert_isolated()


if __name__ == '__main__':
    unittest.main()
