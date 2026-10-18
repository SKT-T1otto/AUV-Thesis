"""Independent identity and an explicit physical-horizon discount contract."""
import copy
import math

METHOD = 'ch3_long_horizon_maddpg_v1'
SCHEMA = 'ch3.long_horizon_maddpg.v1'
ARCHITECTURE = 'ch3.long_horizon_maddpg.field_scaled.v1'


def discount_from_retention(horizon, retention):
    if type(horizon) is not int or horizon < 1:
        raise ValueError('discount horizon must be a positive integer')
    if not isinstance(retention, (int, float)) or not math.isfinite(retention) or not 0 < retention < 1:
        raise ValueError('retention must be strictly between zero and one')
    return math.exp(math.log(retention) / horizon)


def validate(config):
    from core.env.task_protocol import validate_task_config
    c = copy.deepcopy(config)
    if (c.get('schema') != SCHEMA or c.get('method') != METHOD
            or c.get('architecture_version') != ARCHITECTURE
            or c.get('algorithm') != 'long_horizon_maddpg'):
        raise ValueError('independent long-horizon method identity required')
    if (c.get('max_steps') != 400 or c.get('observation_dim') != 28
            or c.get('action_dim') != 3 or c.get('critic_input_dim') != 124
            or c.get('planner_protocol') != 'd2_v1'
            or c.get('task_protocol') != 'collision_terminal_v1'
            or c.get('reward_objective') != 'team_mean_v1'
            or c.get('source_reward_revision') != 'execution_reward_v2_after_terminal_override_v1'
            or c.get('base_candidate') != 'ch3_v3_full_reference'
            or c.get('profile') != 'M20_MOVING_UNKNOWN_MULTI'):
        raise ValueError('frozen D2 task and 28D/3D/124D contracts required')
    validate_task_config(c)
    if c.get('reward') != dict(schema='bser.phase1c.execution_reward.v2', enabled=True,
            freeze_searchers_after_found=True, preserve_discovery_reward=True,
            contact_entry_bonus=.25, hold_increment_bonus=.2, terminal_success_bonus=2.,
            reward_clip=3., executor_id=3, searcher_ids=[0, 1, 2]):
        raise ValueError('this repair does not silently redesign historical rewards')
    d = c['discount']
    if set(d) != {'horizon_steps', 'retention'} or d['horizon_steps'] != c['max_steps'] or d['retention'] < .5:
        raise ValueError('full-horizon reward retention must be at least one half')
    derived = discount_from_retention(d['horizon_steps'], d['retention'])
    rl = c['rl']
    if 'gamma' in rl and not math.isclose(rl['gamma'], derived, rel_tol=0., abs_tol=1e-12):
        raise ValueError('gamma and declared horizon retention disagree')
    rl['gamma'] = derived
    for name in ('batch_size', 'replay_size', 'update_frequency', 'updates_per_train', 'hidden_dim', 'actor_delay'):
        if type(rl[name]) is not int or rl[name] < 1:
            raise ValueError('invalid positive integer rl.' + name)
    for name in ('warmup_steps', 'critic_warmup_updates'):
        if type(rl[name]) is not int or rl[name] < 0:
            raise ValueError('invalid nonnegative integer rl.' + name)
    if rl['replay_size'] < rl['batch_size']:
        raise ValueError('replay must hold a minibatch')
    for name in ('lr_actor', 'lr_critic', 'tau', 'critic_grad_clip', 'actor_grad_clip'):
        if not isinstance(rl[name], (int, float)) or not math.isfinite(rl[name]) or rl[name] <= 0:
            raise ValueError('invalid positive finite rl.' + name)
    if rl['tau'] > 1 or rl.get('target_clip') is not None:
        raise ValueError('invalid tau or arbitrary value target clipping')
    for name in ('mask_inactive_actions', 'actor_use_replay_teammates'):
        if type(rl[name]) is not bool:
            raise ValueError('explicit boolean required: ' + name)
    if rl['input_scaling'] not in ('field_scale', 'legacy_layernorm'):
        raise ValueError('unknown network input scaling')
    noise = c['exploration']
    if (type(noise['decay_steps']) is not int or noise['decay_steps'] < 1
            or not 0 < noise['end_factor'] <= 1):
        raise ValueError('invalid exploration schedule')
    if type(c['seed']) is not int or c['seed'] < 0:
        raise ValueError('invalid seed')
    return c


def discount_report(config):
    c = validate(config)
    gamma = c['rl']['gamma']
    return dict(gamma=gamma, horizon_steps=400, horizon_retention=gamma**400,
                retention_by_steps={str(n): gamma**n for n in (20, 50, 100, 150, 200, 400)},
                previous_gamma=.95, previous_retention_150=.95**150,
                changed_optimization_objective=True,
                warning='Longer discount horizon does not prove fast credit propagation or better task performance.')
