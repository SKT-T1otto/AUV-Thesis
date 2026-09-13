"""Fixed-policy HGR-family evaluation, with strict shared mission accounting."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import time
import traceback

import torch

from core.config.ch3_constants import ALL_SCENARIO_PROFILES
from core.config.ch3_config import build_ch3_config
from core.env.uav_env import UAVEnv
from core.env.mission_env import environment_kwargs_from_config
from core.env.task_protocol import protocol_identity
from core.scenarios.ch3_generator_impl import build_scenario_manifests
from chapter3_bser.experiments.reward_objective import objective_identity
from chapter3_bser.experiments.phase1c_prrac.task_metrics import (
    strict_outcome, validated_rows, aggregate_task_outcomes,
)
from chapter3_bser.experiments.phase1c_prrac.runtime_factory import runtime_contract
from chapter3_bser.online.config import execution_runtime_config, load_phase1b2_config
from chapter3_bser.models.hgr.policy import HandoffPolicy, weights_hash
from .provenance import fresh_source_identity, require_source_match, checkout_identity
from .runtime import collect_trajectory
from .train import digest, load_checkpoint, validated_output, write_json


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# These fields affect training only. Everything else remains in the evaluation
# configuration so future environment switches cannot disappear from identity.
TRAINING_FIELDS = frozenset((
    'schema', 'output_dir', 'seed', 'scenario_manifest', 'scenario_manifest_sha256',
    'total_main_trajectories', 'main_prefix_batch_size', 'suffix_training_episodes_per_cycle',
    'pilot_prefix_episodes_per_cycle', 'correction_draws_per_cycle', 'checkpoint_interval',
    'max_total_environment_steps', 'prefix_lr', 'suffix_lr', 'predictor', 'ablation', 'rl',
))
METHOD_FIELDS = frozenset(('method', 'algorithm', 'architecture_version', 'checkpoint_schema', 'policy'))


def resolved_config(config, episodes, seed, policy_mode):
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError('episodes must be a positive integer')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - episodes:
        raise ValueError('seed and episode sampling seeds must fit nonnegative signed int64')
    if policy_mode not in ('stochastic', 'deterministic_mean'):
        raise ValueError('unknown policy_mode')
    if config.get('profile') not in ALL_SCENARIO_PROFILES:
        raise ValueError('unsupported evaluation profile')
    runtime_contract(config)
    value = {k: copy.deepcopy(v) for k, v in config.items() if k not in TRAINING_FIELDS}
    value.update(protocol_identity(config))
    value.update(objective_identity(config))
    value['base_candidate'] = config.get('base_candidate', 'ch3_v3_full_reference')
    # Mirror the facade's base-constructor introspection, without building an env.
    default_collision_reward = inspect.signature(UAVEnv.__mro__[1].__init__).parameters['collision_terminal_reward'].default
    value['collision_terminal_reward'] = float(config.get('collision_terminal_reward', default_collision_reward))
    environment = build_ch3_config(value['base_candidate'], config['profile'])
    environment.update(protocol_identity(config))
    environment.update(max_steps=config['max_steps'], collision_terminal_reward=value['collision_terminal_reward'])
    environment = environment_kwargs_from_config(environment, device='cpu',
                                                 max_steps=config['max_steps'], return_numpy=False)
    phase = load_phase1b2_config()
    phase_reference_sha = digest(phase)
    phase['execution_runtime'] = execution_runtime_config(config)
    # The online controller does not use the historical offline experiment plan.
    phase.pop('experiment', None)
    value.update(schema='hgr.evaluation.v1', episodes=episodes, seed=seed,
                 policy_mode=policy_mode, gamma=config['rl']['gamma'],
                 execution_runtime=phase['execution_runtime'], phase1b_config=phase,
                 phase1b_reference_config_sha256=phase_reference_sha,
                 environment_config=environment,
                 collector='chapter3_bser.experiments.hgr.runtime.collect_trajectory',
                 device='cpu', collection_mode='serial')
    return value


def evaluation_manifest(config, episodes, seed, manifest, source):
    if manifest is not None:
        path = Path(manifest).resolve()
        raw = path.read_bytes()
        original = json.loads(raw.decode('utf-8'))
        scenarios = original['scenarios']
        origin = dict(kind='external', original_path=str(path),
                      original_file_sha256=hashlib.sha256(raw).hexdigest(),
                      original_schema=original.get('schema', original.get('schema_version')),
                      generator_seed=None, available_count=len(scenarios),
                      fixture_purpose=original.get('purpose'))
    else:
        original = build_scenario_manifests(count=episodes, generator_seed=seed,
                    split='validation', profiles=[config['profile']])[config['profile']]
        scenarios = original['scenarios']
        origin = dict(kind='generated', generator='core.scenarios.ch3_generator_impl.build_scenario_manifests',
                      generator_source_sha256=source['files']['core/scenarios/ch3_generator_impl.py'],
                      production_source_sha256=source['sha256'], generator_seed=seed,
                      original_schema=original.get('schema'), generated_count=len(scenarios))
    if not isinstance(scenarios, list) or len(scenarios) < episodes:
        raise ValueError('evaluation manifest is too short or not a scenario list')
    selected = copy.deepcopy(scenarios[:episodes])
    seen = set()
    for scenario in selected:
        if not isinstance(scenario, dict):
            raise ValueError('evaluation scenario must be an object')
        name, scenario_seed = scenario.get('scenario_id'), scenario.get('scenario_seed')
        if not isinstance(name, str) or not name.strip() or name in seen:
            raise ValueError(f'missing or duplicate evaluation scenario identity: {name!r}')
        seen.add(name)
        if isinstance(scenario_seed, bool) or not isinstance(scenario_seed, int) or not 0 <= scenario_seed < 2**63:
            raise ValueError(f'invalid scenario_seed: {name}')
        if (scenario.get('scenario_split') != 'validation'
                or scenario.get('scenario_role', 'validation') != 'validation'):
            raise ValueError(f'evaluation requires validation scenario labels: {name}')
        if scenario.get('scenario_profile') != config['profile']:
            raise ValueError(f'incompatible evaluation scenario profile: {name}')
        horizon = scenario.get('max_steps')
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < config['max_steps']:
            raise ValueError(f'incompatible evaluation scenario horizon: {name}')
    return dict(schema='hgr.evaluation_manifest.v1', origin=origin, split='validation',
                profile=config['profile'], selected_count=len(selected), scenarios=selected,
                selected_content_sha256=digest(selected),
                split_validation_scope='labels and unique identities within this run; no global training-overlap proof')


def evaluation_identity(resolved, selected, checkpoint_sha, policy_hash, source):
    streams = dict(scheme='policy_sampling_seed=seed+evaluation_episode_index; zero based',
                   policy_mode=resolved['policy_mode'], generator_seed=selected['origin']['generator_seed'],
                   episodes=[dict(evaluation_episode_index=i, scenario_id=s['scenario_id'],
                                  scenario_seed=s['scenario_seed'], policy_sampling_seed=resolved['seed']+i)
                             for i, s in enumerate(selected['scenarios'])])
    conditions = {k: v for k, v in resolved.items() if k not in METHOD_FIELDS}
    pairing = dict(task_conditions=conditions, selected_content_sha256=selected['selected_content_sha256'],
                   random_streams=streams)
    canonical = dict(checkpoint_file_sha256=checkpoint_sha, policy_hash=policy_hash,
                     production_source_sha256=source['sha256'], resolved_evaluation_config=resolved,
                     selected_content_sha256=selected['selected_content_sha256'], random_streams=streams)
    return dict(schema='hgr.evaluation_identity.v1', canonical=canonical, sha256=digest(canonical),
                pairing_conditions=pairing, pairing_sha256=digest(pairing),
                pairing_scope='same task/scenes/streams/mode; method and checkpoint may differ',
                source_identity=source)


def episode_row(trajectory, config, scenario, *, index, sampling_seed, policy_mode,
                checkpoint, policy_hash, wall_seconds):
    row = dict(trajectory['summary'])
    length = row.get('actual_length')
    records = trajectory['records']
    if (isinstance(length, bool) or not isinstance(length, int) or not 0 < length <= config['max_steps']
            or length != len(records) or ('episode_length' in row and row['episode_length'] != length)):
        raise ValueError('actual task clock and collector episode_length disagree')
    if row.get('scenario_id') != scenario['scenario_id'] or row.get('scenario_seed') != scenario['scenario_seed']:
        raise ValueError('collector scenario identity mismatch')
    for identity in (protocol_identity, objective_identity):
        if identity(row) != identity(config):
            raise ValueError('collector task/objective identity mismatch')
    if row.get('gamma') != config['rl']['gamma']:
        raise ValueError('collector gamma mismatch')
    for key in ('task_protocol', 'collision_detection_revision', 'terminal_reward_revision',
                'reward_objective', 'source_reward_revision',
                'first_collision_step', 'first_collision_agent_ids', 'first_collision_phase',
                'found_step', 'handoff_event_step', 'handoff_decision_step', 'source_individual_returns'):
        if key not in row:
            raise ValueError(f'collector missing authoritative field: {key}')
    for flag, counter in (('contact_episode', 'capture_contact_step_count'),
                          ('hold_episode', 'capture_full_hold_step_count')):
        count = row.get(counter)
        if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
            raise ValueError(f'invalid authoritative event counter: {counter}')
        row[flag] = None if count is None else count > 0
    for key in ('team_discounted_return', 'team_undiscounted_return'):
        if not isinstance(row.get(key), (float, int)) or not math.isfinite(row[key]):
            raise ValueError(f'missing or nonfinite {key}')
    # Preserve the authoritative returns; this adapter adds no reward transform.
    row.update(evaluation_episode_index=index, policy_sampling_seed=sampling_seed,
               method=config['method'], policy_mode=policy_mode, checkpoint=str(checkpoint),
               policy_hash=policy_hash, episode_length=length, wall_seconds=wall_seconds,
               optimizer_update_count=0)
    if strict_outcome(row) == 'INCOMPLETE':
        raise ValueError('collector returned an incomplete strict task outcome')
    normalized = validated_rows([row], require_complete=True)[0]
    # Reject an abnormal row before it can enter the persisted valid prefix.
    json.dumps(normalized, allow_nan=False)
    return normalized


def evaluation_summary(rows, resolved, *, finalized, wall_seconds):
    rows = validated_rows(rows, require_complete=True)
    result = aggregate_task_outcomes(rows, expected_episodes=resolved['episodes'])
    complete = result['evaluation_complete'] and finalized
    if not complete:
        for key in result:
            if key.endswith('_rate'):
                result[key] = None
    result.update(schema='hgr.evaluation_summary.v1', evaluation_complete=complete,
                  method=resolved['method'], policy_mode=resolved['policy_mode'], episodes=len(rows),
                  run_mode='same_method_evaluation', reward_objective=resolved['reward_objective'],
                  gamma=resolved['gamma'], training_update=False, actual_training_updates=0,
                  wall_seconds=wall_seconds, performance_claims_supported=False)
    for key in ('team_discounted_return', 'team_undiscounted_return'):
        result['mean_'+key] = sum(r[key] for r in rows)/len(rows) if complete else None
    return result


def evaluate(checkpoint, output, *, episodes=100, seed=12729, policy_mode='stochastic', manifest=None):
    checkpoint = Path(checkpoint).resolve()
    checkpoint_sha = file_sha256(checkpoint)
    payload = load_checkpoint(checkpoint)
    config = copy.deepcopy(payload['config'])
    output = validated_output(config, output)
    resolved = resolved_config(config, episodes, seed, policy_mode)
    source = fresh_source_identity()
    require_source_match(payload['source_identity'], source)
    selected = evaluation_manifest(config, episodes, seed, manifest, source)
    # Temporary network construction must not consume the caller's RNG state.
    with torch.random.fork_rng():
        policy = HandoffPolicy(config['policy'])
    policy.load_state_dict(payload['policy'], strict=True)
    policy.eval()
    policy.requires_grad_(False)
    policy_hash = weights_hash(policy)  # all state_dict entries, including buffers
    identity = evaluation_identity(resolved, selected, checkpoint_sha, policy_hash, source)
    identity.update(checkpoint_path=str(checkpoint), checkout_before=checkout_identity(),
                    checkpoint_training=dict(config=payload['config'], config_hash=payload['config_hash'],
                        costs=payload['costs'], completed_main=payload['completed_main'], cycle=payload['cycle']))
    documents = {'resolved_evaluation_config.json': resolved, 'evaluation_manifest.json': selected,
                 'evaluation_identity.json': identity}
    frozen_inputs = digest(dict(config=config, resolved=resolved, selected=selected, identity=identity))

    def verify_inputs():
        if file_sha256(checkpoint) != checkpoint_sha:
            raise RuntimeError('checkpoint file changed during evaluation')
        if weights_hash(policy) != policy_hash:
            raise RuntimeError('evaluation mutated policy parameters or buffers')
        require_source_match(source, fresh_source_identity(), context='evaluation end')
        if resolved_config(config, episodes, seed, policy_mode) != resolved:
            raise RuntimeError('effective runtime configuration changed during evaluation')
        if frozen_inputs != digest(dict(config=config, resolved=resolved, selected=selected, identity=identity)):
            raise RuntimeError('evaluation input content changed during evaluation')
        if manifest is not None and file_sha256(selected['origin']['original_path']) != selected['origin']['original_file_sha256']:
            raise RuntimeError('external manifest file changed during evaluation')

    verify_inputs()
    # Recheck immediately before the first write. Existing empty directories are allowed.
    validated_output(config, output)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in documents.items():
        write_json(output/name, value)
    document_hashes = {name: file_sha256(output/name) for name in documents}
    rows = []
    started = time.perf_counter()

    def save_progress(status, *, finalized=False):
        summary = evaluation_summary(rows, resolved, finalized=finalized, wall_seconds=time.perf_counter()-started)
        summary['evaluation_identity_sha256'] = identity['sha256']
        summary['pairing_sha256'] = identity['pairing_sha256']
        write_json(output/'episodes.json', rows)
        write_json(output/'summary.json', summary)
        write_json(output/'evaluation_progress.json', dict(schema='hgr.evaluation_progress.v1', status=status,
                   evaluation_identity_sha256=identity['sha256'], n_expected_episodes=episodes,
                   n_valid_episodes=len(rows), evaluation_complete=summary['evaluation_complete'],
                   next_episode_index=len(rows), checkout_before=identity['checkout_before'],
                   checkout_after=checkout_identity() if finalized or status == 'failed' else None))
        return summary

    save_progress('running')
    index, scenario, phase = None, None, 'collection'
    try:
        for index, scenario in enumerate(selected['scenarios']):
            episode_started = time.perf_counter()
            trajectory = collect_trajectory(config, scenario, policy, seed=seed+index, episode_id=index,
                                            deterministic=policy_mode == 'deterministic_mean')
            row = episode_row(trajectory, config, scenario, index=index, sampling_seed=seed+index,
                              policy_mode=policy_mode, checkpoint=checkpoint, policy_hash=policy_hash,
                              wall_seconds=time.perf_counter()-episode_started)
            verify_inputs()
            rows.append(row)
            save_progress('running')
        phase = 'final_input_verification'
        verify_inputs()
        if any(file_sha256(output/name) != value for name, value in document_hashes.items()):
            raise RuntimeError('saved evaluation identity/config/manifest changed during evaluation')
        return save_progress('complete', finalized=True)
    except (Exception, KeyboardInterrupt) as exc:
        write_json(output/'evaluation_failure.json', dict(schema='hgr.evaluation_failure.v1',
                   evaluation_complete=False, phase=phase, evaluation_episode_index=index,
                   scenario=scenario, exception_type=type(exc).__name__, message=str(exc),
                   traceback=traceback.format_exc()))
        save_progress('failed')
        raise
