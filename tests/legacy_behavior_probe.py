"""Copied unchanged into each isolated source snapshot; test weights only."""
import copy
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from tests.prrac_evaluation_support import worker_jobs, write_checkpoint
from tools.run_core_golden_e0 import _capture_state, _normal


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    torch.set_num_threads(1)
    torch.manual_seed(991)
    config = evaluator._load_config(evaluator.ROOT/'configs/chapter3/bser_phase1c_beds.json')
    config['max_steps'] = 4
    config['standby_diagnostics_enabled'] = False
    config.pop('early_discovery')
    config.pop('executor_standby')
    # No protocol fields is the historical legacy protocol; assert identities on
    # the current side separately, never pass unknown fields into old factories.
    assert 'task_protocol' not in config
    transitions = []
    with tempfile.TemporaryDirectory(prefix='test-actor-') as directory:
        job = worker_jobs(write_checkpoint(Path(directory)/'TEST_ONLY_seed991.pt'), 1)[0]
        job['config'] = config
        job['scenario']['max_steps'] = 4
        job['checkpoint_info'].update(checkpoint='TEST_ONLY_seed991.pt',
            checkpoint_runtime_revision='dynamic_public_intercept_v3_atomic_continuity',
            evaluation_runtime_revision='dynamic_public_intercept_v3_atomic_continuity',
            runtime_integration_mode='native', execution_variant='B1_ATOMIC_LAST_VALID',
            search_recovery_variant='S2A1_C2_LOCAL_CONNECTOR')
        original = evaluator.PRRACTrainingEnv.step
        original_reset = evaluator.PRRACTrainingEnv.reset
        initial = []
        def reset(env, *args, **kwargs):
            observations = original_reset(env, *args, **kwargs)
            initial.append(_normal(_capture_state(env.unwrapped, observations)))
            return observations
        def step(env, actions):
            runtime = env.unwrapped
            before = plain(runtime._agent_pos)
            result = original(env, actions)
            task = env.get_task_state()
            transitions.append(plain(dict(actions=actions, state_before=before,
                state_after=runtime._agent_pos, velocity_after=runtime._agent_vel,
                observations=result[0], rewards=result[1], dones=result[2],
                step=task.step, found=task.target_found, success=task.mission_complete,
                collision=runtime._collision_flags,
                full_state=_normal(_capture_state(runtime, result[0], result[1], result[2])),
                contact_steps=getattr(runtime, 'capture_contact_step_count', 0),
                hold_steps=getattr(runtime, 'capture_full_hold_step_count', 0))))
            return result
        with patch.object(evaluator.PRRACTrainingEnv, 'step', step), patch.object(evaluator.PRRACTrainingEnv, 'reset', reset):
            result = evaluator._evaluate_episode_job(copy.deepcopy(job))
        root = Path.cwd().resolve()
        project_modules = {}
        for name, module in tuple(sys.modules.items()):
            if name.split('.')[0] in {'core', 'chapter3_bser', 'tests'} and getattr(module, '__file__', None):
                path = Path(module.__file__).resolve()
                assert path.is_relative_to(root), (name, path, root)
                project_modules[name] = path.relative_to(root).as_posix()
        payload = dict(environment=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__),
            scenario=job['scenario'], config=job['config'], policy_snapshot=plain(job['policy_snapshot']),
            initial_state=initial, transitions=transitions, result=_normal(result),
            isolation=dict(root=str(root), project_modules=project_modules))
        if (root/'core/env/task_protocol.py').exists():
            from core.env.task_protocol import protocol_identity
            payload['protocol_identity'] = protocol_identity(config)
        Path(sys.argv[1]).write_text(json.dumps(payload, sort_keys=True, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
