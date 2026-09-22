"""Frozen independent MADDPG evaluation; no replay, updates or HGR policy."""
from __future__ import annotations

from pathlib import Path
import time
import traceback

import torch

from chapter3_bser.experiments.baselines.common.checkpoint import (
    load_checkpoint, load_model, state_digest, validated_output,
)
from chapter3_bser.experiments.baselines.common.runtime import BaselineMissionRuntime
from . import evaluate as task_evaluation
from .framework_provenance import framework_sources, verify_framework_sources
from .provenance import file_sha256, write_json


def evaluate(checkpoint, output, *, episodes=100, seed=12729, policy_mode="deterministic", manifest):
    if policy_mode != "deterministic":
        raise ValueError("independent MADDPG evaluation requires deterministic actors")
    payload = load_checkpoint(checkpoint)
    config = payload["config"]
    output = validated_output(config, output)
    sources = framework_sources()
    selected = task_evaluation.select_manifest(config, episodes, seed, manifest, sources)
    checkpoint_hash, manifest_hash = file_sha256(checkpoint), file_sha256(manifest)
    model = load_model(payload)
    model.prep_rollouts(device="cpu")
    frozen_state = state_digest(model.training_state_dict())
    rows, actual_steps = [], 0
    runtime, index, scenario = None, None, None
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)

    def verify():
        verify_framework_sources(sources)
        if file_sha256(checkpoint) != checkpoint_hash or file_sha256(manifest) != manifest_hash:
            raise ValueError("checkpoint or manifest changed during evaluation")
        if state_digest(model.training_state_dict()) != frozen_state:
            raise ValueError("evaluation changed model or optimizer state")

    def save(status, finalized=False):
        result = task_evaluation.summary(rows, episodes, finalized=finalized,
            wall_seconds=time.perf_counter()-start, actual_steps=actual_steps)
        result.update(method=config["method"], baseline=config["baseline"],
                      evaluation_policy_mode="deterministic", optimizer_update_count=0)
        write_json(output / "episodes.json", rows)
        write_json(output / "summary.json", result)
        write_json(output / "evaluation_progress.json", dict(status=status,
            completed_episodes=len(rows), actual_environment_steps=actual_steps,
            evaluation_complete=result["evaluation_complete"]))
        return result

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        save("running")
        for index, scenario in enumerate(selected["scenarios"]):
            verify()
            started = time.perf_counter()
            runtime = BaselineMissionRuntime(config, scenario, seed=seed+index, episode_id=index)
            while not runtime.terminal:
                before = runtime.step
                try:
                    runtime.advance(model, explore=False)
                finally:
                    actual_steps += runtime.step - before
            row = task_evaluation.episode_row(runtime, config, scenario, index, seed+index,
                                            time.perf_counter()-started)
            row.update(method=config["method"], baseline=config["baseline"], evaluation_policy_mode="deterministic")
            runtime.close()
            runtime = None
            verify()
            rows.append(row)
            save("running")
        verify()
        return save("complete", finalized=True)
    except (Exception, KeyboardInterrupt) as exc:
        write_json(output / "evaluation_failure.json", dict(evaluation_complete=False,
            evaluation_episode_index=index, scenario=scenario, message=str(exc),
            exception_type=type(exc).__name__, traceback=traceback.format_exc(), actual_environment_steps=actual_steps))
        save("failed")
        raise
    finally:
        if runtime is not None:
            runtime.close()
        torch.set_num_threads(previous_threads)
