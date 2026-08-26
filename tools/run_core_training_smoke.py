"""Run a bounded 2x10-step CPU training-closure smoke for the self-contained core."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from core.algorithms.maddpg import MADDPG
from core.runtime.builder import build_runtime


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments" / "chapter3" / "core_self_contained_training_smoke"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_smoke(output_dir: Path = DEFAULT_OUTPUT) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = 24680
    seed_all(seed)
    runtime = build_runtime(
        "ch3_v3_full_reference",
        "M00_MOVING_CLEAR",
        seed=seed,
        max_steps=10,
        device="cpu",
        replay_size=128,
    )
    env, maddpg, replay = runtime.env, runtime.maddpg, runtime.replay_buffer
    manifest = json.loads(
        (ROOT / "configs" / "scenarios" / "e0_equivalence" / "M00_MOVING_CLEAR.json").read_text(encoding="utf-8")
    )
    transitions = 0
    episode_steps = []
    first_post_load_step = False
    critic_loss = actor_loss = None
    with tempfile.TemporaryDirectory(prefix="phase0b2_core_smoke_") as temporary:
        temporary_path = Path(temporary)
        checkpoint_path = temporary_path / "core_smoke_checkpoint.pt"
        for episode in range(2):
            seed_all(seed + episode)
            observations = env.reset(manifest["scenarios"][episode])
            maddpg.prep_rollouts(device="cpu")
            completed = 0
            for _ in range(10):
                action_list = maddpg.step(observations, explore=True)
                actions = torch.cat(action_list, dim=0).clamp(-1.0, 1.0)
                next_obs, rewards, dones = env.step(actions)
                replay.push(
                    observations, actions, rewards, next_obs, dones,
                    [bool(env.mission_complete)] * env.num_agents,
                )
                observations = next_obs
                transitions += 1
                completed += 1
                if all(bool(value) for value in dones):
                    break
            episode_steps.append(completed)
        if len(replay) < 2:
            raise RuntimeError("training smoke did not populate replay")
        sample = replay.sample(min(8, len(replay)), norm_rews=False, device="cpu")
        maddpg.prep_training(device="cpu")
        critic_loss, critic_td = maddpg.update_critic_only(sample, 0)
        _, actor_loss, actor_td = maddpg.update(sample, 1)
        replay.update_priorities(sample[6], torch.maximum(critic_td.abs(), actor_td.abs()), sample[7])
        maddpg.update_all_targets()
        maddpg.save(checkpoint_path, metadata={"purpose": "phase0b2_core_training_smoke", "seed": seed})
        checkpoint_sha256 = sha(checkpoint_path)
        loaded = MADDPG.init_from_save(checkpoint_path, device="cpu")
        loaded.prep_rollouts(device="cpu")
        action_list = loaded.step(observations, explore=False)
        actions = torch.cat(action_list, dim=0).clamp(-1.0, 1.0)
        env.step(actions)
        first_post_load_step = True
        checkpoint_roundtrip = (
            loaded.checkpoint_metadata == {"purpose": "phase0b2_core_training_smoke", "seed": seed}
            and all(torch.isfinite(action).all().item() for action in action_list)
        )
    close = getattr(env, "close", None)
    if callable(close):
        close()
    passed = bool(
        transitions > 0 and critic_loss is not None and actor_loss is not None
        and np.isfinite(critic_loss) and np.isfinite(actor_loss)
        and checkpoint_roundtrip and first_post_load_step
    )
    summary = {
        "schema": "phase0b2.core_training_smoke.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "training closure only; not an algorithm performance result",
        "status": "PASS" if passed else "FAIL",
        "implementation_source": "core",
        "device": "cpu",
        "episodes": 2,
        "max_steps_per_episode": 10,
        "episode_steps": episode_steps,
        "transitions_pushed": transitions,
        "sample_batch_size": min(8, transitions),
        "critic_update_completed": critic_loss is not None,
        "actor_update_completed": actor_loss is not None,
        "critic_loss_finite": bool(np.isfinite(critic_loss)),
        "actor_loss_finite": bool(np.isfinite(actor_loss)),
        "checkpoint_sha256_from_temporary_file": checkpoint_sha256,
        "checkpoint_persisted_in_repository": False,
        "checkpoint_roundtrip_passed": bool(checkpoint_roundtrip),
        "post_load_step_completed": first_post_load_step,
        "external_data_used": False,
        "legacy_used": False,
    }
    (output_dir / "training_smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "test_report.md").write_text(
        "# Core self-contained training smoke\n\n"
        f"Status: **{summary['status']}**\n\n"
        "This bounded CPU smoke executed two episodes of at most ten steps, populated and sampled replay, completed critic and actor updates, saved and safely reloaded a temporary checkpoint, and stepped once after reload. It is not an algorithm-performance result.\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("core training smoke failed")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
