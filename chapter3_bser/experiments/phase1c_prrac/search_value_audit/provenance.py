"""Read-only checkpoint, scenario identity, and atomic audit output utilities."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from . import BEHAVIOR_REFERENCE, FEATURE_SCHEMA, HORIZON, MAX_STEPS, PROFILE, SCHEMA, THRESHOLD, TRAINING_REFERENCE

ROOT = Path(__file__).resolve().parents[4]


def json_value(value):
    if torch.is_tensor(value):
        return json_value(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def canonical(value):
    return json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def weights_hash(state):
    values = []
    for key, value in sorted(state.items()):
        array = torch.as_tensor(value).detach().cpu().contiguous().numpy()
        values.append((key, array.dtype.str, array.shape, hashlib.sha256(array.tobytes()).hexdigest()))
    return digest(values)


def actor_hash(snapshot):
    return digest([weights_hash(item) for item in snapshot])


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def atomic_csv(path, rows, *, fields=()):
    path = Path(path)
    columns = list(dict.fromkeys([*fields, *(k for row in rows for k in row)])) or ["status"]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: canonical(v) if isinstance(v, (dict, list, tuple, np.ndarray)) else json_value(v) for k, v in row.items()})
    temporary.replace(path)


def git_identity():
    def git(*args):
        return subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT)
    status = git("status", "--porcelain=v1").decode().splitlines()
    diff = git("diff", "--binary", "HEAD")
    untracked = git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    untracked_hashes = {name: file_hash(ROOT / name) for name in untracked if name and (ROOT/name).is_file()}
    relevant = ["chapter3_bser/models/search_value_head.py", "chapter3_bser/models/prrac/prrac_maddpg.py",
                "chapter3_bser/experiments/phase1c_prrac", "chapter3_bser/online/allocator.py", "core/env/observation_contract.py", "core/env/uav_env.py"]
    comparisons = {}
    for reference in (BEHAVIOR_REFERENCE, TRAINING_REFERENCE):
        try:
            comparisons[reference] = git("diff", "--stat", reference, "HEAD", "--", *relevant).decode()
        except subprocess.CalledProcessError:
            comparisons[reference] = "reference_unavailable"
    return dict(head=git("rev-parse", "HEAD").decode().strip(), branch=git("branch", "--show-current").decode().strip(),
                status=status, worktree_diff_sha256=digest([hashlib.sha256(diff).hexdigest(), untracked_hashes]),
                untracked_file_hashes=untracked_hashes, reference_differences=comparisons)


def training_label_baseline(payload):
    """Inspect stored unique PRE_FOUND transitions, never sample the replay."""
    replay = payload.get("prrac_replay_state", {})
    base = replay.get("base_replay", {})
    result = dict(source="checkpoint.prrac_replay_state", positive_fraction=None,
                  mask="occupied ring slot AND search_value_valid AND stage_before==SEARCH(0)",
                  unit="unique (episode_id, transition step); one shared team label, not three searcher labels",
                  sample_weights_used=False, reason=None)
    required = ("search_value_valid", "stage_before", "future_found")
    if not all(name in replay for name in required) or not all(name in base for name in ("filled_i", "max_steps", "episode_ids", "steps")):
        return dict(result, reason="stored_training_labels_or_transition_identity_unavailable")
    n = int(base["filled_i"])
    if not 0 <= n <= int(base["max_steps"]):
        raise ValueError("invalid stored replay size")
    mask = torch.as_tensor(replay["search_value_valid"])[:n].bool() & (torch.as_tensor(replay["stage_before"])[:n] == 0)
    labels = torch.as_tensor(replay["future_found"])[:n]
    if labels.shape != (n, 3, 1):
        raise ValueError("training future_found must have shape [occupied,3,1]")
    unique = {}
    for index in torch.nonzero(mask).flatten().tolist():
        team = labels[index, :, 0].numpy()
        if not np.isin(team, (0, 1)).all() or not np.all(team == team[0]):
            raise ValueError("training searchers do not share one binary team label")
        key = (int(base["episode_ids"][index]), int(base["steps"][index]))
        value = int(team[0])
        if key in unique and unique[key] != value:
            raise ValueError("inconsistent duplicate training transition labels")
        unique[key] = value
    denominator = len(unique)
    result.update(occupied_slots=n, masked_slots=int(mask.sum()), unique_transitions=denominator,
                  duplicate_transitions=int(mask.sum())-denominator, positives=sum(unique.values()),
                  positive_fraction=sum(unique.values()) / denominator if denominator else None,
                  reason=None if denominator else "no_valid_PRE_FOUND_labels",
                  limitations=["stored ring only, not the prioritized sample distribution",
                               "training tail labels follow the original trainer, not D1 administrative censoring"],
                  ring_overwritten=int(base.get("total_push_count", n)) > n)
    return result


def load_checkpoint(path):
    from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
    from chapter3_bser.experiments.phase1c_prrac.search_value_guidance import SearchValueGuidedCandidateScore
    path = Path(path).resolve(strict=True)
    before = file_hash(path)
    payload = evaluator._validate_checkpoint_payload(torch.load(path, map_location="cpu", weights_only=True))
    metadata, state = payload["metadata"], payload["prrac_training_state"]
    for key, expected in (("execution_runtime_revision", "dynamic_public_intercept_v3_atomic_continuity"),
                          ("execution_variant", "B1_ATOMIC_LAST_VALID"), ("runtime_integration_mode", "native")):
        if metadata.get(key) != expected:
            raise ValueError(f"audit checkpoint {key} must equal {expected}")
    head_config = state.get("search_value", {})
    if not head_config.get("enabled") or int(head_config.get("horizon", -1)) != HORIZON or not state.get("search_value_head"):
        raise ValueError("diagnostics require a trained 34D search_value_head with horizon=50")
    snapshot = tuple({key: value.detach().cpu().numpy().copy() for key, value in agent["actor"].items()} for agent in state["agents"])
    head = {key: value.detach().cpu().numpy().copy() for key, value in state["search_value_head"].items()}
    SearchValueGuidedCandidateScore.from_snapshot({"enabled": True}, snapshot=head, head_config=head_config)
    from chapter3_bser.models.prrac.prrac_maddpg import PRRACMADDPG
    with torch.random.fork_rng(devices=[]):
        actor = PRRACMADDPG(architecture=metadata["architecture"], loss=metadata["loss"])
    actor.load_policy_snapshot(snapshot)  # Exact key/shape validation; no optimizer calls.
    if before != file_hash(path):
        raise RuntimeError("checkpoint changed during read")
    model = dict(architecture=metadata["architecture"], loss=metadata["loss"], gamma=state["gamma"], tau=state["tau"],
                 reward=metadata["reward"], policy_snapshot=snapshot, search_value_snapshot=head, search_value_config=head_config)
    identity = dict(checkpoint_path=str(path), checkpoint_sha256=before, actor_weights_sha256=actor_hash(snapshot),
                    head_weights_sha256=weights_hash(head), checkpoint_metadata=metadata)
    return model, identity, payload


def scenarios_from_manifest(value):
    if "scenarios" in value:
        return list(value["scenarios"])
    if PROFILE in value:
        return scenarios_from_manifest(value[PROFILE])
    raise ValueError("manifest has no M20 scenarios")


def scenario_identity(scenario):
    # Fingerprint the complete control-generating initial scenario, excluding
    # descriptive IDs and sampling-validation summaries. Also retain components.
    mandatory = ("initial_agent_positions", "target_initial_position", "target_initial_velocity", "obstacles",
                 "target_motion_mode")
    physical = {key: scenario.get(key) for key in (*mandatory, "initial_executor_wait_point", "flow_phase_x", "flow_phase_y")}
    missing = [name for name in mandatory if name not in scenario]
    return dict(scenario_id=str(scenario["scenario_id"]), scenario_seed=int(scenario["scenario_seed"]),
                physical_sha256=digest(physical), obstacle_sha256=digest(scenario.get("obstacles")),
                agents_sha256=digest(scenario.get("initial_agent_positions")),
                target_sha256=digest({k: scenario.get(k) for k in ("target_initial_position", "target_initial_velocity", "target_motion_mode")}),
                missing_physical_fields=missing)


def overlap_report(new, reference):
    others = [scenario_identity(s) for s in reference]
    matches = []
    for scenario in new:
        item = scenario_identity(scenario)
        for other in others:
            reasons = []
            if item["scenario_seed"] == other["scenario_seed"]:
                reasons.append("same_scenario_seed")
            if item["physical_sha256"] == other["physical_sha256"]:
                reasons.append("same_physical_fingerprint")
            if all(item[k] == other[k] for k in ("obstacle_sha256", "agents_sha256", "target_sha256")):
                reasons.append("same_layout_agents_target")
            if reasons:
                matches.append(dict(new=item, reference=other, reasons=reasons))
    return dict(overlaps=matches, overlapping=bool(matches),
                physical_fields_complete=all(not scenario_identity(s)["missing_physical_fields"] for s in [*new, *reference]))


def experiment_identity(config, model_identity, manifest, *, workers, mode):
    return dict(schema=SCHEMA, diagnostic_only=True, git=git_identity(), behavior_reference=BEHAVIOR_REFERENCE,
                training_reference=TRAINING_REFERENCE, **model_identity, config_sha256=digest(config),
                manifest_sha256=digest(manifest), feature_schema=FEATURE_SCHEMA,
                horizon=HORIZON, threshold=THRESHOLD, max_steps=MAX_STEPS,
                python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                device="cpu", workers=workers, mode=mode,
                branch_selection="C: best original objective single-searcher substitution excluding A/B geometries; stable key ties",
                subsequent_policy="all OFF after the single root decision")


class Progress:
    def __init__(self, output, total):
        self.output, self.total = Path(output), int(total)
        self.started, self.completed = time.monotonic(), set()
        self.write(stage="initializing", status="running")

    def write(self, *, stage, status="running", unit=None, scenario=None, branch=None, error=None):
        if unit is not None:
            if unit in self.completed:
                raise ValueError(f"duplicate completed work unit: {unit}")
            self.completed.add(unit)
        elapsed = time.monotonic() - self.started
        count = len(self.completed)
        value = dict(stage=stage, status=status, completed=count, total=self.total,
                     percent=100*count/self.total if self.total else 0, elapsed_seconds=elapsed,
                     estimated_remaining_seconds=elapsed/count*(self.total-count) if count else None,
                     current_scenario=scenario, current_branch=branch, error=error)
        atomic_json(self.output / "progress.json", value)
        print(canonical(value), flush=True)


def fresh_output(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty (resume unsupported): {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path
