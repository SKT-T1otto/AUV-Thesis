"""Opt-in phase-one contracts, exact behavior identities and addressed noise.

No environment is constructed here. Hashes are audit handles; equality checks
also compare the complete canonical bytes, not a numerical tolerance.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import random
import types

import numpy as np
import torch
from torch import nn

REVISION = "hgr.phase1.fixed_k.v1"
STREAM_REVISION = "hgr.phase1.named_streams.v1"
PREDICTOR_REVISION = "hgr.phase1.zero_ridge.v1"
CHECKPOINT_PHASE1 = "hgr.complete_cycle.phase1.v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def phase1_options(config):
    value = config.get("phase1")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("phase1 must be an explicit object")
    if value.get("enabled") is False:
        return None
    if value.get("enabled") is not True or value.get("revision") != REVISION:
        raise ValueError("unknown phase1 revision/enabled flag")
    allowed = {"enabled", "revision", "zero_update_bypass", "pairing", "predictor_revision",
               "predictor_mode", "ridge_alpha", "scale_kappa", "reference_scale", "lambda"}
    if set(value) != allowed:
        raise ValueError("phase1 configuration fields must be explicit")
    if not isinstance(value["zero_update_bypass"], bool):
        raise ValueError("zero_update_bypass must be boolean")
    if value["pairing"] not in ("policy_crn", "independent"):
        raise ValueError("unknown phase1 pairing")
    if value["predictor_revision"] != PREDICTOR_REVISION or value["predictor_mode"] not in ("zero", "ridge"):
        raise ValueError("unknown phase1 predictor revision/mode")
    for key in ("ridge_alpha", "scale_kappa", "reference_scale"):
        if isinstance(value[key], bool) or not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if isinstance(value["lambda"], bool) or not math.isfinite(value["lambda"]) or not 0 <= value["lambda"] <= 1:
        raise ValueError("lambda must be in [0,1]")
    if (config.get("profile") != "M20_MOVING_UNKNOWN_MULTI" or config.get("max_steps") != 400
            or config["rl"]["gamma"] != .95 or config.get("ablation", "none") != "none"):
        raise ValueError("phase1 freezes profile, horizon, gamma and separate legacy ablations")
    if (config["reward"].get("executor_id") != 3
            or config["reward"].get("searcher_ids") != [0, 1, 2]):
        raise ValueError("phase1 requires searchers 0/1/2 and executor 3")
    return dict(value)


def runtime_contract(config):
    # All resolved configuration except output/sampling/training metadata.
    # Conservative inclusion is intentional; optimizer state is never included.
    omit = {"output_dir", "total_main_trajectories", "max_total_environment_steps",
            "checkpoint_interval", "seed", "main_prefix_batch_size",
            "suffix_training_episodes_per_cycle", "pilot_prefix_episodes_per_cycle",
            "correction_draws_per_cycle", "prefix_lr", "suffix_lr", "predictor", "phase1",
            "rl", "policy"}
    result = {k: v for k, v in config.items() if k not in omit}
    result["gamma"] = config["rl"]["gamma"]
    result["implementation"] = REVISION
    options = phase1_options(config)
    result["phase1_bypass_authorized"] = bool(options and options["zero_update_bypass"])
    result["action_transform"] = "latent_normal_logstd_clamp_-10_2_then_tanh"
    return result


def _plain(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple, torch.Size)):
        return [_plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(v) for v in value)
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("non-string behavior attribute key")
        return {k: _plain(v) for k, v in value.items()}
    raise ValueError(f"unreviewed behavior attribute type: {type(value).__name__}")


@dataclass(frozen=True)
class BehaviorIdentity:
    exact: bytes
    sha256: str
    fields: dict

    def public(self):
        return dict(sha256=self.sha256, fields=self.fields, revision=REVISION)


def behavior_identity(module, contract):
    # Read live module attributes, including router.temperature, LayerNorm.eps,
    # mean_epsilon, activation inplace flags and each module's train/eval mode.
    # Reject dynamic hooks/callables instead of silently missing their behavior.
    internals = {"_parameters", "_buffers", "_modules", "_non_persistent_buffers_set",
                 "_backward_pre_hooks", "_backward_hooks", "_is_full_backward_hook",
                 "_forward_hooks", "_forward_hooks_with_kwargs", "_forward_hooks_always_called",
                 "_forward_pre_hooks", "_forward_pre_hooks_with_kwargs", "_state_dict_hooks",
                 "_state_dict_pre_hooks", "_load_state_dict_pre_hooks", "_load_state_dict_post_hooks"}
    nodes, pieces, tensor_inventory = {}, [], {}
    for name, child in module.named_modules():
        for key, value in vars(child).items():
            if "hook" in key and value:
                raise ValueError("behavior identity does not permit dynamic hooks")
        attrs = {k: _plain(v) for k, v in vars(child).items() if k not in internals}
        for key in ("observation_dim", "action_dim", "PUBLIC_CONTEXT_DIM"):
            if hasattr(child, key):
                attrs[key] = _plain(getattr(child, key))
        nodes[name] = dict(type=type(child).__module__ + "." + type(child).__qualname__, attributes=attrs)
    # Include nonpersistent buffers too: state_dict alone is not sufficient.
    for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        item = tensor.detach().cpu().contiguous()
        if not bool(torch.isfinite(item).all()):
            raise ValueError("nonfinite behavior state")
        raw = item.reshape(-1).view(torch.uint8).numpy().tobytes()
        meta = canonical(dict(name=name, shape=list(item.shape), dtype=str(item.dtype), size=len(raw)))
        pieces.extend((len(meta).to_bytes(8, "big"), meta, len(raw).to_bytes(8, "big"), raw))
        tensor_inventory[name] = dict(shape=list(item.shape), dtype=str(item.dtype),
                                      sha256=hashlib.sha256(raw).hexdigest())
    fields = dict(modules=nodes, runtime=contract, tensors=tensor_inventory)
    header = canonical(fields)
    exact = len(header).to_bytes(8, "big") + header + b"".join(pieces)
    return BehaviorIdentity(exact, hashlib.sha256(exact).hexdigest(), fields)


def identical_behavior(left, right):
    return left.sha256 == right.sha256 and left.exact == right.exact


@dataclass(frozen=True)
class NoUpdateProof:
    old: nn.Module
    new: nn.Module
    identity: BehaviorIdentity
    contract: dict

    @classmethod
    def create(cls, old, new, contract):
        if contract.get("phase1_bypass_authorized") is not True or contract.get("implementation") != REVISION:
            raise ValueError("bypass requires an explicitly enabled phase1 contract")
        left, right = behavior_identity(old, contract), behavior_identity(new, contract)
        if not identical_behavior(left, right):
            raise ValueError("suffix behavior is not identical")
        # Canonical round trip owns the contract; caller mutations cannot alter it.
        return cls(old, new, left, json.loads(canonical(contract)))

    def validate(self, policy, trajectories, predictions, draws, method, ablation):
        if self.contract.get("phase1_bypass_authorized") is not True or self.contract.get("implementation") != REVISION:
            raise ValueError("bypass requires an explicitly enabled phase1 contract")
        if method != "hgr" or ablation != "none" or draws or any(float(x) != 0 for x in predictions):
            raise ValueError("invalid identical-suffix bypass request")
        if self.new is not policy.phi:
            raise ValueError("bypass proof belongs to another policy")
        for block in (self.old, self.new):
            if not identical_behavior(self.identity, behavior_identity(block, self.contract)):
                raise ValueError("bypass behavior identity changed")
        theta = behavior_identity(policy.theta_minus, self.contract).sha256
        for trajectory in trajectories:
            records = trajectory["records"]
            if (trajectory.get("consumed", False) or trajectory.get("theta_behavior_sha256") != theta
                    or trajectory.get("suffix_behavior_sha256") != self.identity.sha256
                    or trajectory.get("trajectory_complete") is not True or not records
                    or len(records) != len(trajectory["rewards"])
                    or [r["t"] for r in records] != list(range(len(records)))
                    or not all(records[-1]["dones"]) or len(records[-1]["dones"]) != 4
                    or any(all(r["dones"]) for r in records[:-1])):
                raise ValueError("bypass requires a complete fresh main trajectory")
            tau = trajectory["tau"]
            if tau is not None and not 0 <= tau < len(records):
                raise ValueError("invalid bypass boundary")
            if any(bool(r["suffix"]) != (tau is not None and j >= tau) for j, r in enumerate(records)):
                raise ValueError("bypass prefix/suffix stopping-time mismatch")


def named_seed(run_seed, cycle, purpose, index=0):
    descriptor = dict(revision=STREAM_REVISION, run_seed=int(run_seed), cycle=int(cycle),
                      purpose=str(purpose), index=int(index))
    raw = canonical(descriptor)
    digest = hashlib.sha256(raw).hexdigest()
    return digest, int.from_bytes(bytes.fromhex(digest)[:8], "big") & ((1 << 63) - 1)


@dataclass(frozen=True)
class PairNoise:
    pair_id: str
    policy_seed: int
    environment_seed: int
    role: str
    mode: str

    @classmethod
    def make(cls, run_seed, cycle, purpose, index, role, mode):
        if role not in ("old", "new") or mode not in ("policy_crn", "independent"):
            raise ValueError("invalid pair role/mode")
        pair_id, _ = named_seed(run_seed, cycle, "pair/" + purpose, index)
        policy_key = "policy/shared" if mode == "policy_crn" else "policy/" + role
        _, policy_seed = named_seed(run_seed, cycle, pair_id + "/" + policy_key, index)
        _, environment_seed = named_seed(run_seed, cycle, pair_id + "/environment/" + role, index)
        return cls(pair_id, policy_seed, environment_seed, role, mode)

    def table(self, tau, horizon):
        if not 0 <= tau < horizon:
            raise ValueError("invalid noise horizon")
        generator = torch.Generator(device="cpu").manual_seed(self.policy_seed)
        return torch.randn((horizon - tau, 4, 3), generator=generator)

    def public(self):
        return dict(pair_id=self.pair_id, trace_id=self.pair_id + "/" + self.role,
                    policy_seed=self.policy_seed, environment_seed=self.environment_seed,
                    pairing=self.mode, random_source_revision=STREAM_REVISION,
                    coupling_scope="policy_only_environment_independent" if self.mode == "policy_crn"
                    else "independent_policy_and_environment")


@contextmanager
def isolated_global_rng():
    saved = random.getstate(), np.random.get_state(), torch.get_rng_state()
    # Do not initialize CUDA for this CPU runtime. Preserve it if the caller did.
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(saved[0])
        np.random.set_state(saved[1])
        torch.set_rng_state(saved[2])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def audit_local_generators(runtime):
    """Fail closed on an unreviewed persistent RNG in the restored object graph."""
    visited, found = set(), []
    allowed = id(runtime.action_rng)
    def visit(value, path):
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, (torch.Generator, np.random.Generator, np.random.RandomState, random.Random)):
            if id(value) != allowed:
                raise ValueError("unreviewed persistent random generator: " + path)
            found.append(path)
            return
        if isinstance(value, (torch.Tensor, np.ndarray, str, bytes, int, float, bool,
                              type(None), type, types.ModuleType, types.FunctionType, types.MethodType)):
            return
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, path + "." + str(key))
        elif isinstance(value, (list, tuple, set)):
            for index, child in enumerate(value):
                visit(child, path + f"[{index}]")
        elif hasattr(value, "__dict__"):
            visit(vars(value), path)
    visit(runtime, "runtime")
    return dict(local_generators=found, global_innovations=["python", "numpy_legacy", "torch_cpu"],
                process_states="preserved_object_graph",
                environment_alignment="branch_global_sequence_only_not_component_addressed")
