"""Strict read-only state inventory for deterministic replay, not restoration.

Unknown object types fail instead of using repr, object addresses, or incomplete
pickle snapshots. Privileged environment internals enter hashes only, never the
candidate scoring interface.
"""

from collections import deque
from enum import Enum
import hashlib
import random
from pathlib import Path
import types
import marshal

import numpy as np
import torch

from .provenance import digest

# Only these named references/counters are excluded. The actual excluded paths
# are emitted with the inventory. Core physics, map caches and all RNGs remain.
EXCLUSIONS = {
    "diagnostics": "write-only diagnostic collector; never read by control",
    "_last_update_seconds": "wall-clock timing only",
    "last_planning_seconds": "wall-clock timing only",
    "_started_at": "wall-clock timing only",
}
RECOVERY_LOG_FIELDS = {"_counts", "_agent_counts", "_durations", "_recovery_collision_count",
                       "_recovery_max_collision_streak", "_observed_search_steps", "_tracking_deltas",
                       "_planning_audits", "_activation_steps"}


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                torch_deterministic=torch.are_deterministic_algorithms_enabled(),
                torch_threads=torch.get_num_threads())


def normalized(value, *, path="root", exclusions=None, active=None):
    exclusions = {} if exclusions is None else exclusions
    active = set() if active is None else active
    if isinstance(value, Enum):
        return dict(enum=type(value).__qualname__, value=value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return normalized(value.item(), path=path, exclusions=exclusions, active=active)
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous().numpy()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(f"object array cannot be fingerprinted: {path}")
        return dict(dtype=value.dtype.str, shape=list(value.shape), sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value, (torch.device, torch.dtype, Path)):
        return str(value)
    if isinstance(value, np.random.Generator):
        return normalized(value.bit_generator.state, path=path, exclusions=exclusions, active=active)
    if isinstance(value, np.random.RandomState):
        return normalized(value.get_state(), path=path, exclusions=exclusions, active=active)
    if isinstance(value, (random.Random, torch.Generator)):
        return normalized(value.getstate() if isinstance(value, random.Random) else value.get_state(), path=path, exclusions=exclusions, active=active)
    if isinstance(value, types.MethodType):
        function = value.__func__
        allowed = {"core.env.uav_env", "chapter3_bser.integration.guided_env"}
        if function.__module__ not in allowed:
            raise TypeError(f"uninventoried runtime method at {path}")
        # The only per-instance closure is the existing GuidedEnv navigation hook.
        # Its owner and bound runtime are already inventoried in wrappers/env.
        closures = {}
        for name, cell in zip(function.__code__.co_freevars, function.__closure__ or ()):
            owner = cell.cell_contents
            if name != "owner" or type(owner).__module__ != "chapter3_bser.integration.guided_env":
                raise TypeError(f"uninventoried method closure at {path}.{name}")
            closures[name] = type(owner).__qualname__
        exclusions[path+".__self__/closure.owner"] = "bound runtime and GuidedEnv owner inventoried separately; method bytecode retained"
        return dict(method=function.__module__+"."+function.__qualname__,
                    code_sha256=hashlib.sha256(marshal.dumps(function.__code__)).hexdigest(), closures=closures)
    ident = id(value)
    if ident in active:
        raise TypeError(f"unresolved cyclic runtime reference: {path}")
    active.add(ident)
    try:
        descend = lambda v, p: normalized(v, path=p, exclusions=exclusions, active=active)
        if isinstance(value, dict):
            fields = {}
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0])):
                if k in EXCLUSIONS:
                    exclusions[f"{path}.{k}"] = EXCLUSIONS[k]
                else:
                    fields[str(k)] = descend(v, f"{path}.{k}")
            return fields
        if isinstance(value, (tuple, list, deque)):
            return [descend(v, f"{path}[{i}]") for i, v in enumerate(value)]
        if isinstance(value, (set, frozenset)):
            return sorted((descend(v, path) for v in value), key=str)
        if hasattr(value, "__dict__") and type(value).__module__.startswith(("core.", "chapter3_bser.", "tests.", "test_", "types")):
            fields = {}
            for key, val in sorted(vars(value).items()):
                if key in EXCLUSIONS:
                    exclusions[f"{path}.{key}"] = EXCLUSIONS[key]
                elif type(value).__name__ == "SearchCollisionRecoveryControllerV2" and key in RECOVERY_LOG_FIELDS:
                    exclusions[f"{path}.{key}"] = "C2 reporting-only accumulator/list; detector, failure memory, plans and activation flags retained"
                else:
                    fields[key] = descend(val, f"{path}.{key}")
            return dict(type=f"{type(value).__module__}.{type(value).__qualname__}", fields=fields)
        raise TypeError(f"unsupported runtime state type at {path}: {type(value)}")
    finally:
        active.remove(ident)


def component_fingerprint(components):
    excluded, inventory, hashes = {}, {}, {}
    for name, value in components.items():
        encoded = normalized(value, path=name, exclusions=excluded)
        hashes[name] = digest(encoded)
        inventory[name] = sorted(vars(value)) if hasattr(value, "__dict__") else sorted(value) if isinstance(value, dict) else type(value).__name__
    return dict(sha256=digest(hashes), components=hashes, inventory=inventory, exclusions=excluded)


def runtime_fingerprint(env, provider, controller, bridge, recovery, scorer, *, context, observations, action_adapter=None):
    wrappers, owners, current = [], [], env
    while current is not env.unwrapped:
        attributes = vars(current)
        owners.append(current)
        child_key = next((key for key in ("env", "_env") if key in attributes), None)
        if child_key is None:
            raise TypeError("unknown environment wrapper; cannot establish complete runtime inventory")
        # GuidedEnv also points directly at the already-inventoried runtime.
        wrappers.append({k: v for k, v in attributes.items() if k not in (child_key, "_runtime")})
        current = attributes[child_key]
    hook = vars(env.unwrapped).get("_update_nav_targets")
    if isinstance(hook, types.MethodType):
        if hook.__self__ is not env.unwrapped:
            raise TypeError("navigation hook references an uninventoried runtime")
        for name, cell in zip(hook.__func__.__code__.co_freevars, hook.__func__.__closure__ or ()):
            if name != "owner" or not any(cell.cell_contents is owner for owner in owners):
                raise TypeError("navigation hook references an uninventoried wrapper")
    legacy = getattr(controller, "legacy", controller)
    controller_fields = {k: v for k, v in vars(controller).items() if k not in ("legacy", "allocator")}
    legacy_fields = {k: v for k, v in vars(legacy).items() if k != "allocator"}
    allocator = legacy.allocator
    # The observer wrapper owns no dynamics; fingerprint its original delegate.
    delegate = getattr(allocator, "delegate", allocator)
    allocator_fields = {k: v for k, v in vars(delegate).items() if k != "scorer"}
    scorer_fields = {k: v for k, v in vars(scorer).items() if k in ("active", "config", "extractor", "_features", "_state_step")}
    components = dict(environment=env.unwrapped, wrappers=wrappers,
                      provider={k: v for k, v in vars(provider).items() if k != "env"},
                      controller=controller_fields, legacy_controller=legacy_fields, allocator=allocator_fields,
                      bridge=bridge, recovery=recovery, scorer=scorer_fields, context=context,
                      observations=observations, installed_guidance=env.installed_context,
                      action_adapter=action_adapter, rng=rng_state())
    result = component_fingerprint(components)
    result["exclusions"].update({"wrapper.env/_env/_runtime": "references represented once by wrappers and environment",
                                 "controller.allocator.observer": "audit-only observer; delegate config/execution inventoried",
                                 "scorer.head": "weights frozen and hashed independently",
                                 "scorer.reporting_fields": "candidate/selected/ranking/change counts, sums, and _changed_proposal only update metrics; not read by candidate scoring or acceptance"})
    return result
