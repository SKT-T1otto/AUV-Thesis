"""Single construction boundary for PRRAC online controllers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from chapter3_bser.experiments.phase1c_prrac.execution_continuity import (
    CHECKPOINT_RUNTIME_REVISION,
    ExecutionContinuityController,
    ExecutionVariant,
    parse_execution_variant,
)
from chapter3_bser.online.controller import OnlineBSERController


NATIVE_B1_RUNTIME_REVISION = "dynamic_public_intercept_v3_atomic_continuity"
CONTROLLER_FACTORY_VERSION = "prrac.controller_factory.v1"
RUNTIME_INTEGRATION_MODES = ("legacy", "overlay", "native")


@dataclass(frozen=True)
class PRRACRuntimeContract:
    checkpoint_runtime_revision: str
    execution_variant: ExecutionVariant
    runtime_integration_mode: str
    controller_factory_version: str = CONTROLLER_FACTORY_VERSION


def runtime_contract(
    config: Mapping[str, Any],
    *,
    execution_variant: str | ExecutionVariant | None = None,
    runtime_integration_mode: str | None = None,
    checkpoint_runtime_revision: str | None = None,
) -> PRRACRuntimeContract:
    revision = str(
        checkpoint_runtime_revision
        or config.get("checkpoint_runtime_revision")
        or config.get("execution_runtime_revision")
        or CHECKPOINT_RUNTIME_REVISION
    )
    variant = parse_execution_variant(
        execution_variant
        or config.get("execution_variant")
        or ExecutionVariant.B0_LEGACY_V2_1.value
    )
    default_mode = (
        "legacy"
        if variant is ExecutionVariant.B0_LEGACY_V2_1
        else "native"
        if revision == NATIVE_B1_RUNTIME_REVISION
        else "overlay"
    )
    mode = str(runtime_integration_mode or config.get("runtime_integration_mode") or default_mode)
    if mode not in RUNTIME_INTEGRATION_MODES:
        raise ValueError(f"unsupported PRRAC runtime integration mode: {mode!r}")
    factory_version = str(
        config.get("controller_factory_version", CONTROLLER_FACTORY_VERSION)
    )
    if factory_version != CONTROLLER_FACTORY_VERSION:
        raise ValueError(f"unsupported PRRAC controller factory: {factory_version!r}")

    if variant is ExecutionVariant.B0_LEGACY_V2_1:
        if revision != CHECKPOINT_RUNTIME_REVISION or mode != "legacy":
            raise ValueError("B0 requires the legacy v2.1 runtime contract")
    elif revision == CHECKPOINT_RUNTIME_REVISION:
        if mode != "overlay":
            raise ValueError("legacy checkpoints require overlay integration for B1/B2/B3")
    elif revision == NATIVE_B1_RUNTIME_REVISION:
        if variant is not ExecutionVariant.B1_ATOMIC_LAST_VALID or mode != "native":
            raise ValueError("native runtime supports only B1_ATOMIC_LAST_VALID")
    else:
        raise ValueError(f"unsupported checkpoint runtime revision: {revision!r}")
    return PRRACRuntimeContract(revision, variant, mode)


def build_prrac_online_controller(
    phase1b_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
    *,
    execution_variant: str | ExecutionVariant | None = None,
    runtime_integration_mode: str | None = None,
    checkpoint_runtime_revision: str | None = None,
    search_value_scorer=None,
):
    """Build legacy, diagnostic-overlay, or native-B1 from one strict contract."""

    contract = runtime_contract(
        experiment_config,
        execution_variant=execution_variant,
        runtime_integration_mode=runtime_integration_mode,
        checkpoint_runtime_revision=checkpoint_runtime_revision,
    )
    from chapter3_bser.experiments.phase1c_prrac.search_value_guidance import (
        SearchValueGuidedBSERAllocator, resolve_search_value_guidance,
    )
    guidance_config = resolve_search_value_guidance(experiment_config.get("search_value_guidance"))
    allocator = None
    if guidance_config["enabled"] and guidance_config["weight"] > 0.0:
        if search_value_scorer is None or search_value_scorer.config != guidance_config:
            raise ValueError("active candidate ranking requires a matching frozen SearchValue scorer")
        allocator = SearchValueGuidedBSERAllocator(search_value_scorer)
    legacy = (
        OnlineBSERController(dict(phase1b_config))
        if allocator is None
        else OnlineBSERController(dict(phase1b_config), allocator=allocator)
    )
    controller = (
        legacy
        if contract.execution_variant is ExecutionVariant.B0_LEGACY_V2_1
        else ExecutionContinuityController(
            legacy,
            variant=contract.execution_variant,
            config=experiment_config,
        )
    )
    controller.prrac_runtime_contract = contract
    return controller


__all__ = (
    "CONTROLLER_FACTORY_VERSION",
    "NATIVE_B1_RUNTIME_REVISION",
    "PRRACRuntimeContract",
    "RUNTIME_INTEGRATION_MODES",
    "build_prrac_online_controller",
    "runtime_contract",
)
