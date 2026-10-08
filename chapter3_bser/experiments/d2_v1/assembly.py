"""Reuse the sealed D2 algorithm without its zero-action experiment runtime.

No actor, reward, dynamics or execution/handoff contract is replaced here.
Imports are lazy so historical runtimes keep their original construction path.
"""
from collections import Counter

from .contract import enabled


def prepare(owner, config, scenario):
    if enabled(config):
        from chapter3_bser.experiments.safe_search_v1.runtime import SafetyMixin
        SafetyMixin._setup_safe(owner, scenario, "V4")
        from chapter3_bser.experiments.d2_performance.planning import D2PlanningViews
        owner.planning_views = D2PlanningViews(owner, owner.planning_views.clearance)
        owner.effect_arm = "D2"
        owner.effect_counts = Counter()


def build_controller(owner, parent):
    from chapter3_bser.experiments.bser_effect_v1.allocator import EffectAllocator
    from chapter3_bser.experiments.bser_effect_v1.controller import EffectController, EffectPolicy
    from chapter3_bser.online.controller import OnlineBSERController
    from .controller import D2SearchController
    if type(parent) is not OnlineBSERController:
        raise ValueError("D2 requires the reference controller construction contract")
    allocator = EffectAllocator(owner, parent.allocator.config, "D2", owner.env.get_agent_state().positions[3])
    inner = EffectController(parent.config, allocator)
    inner.prrac_runtime_contract = parent.prrac_runtime_contract
    wrapped = D2SearchController(inner, owner)
    inner.policy = EffectPolicy(inner.config)
    return wrapped


def build_bridge(owner):
    from chapter3_bser.experiments.safe_search_v1.safe_guidance import SafetyBridge
    return SafetyBridge(owner)


def diagnostics(owner):
    return dict(planner_protocol="d2_v1", reference_arm="D2", safety_variant="V4",
                safe_counts=dict(owner.safe_counts), effect_counts=dict(owner.effect_counts),
                unresolved_searcher_ids=sorted(owner.safe_pending),
                audit_record_count=len(owner.controller.allocator.audit))
