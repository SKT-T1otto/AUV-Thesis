"""Keep parent rewards, gamma and network dimensions; version repairs separately."""
import copy

from chapter3_bser.experiments.baselines.common.checkpoint import digest, validate_config

VERSION = "ch3.d2.implementation_repair.v2"
PARENTS = {
    "D2_B2_repair_v2": ("B2_direct_mc", "maddpg", "ch3_baseline_direct_mc", "ch3.baseline.maddpg.v1"),
    "D2_B3_repair_v2": ("B3_direct_boundary", "direct_boundary_maddpg", "ch3_baseline_direct_boundary", "ch3.baseline.boundary_maddpg.v1"),
}
FLAGS = ("waypoint_arrival", "inactive_actor", "hold_residual")


def scenario_fingerprint(scene):
    """Detect exact task copies even when identity/role fields are renamed."""
    metadata = {"scenario_id", "scenario_seed", "scenario_split", "scenario_role"}
    return digest({k:v for k,v in scene.items() if k not in metadata})


def validate(config):
    c = copy.deepcopy(config)
    if c.get("schema") != VERSION or c.get("baseline") not in PARENTS:
        raise ValueError("requires an independent D2 repair configuration")
    baseline, algorithm, method, architecture = PARENTS[c["baseline"]]
    if (c.get("method") != c["baseline"] or c.get("algorithm") != algorithm
            or c.get("architecture_version") != architecture or c.get("checkpoint_schema") != VERSION):
        raise ValueError("repair method/architecture identity mismatch")
    flags = c.get("repairs", {})
    if set(flags) != set(FLAGS) or any(type(flags[k]) is not bool for k in FLAGS):
        raise ValueError("three explicit boolean repair switches are required")
    parent = copy.deepcopy(c)
    parent.update(schema="ch3.baseline.training.v1", baseline=baseline, method=method,
                  checkpoint_schema="ch3.baseline.maddpg.v1")
    validate_config(parent, baseline)
    if c["rl"]["gamma"] != .95 or c.get("planner_protocol") != "d2_v1":
        raise ValueError("repair v2 retains gamma=.95 and the D2 planner")
    if c.get("performance", {}).get("learner_device", "cpu") != "cpu" or c.get("performance", {}).get("cpu_threads", 1) != 1:
        raise ValueError("repair v2 currently uses the reference CPU learner with one thread")
    expected = {"schema": "bser.phase1c.execution_reward.v2", "enabled": True,
                "freeze_searchers_after_found": True, "preserve_discovery_reward": True,
                "contact_entry_bonus": .25, "hold_increment_bonus": .2,
                "terminal_success_bonus": 2., "reward_clip": 3., "executor_id": 3,
                "searcher_ids": [0, 1, 2]}
    if c["reward"] != expected:
        raise ValueError("repair v2 preserves the original composite reward settings")
    if c.get("collision_terminal_reward") != -2.:
        raise ValueError("repair v2 preserves the original collision reward -2")
    timing = "arrival_before_reward_v1" if flags["waypoint_arrival"] else "legacy_v1"
    if c.get("waypoint_reward_timing", timing) != timing:
        raise ValueError("conflicting waypoint repair revision")
    c["waypoint_reward_timing"] = timing
    return c
