"""Complete mission accounting, separate from partial/aborted evaluation work."""
import json
from collections import Counter
from dataclasses import replace
from core.env.task_protocol import STRICT, protocol_identity


def terminal_planning_snapshot(env, previous):
    """Read-only final kinematics; no provider refresh, planning, or guidance."""
    task, agents = env.get_task_state(), env.get_agent_state()
    return replace(previous, step=task.step, target_found=task.target_found,
                   mission_complete=task.mission_complete,
                   agents=tuple(replace(agent, position=agents.positions[agent.agent_id],
                       velocity=agents.velocities[agent.agent_id],
                       current_navigation_target=agents.navigation_targets[agent.agent_id])
                       for agent in previous.agents))


def boolean(value):
    """Parse output/CSV booleans without turning missing values into False."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("", "none"):
            return None
        if value in ("true", "1"):
            return True
        if value in ("false", "0"):
            return False
    elif isinstance(value, (bool, int)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"invalid task boolean: {value!r}")


OUTCOME_FLAGS = ("success", "safe_success", "found", "terminated", "truncated", "collision_episode")
OUTCOME_STAGES = {"success": "SUCCESS", "obstacle_collision": "OBSTACLE_COLLISION",
                  "timeout": "TIMEOUT", "running": "INCOMPLETE"}


def strict_outcome(row):
    """Validate known facts first; missing required facts never certify completion."""
    reason = row.get("termination_reason")
    context = f"scenario={row.get('scenario_id', row.get('episode_id', '<unknown>'))!r}, termination_reason={reason!r}"
    def fail(message):
        raise ValueError(f"invalid strict task outcome ({context}): {message}")
    if reason is not None and (not isinstance(reason, str) or reason.strip()):
        if reason not in OUTCOME_STAGES:
            fail("unknown termination reason")
    else:
        reason = None
    try:
        flags = {key: boolean(row.get(key)) for key in OUTCOME_FLAGS}
    except ValueError as exc:
        fail(str(exc))
    success, safe, found, terminated, truncated, collision = (flags[k] for k in OUTCOME_FLAGS)
    if success is not None and safe is not None and success != safe:
        fail("success and safe_success disagree")
    if (success is True or safe is True) and (found is False or collision is True):
        fail("success requires Found and excludes obstacle collision")
    if (success is True or safe is True or collision is True) and (terminated is False or truncated is True):
        fail("success/collision requires a terminated, non-truncated episode")
    if terminated is True and truncated is True:
        fail("terminated and truncated cannot both be true")
    if reason is not None:
        expected = {"success": reason == "success", "safe_success": reason == "success",
                    "collision_episode": reason == "obstacle_collision",
                    "terminated": reason != "running", "truncated": False}
        if reason == "success":
            expected["found"] = True
        for key, value in expected.items():
            if flags[key] is not None and flags[key] != value:
                fail(f"{key}={flags[key]!r} contradicts {reason}")
    if reason in (None, "running") or any(value is None for value in flags.values()):
        return "INCOMPLETE"
    return OUTCOME_STAGES[reason]


def validated_rows(rows, *, require_complete=False):
    """Shared episode boundary for summaries, CSV recovery, funnels and pairs."""
    rows = list(rows)
    identities = {tuple(protocol_identity(row).values()) for row in rows}
    if len(identities) > 1:
        raise ValueError("episode rows cannot mix task protocols")
    if not rows or protocol_identity(rows[0])["task_protocol"] != STRICT:
        return rows
    normalized = []
    for row in rows:
        stage = strict_outcome(row)
        if require_complete and stage == "INCOMPLETE":
            raise ValueError(f"strict outcomes must be complete: scenario={row.get('scenario_id')!r}")
        value = dict(row)
        for key in (*OUTCOME_FLAGS, "contact_episode", "hold_episode"):
            if key in row:
                value[key] = boolean(row[key])
        value["failure_stage"] = stage
        normalized.append(value)
    return normalized


def aggregate_task_outcomes(rows, expected_episodes=None):
    if any(protocol_identity(row)["task_protocol"] != STRICT for row in rows):
        raise ValueError("strict outcome summary cannot mix task protocols")
    expected = len(rows) if expected_episodes is None else int(expected_episodes)
    rows = validated_rows(rows)
    valid = [row for row in rows if row["failure_stage"] != "INCOMPLETE"]
    counts = Counter(r["termination_reason"] for r in valid)
    found = [r for r in valid if boolean(r["found"]) is True]
    success_found = sum(r["termination_reason"] == "success" for r in found)
    roles, agents, phases = Counter(), Counter(), Counter()
    for row in valid:
        if row["termination_reason"] != "obstacle_collision":
            continue
        ids = row.get("first_collision_agent_ids", [])
        ids = json.loads(ids) if isinstance(ids, str) else ids
        agents.update(str(i) for i in ids)
        roles.update({"Executor" if i == 3 else "Searcher" for i in ids})
        phases.update([row.get("first_collision_phase")])
    count = len(valid)
    complete = count == expected == len(rows) and count > 0
    def rate(n):
        return n / count if count and complete else None
    return {
        "n_valid_episodes": count, "n_expected_episodes": expected,
        "n_missing_or_abnormal_episodes": max(0, expected - count),
        "evaluation_complete": complete,
        "n_success": counts["success"], "n_collision_failure": counts["obstacle_collision"],
        "n_timeout": counts["timeout"],
        "safe_success_rate": rate(counts["success"]), "success_rate": rate(counts["success"]),
        "collision_failure_rate": rate(counts["obstacle_collision"]),
        "timeout_failure_rate": rate(counts["timeout"]), "found_rate": rate(len(found)),
        "success_if_found_numerator": success_found, "success_if_found_denominator": len(found),
        "success_if_found_rate": success_found / len(found) if found and complete else None,
        "first_collision_by_role": dict(roles), "first_collision_by_agent": dict(agents),
        "first_collision_by_phase": dict(phases),
        "actual_environment_steps": sum(int(r.get("episode_length") or 0) for r in rows),
        "actual_training_updates": sum(int(r.get("optimizer_update_count") or 0) for r in rows),
        "collector_wall_seconds": sum(float(r["wall_seconds"]) for r in rows)
            if rows and all(r.get("wall_seconds") is not None for r in rows) else None,
    }
