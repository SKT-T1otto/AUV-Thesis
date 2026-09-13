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
    if value is True or value in ("True", "true", "1"):
        return True
    if value is False or value in ("False", "false", "0"):
        return False
    return None


def aggregate_task_outcomes(rows, expected_episodes=None):
    if any(protocol_identity(row)["task_protocol"] != STRICT for row in rows):
        raise ValueError("strict outcome summary cannot mix task protocols")
    expected = len(rows) if expected_episodes is None else int(expected_episodes)
    valid = []
    for row in rows:
        reason = row.get("termination_reason")
        if (boolean(row.get("terminated")) is True and boolean(row.get("truncated")) is False
                and reason in ("success", "obstacle_collision", "timeout")
                and boolean(row.get("success")) is not None and boolean(row.get("found")) is not None):
            success = reason == "success"
            if boolean(row["success"]) != success or boolean(row.get("safe_success")) != success:
                raise ValueError("authoritative task outcome/success disagreement")
            if success and boolean(row["found"]) is not True:
                raise ValueError("success requires Found")
            if boolean(row.get("collision_episode")) != (reason == "obstacle_collision"):
                raise ValueError("collision outcome disagreement")
            valid.append(row)
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
    complete = count == expected == len(rows)
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
