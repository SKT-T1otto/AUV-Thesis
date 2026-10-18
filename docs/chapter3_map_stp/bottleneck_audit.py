"""Offline audit of retained CH3 experiments; never creates or steps an environment.

Run from repository root with the existing AUV Python environment:
  python -B -m docs.chapter3_map_stp.bottleneck_audit --output docs/chapter3_map_stp/bottleneck_20261017_v1
Truth is read only by the retrospective evaluator, never by a planner.
All retained inputs are read-only; output must be a new directory.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "D": "runs/bser_effect_v1/development_20260929_v1",
    "F": "runs/bser_final_v1/development_v1",
    "R": "runs/found_search_v2/development_20260929_v1",
}
MANIFEST = "3090结果/collision_terminal/B0_search_prior_eval100_seed12729_v1/evaluation_manifest.json"
SNAPSHOTS = "runs/map_stp_shadow_v1/collection_20261017_v1"
INPUTS = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    INPUTS[path.relative_to(ROOT).as_posix()] = sha(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def trace(tag, row):
    path = ROOT / DATASETS[tag] / row["artifact_dir"] / "step_trace.jsonl"
    actual = sha(path)
    assert actual == row["trace_sha256"], path
    INPUTS[path.relative_to(ROOT).as_posix()] = actual
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(tag, row):
    path = ROOT / DATASETS[tag] / row["artifact_dir"] / "planning_audit.json"
    value = read(path)
    assert INPUTS[path.relative_to(ROOT).as_posix()] == row["planning_audit_sha256"]
    return value


def longest(flags):
    best = current = 0
    for value in flags:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def center(index):
    # Historical manifest pins a 10 x 10 x 8 grid over 20 x 20 x 8.
    return [1 + 2 * (index // 80), 1 + 2 * ((index // 8) % 10), .5 + index % 8]


def summarize(rows):
    return {
        "episodes": len(rows),
        "found": sum(row["found_within_budget"] for row in rows),
        "pre_found_collision": sum(row["pre_found_collision"] for row in rows),
        "not_found_timeout": sum(not row["found_within_budget"] and not row["pre_found_collision"] for row in rows),
        "penalized_found_steps_mean": statistics.mean(row["found_step"] if row["found_within_budget"] else 400 for row in rows),
        "hold_agent_fraction_mean": statistics.mean(row["searcher_hold_fraction"] for row in rows),
        "stall_agent_fraction_mean": statistics.mean(row["searcher_motion_stall_proxy_fraction"] for row in rows),
        "effective_observation_fraction_mean": statistics.mean(row["search_coverage"]["effective_observation_steps"] / row["pre_found_exposure_steps"] for row in rows),
        "actor_forward_calls": sum(row["controller"]["actor_forward_calls"] for row in rows),
        "optimizer_updates": sum(row["controller"]["optimizer_update_count"] for row in rows),
    }


def candidate_support(rows):
    proposals, initial, installed = [], [], []
    for row in rows:
        idx = row["original_episode_index"]
        for item in audit("D", row):
            if item["kind"] != "proposal":
                continue
            pools = defaultdict(list)
            for candidate in item["candidate_ids"]:
                agent, _, cell = candidate.split("_")
                pools[int(agent[1:])].append(int(cell))
            points = [center(i) for pool in pools.values() for i in pool]
            selected = [int(c.rsplit("_", 1)[1]) for c in item["selected_ids"]]
            full = set(pools) == {0, 1, 2}
            record = {
                "scenario_index": idx, "step": item["step"], "full_three_agent_pool": full,
                "cell_indices_by_agent": dict(pools), "selected_cells": selected,
                "same_pool_for_all_three": full and len({tuple(sorted(p)) for p in pools.values()}) == 1,
                "all_agents_have_origin_cell": full and all(0 in p for p in pools.values()),
                "maximum_x": max(p[0] for p in points) if points else None,
                "x_span": max(p[0] for p in points) - min(p[0] for p in points) if points else None,
                "y_span": max(p[1] for p in points) - min(p[1] for p in points) if points else None,
                "selected_distinct_goal_count": len(set(selected)),
            }
            proposals.append(record)
            if item["step"] == 0:
                initial.append(record)
        t0 = trace("D", row)[0]["before"]["agents"][:3]
        goals = [tuple(a["semantic_waypoint"]) for a in t0]
        installed.append({"scenario_index": idx, "initial_goals": goals, "distinct": len(set(goals))})
    full = [p for p in proposals if p["full_three_agent_pool"]]
    later = [p for p in full if p["step"] > 0]
    return {
        "initial": initial, "installed_initial_goals": installed, "proposals": proposals,
        "summary": {
            "initial_scenarios": len(initial),
            "initial_identical_three_agent_pools": sum(p["same_pool_for_all_three"] for p in initial),
            "initial_all_include_origin": sum(p["all_agents_have_origin_cell"] for p in initial),
            "initial_all_goals_x_at_most_5": sum(p["maximum_x"] <= 5 for p in initial),
            "initial_median_x_span": statistics.median(p["x_span"] for p in initial),
            "initial_duplicate_installed_goal_scenarios": sum(p["distinct"] < 3 for p in installed),
            "all_proposals": len(proposals), "full_three_agent_proposals": len(full),
            "later_full_three_agent_proposals": len(later),
            "later_identical_three_agent_pools": sum(p["same_pool_for_all_three"] for p in later),
            "later_all_goals_x_at_most_5": sum(p["maximum_x"] <= 5 for p in later),
        },
        "limitations": ["Proposals are not necessarily installed; step-zero installed goals are verified from traces.",
                        "Shared endpoints do not imply identical approach routes or redundant observations.",
                        "Spatial support bias is established; its effect on Found requires an intervention."],
    }


def trajectory_diagnostics(rows):
    cases, collisions = [], []
    for row in rows:
        steps = [t for t in trace("F", row) if t["search_transition"]]
        assert len(steps) == row["pre_found_exposure_steps"]
        c = row["search_coverage"]
        per_agent = []
        for i in range(3):
            states = [t["before"]["agents"][i] for t in steps]
            per_agent.append({
                "agent_id": i,
                "longest_hold_run": longest(a["hold"] for a in states),
                "longest_motion_stall_run": longest(t["agent_metrics"][i]["motion_stall_proxy"] for t in steps),
                "distance_travelled": sum(t["agent_metrics"][i]["displacement"] for t in steps),
                "distinct_semantic_goals": len({tuple(a["semantic_waypoint"]) for a in states if a["semantic_waypoint"] is not None}),
                "cross_track_error_median": statistics.median(a["active_segment_cross_track_error"] for a in states if a["active_segment_cross_track_error"] is not None),
            })
        cases.append({"scenario_index": row["original_episode_index"], "found_step": row["found_step"],
                      "collision": row["pre_found_collision"], "exposure_steps": len(steps),
                      "hold_agent_fraction": row["searcher_hold_fraction"], "stall_agent_fraction": row["searcher_motion_stall_proxy_fraction"],
                      "unique_observed_cells": c["unique_observed_cells"], "all_grid_cells": c["grid_cell_count"],
                      "effective_observation_steps": c["effective_observation_steps"], "observation_update_steps": c["observation_update_steps"],
                      "overlap_fraction": c["inter_agent_overlap"]["redundant_agent_cell_fraction"], "agents": per_agent})
        for collision in steps[-1]["collision_records"]:
            i = collision["agent_id"]
            states = [t["before"]["agents"][i] for t in steps]
            trailing = 0
            for a in reversed(states):
                if not a["hold"]:
                    break
                trailing += 1
            collisions.append({"scenario_index": row["original_episode_index"], "step": steps[-1]["step_after"],
                               "agent_id": i, "preceding_hold": states[-1]["hold"], "preceding_reachable": states[-1]["reachable"],
                               "consecutive_hold_before_collision": trailing,
                               "speed_on_hold_entry": states[-trailing]["speed"] if trailing else None,
                               "speed_last_frame": states[-1]["speed"], "collision_record": collision})
    safety = Counter()
    for row in rows:
        safety.update(row["controller"]["safe_search"]["counts"])
    return {"cases": cases, "collisions": collisions, "safety_counts": dict(safety)}


def belief_diagnostics(rows, scenarios):
    # Imports load existing numerical functions only: no simulator construction.
    import numpy as np
    import torch
    from core.env.target_motion import TargetState, simulate_target_trajectory, swept_relative_min_distance
    from core.mapping.path_planner import OnlineUnknownMapTaskPlanner
    torch.set_num_threads(1)
    trajectories, validation = {}, []
    for row in rows:
        idx = row["original_episode_index"]
        s = scenarios[idx]
        assert digest(s) == row["scenario_sha256"]
        traj = simulate_target_trajectory(TargetState(s["target_initial_position"], s["target_initial_velocity"], 0, s["target_motion_mode"]),
                                          400, .2, [20, 20, 8], s["obstacles"], clearance=.2, max_reflections=4)
        trajectories[idx] = traj
        found = None
        for t in trace("F", row):
            if not t["search_transition"] or t["collision_records"]:
                # Strict collision returns before target movement/detection.
                continue
            j, k = t["step_before"], t["step_after"]
            if any(swept_relative_min_distance(a["position"], b["position"], traj[j].position, traj[k].position)[0] <= radius + .1
                   for a, b, radius in zip(t["before"]["agents"][:3], t["after"]["agents"][:3], [2., 2.35, 2.75])):
                found = k
                break
        validation.append({"scenario_index": idx, "recorded_found_step": row["found_step"], "reconstructed_found_step": found,
                           "matches": found == row["found_step"]})
    assert all(v["matches"] for v in validation), "Offline truth reconstruction failed; do not label beliefs."
    values = []
    for path in sorted((ROOT / SNAPSHOTS).glob("*.json.gz")):
        INPUTS[path.relative_to(ROOT).as_posix()] = sha(path)
        snap = json.loads(gzip.decompress(path.read_bytes()))
        state = snap["state"]
        idx, start = snap["identity"]["scenario_index"], state["step"]
        shape = tuple(state["grid"]["shape"])
        initial = np.asarray(state["target_belief"]["probabilities"], dtype=np.float32).reshape(shape)
        valid = np.asarray(state["planning_graph"]["valid_mask"], dtype=np.float32).reshape(shape)
        proxy_fields = dict(
            target_belief_transition_mode="occupancy_constrained_diffusion_v1", target_belief_diffusion_rate=.12,
            belief_map=torch.tensor(initial), occupancy_probability=torch.tensor(state["occupancy"]["occupancy_probability"], dtype=torch.float32).reshape(shape),
            valid_mask_float=torch.tensor(valid), dtype=torch.float32, device=torch.device("cpu"), eps=1e-6,
            claim_count=torch.zeros(shape), reservation_decay=.985, _sync_belief_diagnostics=lambda: None)
        proxies = {}
        for method, mode, alpha in [
            ("existing_diffusion_frozen_map", "occupancy_constrained_diffusion_v1", .12),
            ("existing_isotropic_without_occupancy_weight", "isotropic_diffusion_v1", .12),
            ("occupancy_weight_without_diffusion", "occupancy_constrained_diffusion_v1", 0.),
        ]:
            fields = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in proxy_fields.items()}
            fields.update(target_belief_transition_mode=mode, target_belief_diffusion_rate=alpha)
            proxies[method] = SimpleNamespace(**fields)
        centers = np.asarray(state["grid"]["cell_centers"])
        origin, spacing = np.asarray(state["grid"]["origin"]), np.asarray(state["grid"]["spacing"])
        # Uniform reference uses the same public planning valid-cell mask.
        uniform = valid.reshape(-1) / valid.sum()
        for horizon in range(61):
            if horizon:
                for proxy in proxies.values():
                    OnlineUnknownMapTaskPlanner.predict_belief_motion(proxy)
            if horizon not in [0, 10, 30, 60]:
                continue
            truth = trajectories[idx][start + horizon].position
            cell = np.clip(np.floor((truth - origin) / spacing).astype(int), 0, np.asarray(shape)-1)
            flat = int(np.ravel_multi_index(tuple(cell), shape))
            ball = np.linalg.norm(centers - truth, axis=1) <= 2.45
            forecasts = [("frozen_current_belief", initial.reshape(-1))]
            forecasts += [(method, proxy.belief_map.numpy().reshape(-1)) for method, proxy in proxies.items()]
            known_free = np.asarray(state["occupancy"]["free_mask"], dtype=bool)
            unknown = np.asarray(state["occupancy"]["unknown_mask"], dtype=bool)
            for method, mass in forecasts:
                assert np.isclose(float(mass.sum()), 1., atol=1e-4), "belief normalization failed"
                u = float(uniform[flat])
                ball_u = float(uniform[ball].sum())
                values.append({"scenario_index": idx, "snapshot_step": start, "horizon_steps": horizon, "method": method,
                               "truth_cell_public_valid": bool(valid[tuple(cell)]), "truth_cell_probability": float(mass[flat]),
                               "truth_cell_nll": -math.log(max(float(mass[flat]), 1e-12)),
                               "log_density_ratio_to_uniform": math.log(max(float(mass[flat]), 1e-12) / u) if u else None,
                               "truth_ball_radius": 2.45, "truth_ball_mass": float(mass[ball].sum()),
                               "belief_mass_known_free": float(mass[known_free].sum()), "belief_mass_unknown": float(mass[unknown].sum()),
                               "uniform_mass_known_free": float(uniform[known_free].sum()),
                               "truth_ball_mass_ratio_to_uniform": float(mass[ball].sum()) / ball_u if ball_u else None})
    groups = defaultdict(list)
    for v in values:
        groups[(v["method"], v["horizon_steps"])].append(v)
    summary = []
    for (method, horizon), vs in groups.items():
        per_scene = defaultdict(list)
        for v in vs:
            per_scene[v["scenario_index"]].append(v)
        record = {"method": method, "horizon_steps": horizon, "snapshots": len(vs), "scenarios": len(per_scene),
                  "public_invalid_truth_cells": sum(not v["truth_cell_public_valid"] for v in vs)}
        for metric in ["truth_cell_nll", "log_density_ratio_to_uniform", "truth_ball_mass", "truth_ball_mass_ratio_to_uniform", "belief_mass_known_free", "belief_mass_unknown", "uniform_mass_known_free"]:
            scene_means = [statistics.mean(v[metric] for v in group if v[metric] is not None) for group in per_scene.values()
                           if any(v[metric] is not None for v in group)]
            record["scene_equal_mean_" + metric] = statistics.mean(scene_means) if scene_means else None
        summary.append(record)
    return {"reconstruction_validation": validation, "records": values, "summary": summary,
            "limitations": ["17 retained snapshots from six selected scenes, only steps 0/10/30: not a representative calibration study.",
                            "Rounded manifest truth reconstructed by the existing target-motion function; all 20 F0 first-detection labels match.",
                            "Existing prediction function applied to a detached public-only tensor proxy; occupancy frozen and future observations omitted.",
                            "Offline ablations toggle already-supported prediction modes/alpha on detached tensors, never an environment configuration.",
                            "The 2.45-radius truth-centered mass is a location-accuracy diagnostic, not a route discovery probability or calibration guarantee.",
                            "Predictor is called once per physical step, matching the environment; negative observations normally occur every two steps."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    parent = ROOT / "docs/chapter3_map_stp"
    if parent not in output.parents or output.exists():
        raise ValueError("Output must be a new directory below docs/chapter3_map_stp")
    grouped = {}
    scenario_lists = []
    for tag, directory in DATASETS.items():
        rows = read(ROOT / directory / "episodes.json")
        groups = defaultdict(list)
        for row in rows:
            assert row["terminal"] and row["source_and_input_verification_passed"]
            groups[row["baseline"] + "/" + row["arm"]].append(row)
        grouped[tag] = groups
        for group in groups.values():
            assert len(group) == 20
            scenario_lists.append({r["original_episode_index"]: (r["scenario_sha256"], r["scenario_seed"], r["environment_innovation_seed"]) for r in group})
    assert all(s == scenario_lists[0] for s in scenario_lists), "Scenario pairing failed"
    d2 = grouped["D"]["bser_effect_v1/D2"]
    f0 = grouped["F"]["bser_final_v1/F0"]
    d2_by = {r["original_episode_index"]: r for r in d2}
    f0_d2 = all(r["signature_sha256"] == d2_by[r["original_episode_index"]]["signature_sha256"] for r in f0)
    assert f0_d2, "F0/D2 identity mismatch"
    result = {"schema": "ch3.map_bottleneck.offline_audit.v1", "data_grain": "20 reused development scenes, paired arms; 420 rows are not 420 independent scenes",
              "paired_scenario_validation": True, "F0_D2_all_20_signature_match": f0_d2,
              "group_summary": {tag: {name: summarize(rows) for name, rows in groups.items()} for tag, groups in grouped.items()},
              "candidate_support": candidate_support(d2), "F0_execution": trajectory_diagnostics(f0)}
    collision_checks = []
    for name, rows in grouped["D"].items():
        for row in rows:
            if not row["pre_found_collision"]:
                continue
            final = trace("D", row)[-1]
            for collision in final["collision_records"]:
                a = final["before"]["agents"][collision["agent_id"]]
                collision_checks.append({"arm": name, "scenario_index": row["original_episode_index"], "agent_id": a["agent_id"],
                                         "preceding_hold_and_unreachable": a["hold"] and not a["reachable"]})
    result["D_collision_preceding_state"] = collision_checks
    result["belief_diagnostics"] = belief_diagnostics(f0, read(ROOT / MANIFEST)["scenarios"])
    sources = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
               for directory, pattern in [("core", "*.py"), ("chapter3_bser", "*.py"), ("tools/ch3_baselines", "*.py"), ("configs", "*.json"), ("scripts", "*.py"), ("scripts", "*.sh")]
               for p in (ROOT / directory).rglob(pattern)}
    git = lambda *a: subprocess.check_output(["git", "-c", "core.fsmonitor=false", *a], cwd=ROOT, text=True, encoding="utf-8").strip()
    identity = {"head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"), "working_tree": git("status", "--short"),
                "current_sources": sources, "audit_script_sha256": sha(Path(__file__)), "inputs": INPUTS}
    output.mkdir(parents=True)
    write(output / "evidence.json", result)
    write(output / "identity.json", identity)
    print(json.dumps({"output": str(output), "group_summary": result["group_summary"], "candidate_support": result["candidate_support"]["summary"],
                      "collision_checks": len(collision_checks), "hold_unreachable_before_collision": sum(c["preceding_hold_and_unreachable"] for c in collision_checks),
                      "belief_summary": result["belief_diagnostics"]["summary"], "hashed_inputs": len(INPUTS)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
