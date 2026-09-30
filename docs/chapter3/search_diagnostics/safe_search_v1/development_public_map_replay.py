"""Independent read-only replay for D2 scene 7 / V4; never a D2 arm.

Run from the repository root using ``python -B -m
docs.chapter3.search_diagnostics.safe_search_v1.development_public_map_replay``.
Only the returned capture_runtime dependency is wrapped. Public map arrays are
copied directly, following core/mapping/planning_state.py:extract_planning_state
and core/mapping/path_planner.py:OnlineUnknownMapTaskPlanner._refresh_online_masks.
There is no graph extraction, planner query, map update, or hidden-truth access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[4]
INDEX, VARIANT, BUDGET = 7, "V4", 100


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def segment_touches_box(start, end, lower, upper):
    """Closed slab intersection, including face/edge/corner contact."""
    entry, leave = 0.0, 1.0
    for a, b, low, high in zip(start, end, lower, upper):
        delta = b - a
        if delta == 0.0:
            if a < low or a > high:
                return False
        else:
            near, far = sorted(((low - a) / delta, (high - a) / delta))
            entry, leave = max(entry, near), min(leave, far)
            if entry > leave:
                return False
    return True


def touched_cells(start, end, centers, spacing, clearance):
    half = [0.5 * float(value) + clearance for value in spacing]
    return [index for index, center in enumerate(centers)
            if segment_touches_box(start, end,
                [float(c) - h for c, h in zip(center, half)],
                [float(c) + h for c, h in zip(center, half)])]


def _copied(value, dtype):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, dtype=dtype, copy=True)


def add_public_capture(runtime, row):
    """Append independent diagnostics using existing public arrays only.

    target_state, obstacles, raycast and physical collision geometry are never
    accessed. Geometry works on copied arrays, not a PlanningStateView graph.
    Safety status is explicitly dated to installed guidance; terminal steps can
    retain the previous guidance because no new command is compiled after done.
    """
    import numpy as np
    from chapter3_bser.experiments.safe_search_v1.public_geometry import PublicGeometry

    planner = runtime.env.unwrapped.map_module
    if runtime.env.get_scenario_identity()["obstacle_knowledge_mode"] != "online_unknown":
        raise ValueError("public-map replay only accepts online_unknown scenarios")
    centers = _copied(planner.flat_xyz_centers, np.float64).reshape(-1, 3)
    shape = tuple(int(v) for v in planner.grid_size)
    spacing = (float(planner.cell_dx), float(planner.cell_dy), float(planner.cell_dz))
    origin = tuple(float(centers[0, axis] - 0.5 * spacing[axis]) for axis in range(3))
    probability = _copied(planner.occupancy_probability, np.float64).reshape(-1)
    occupied = _copied(planner.known_occupied_mask, np.bool_).reshape(-1)
    free = _copied(planner.known_free_mask, np.bool_).reshape(-1)
    unknown = _copied(planner.unknown_mask, np.bool_).reshape(-1)
    last = _copied(planner.occupancy_last_observed_step, np.int64).reshape(-1)
    count = _copied(planner.occupancy_observation_count, np.int64).reshape(-1)
    if any(len(v) != len(centers) for v in (probability, occupied, free, unknown, last, count)):
        raise ValueError("public map array dimensions differ")
    if not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
        raise ValueError("public occupancy probability is invalid")
    clearance = float(runtime.planning_views.clearance)
    if clearance != 0.4:
        raise ValueError("this bounded case replay requires the existing 0.4 clearance")
    geometry_input = SimpleNamespace(grid=SimpleNamespace(shape=shape, origin=origin,
        spacing=spacing, cell_centers=centers), occupancy=SimpleNamespace(occupied_mask=occupied))
    geometry = PublicGeometry(geometry_input, clearance)
    occupied_ids = set(np.flatnonzero(occupied).tolist())
    unknown_ids = set(np.flatnonzero(unknown).tolist())

    def segment(start, end):
        cells = touched_cells(start, end, centers, spacing, clearance)
        return dict(start=list(start), end=list(end),
                    free_of_public_occupied=geometry.segment_free(start, end),
                    touched_cell_indices=cells,
                    touched_occupied_cell_indices=sorted(set(cells) & occupied_ids),
                    touched_unknown_cell_indices=sorted(set(cells) & unknown_ids))

    status = getattr(runtime.bridge, "last_safety_status", {})
    agents = []
    for agent in row["agents"]:
        agent_id, position = agent["agent_id"], agent["position"]
        tracking = runtime.bridge.path_tracker.snapshot(agent_id)
        agents.append(dict(agent_id=agent_id,
            safety_status=status.get(agent_id, status.get(str(agent_id))),
            actual_guidance_segment=segment(position, agent["tracking_waypoint"]),
            underlying_tracker_target=tracking.current_target,
            underlying_tracker_segment=(segment(position, tracking.current_target)
                                        if tracking.current_target is not None else None),
            remaining_tracker_points=[list(point) for point in tracking.remaining_path_points],
            remaining_tracker_path_free=geometry.path_free((position, *tracking.remaining_path_points)),
            current_position_free=geometry.point_free(position),
            semantic_waypoint_free=geometry.point_free(agent["semantic_waypoint"])))
    row["live_public_map"] = dict(diagnostic_only=True, step=int(row["step"]),
        map_revision=int(planner.map_revision), grid_shape=list(shape),
        grid_origin=list(origin), grid_spacing=list(spacing),
        occupied_cell_indices=sorted(occupied_ids), unknown_cell_indices=sorted(unknown_ids),
        free_cell_indices=np.flatnonzero(free).tolist(), occupancy_probability=probability.tolist(),
        occupancy_last_observed_step=last.tolist(), occupancy_observation_count=count.tolist())
    row["live_public_safety"] = dict(diagnostic_only=True, clearance=clearance,
        physical_step=int(row["step"]), guidance_step=int(runtime.guidance.step),
        geometry_uses_copied_public_arrays=True, planning_graph_extracted=False,
        privileged_truth_read=False, agents=agents)
    return row


def wrapped_loader(original, extra_capture=add_public_capture):
    def load(*args, **kwargs):
        dependencies = dict(original(*args, **kwargs))
        original_capture = dependencies["capture_runtime"]

        def capture(*capture_args, **capture_kwargs):
            row = original_capture(*capture_args, **capture_kwargs)
            runtime = capture_args[0] if capture_args else capture_kwargs["runtime"]
            return extra_capture(runtime, row)

        dependencies["capture_runtime"] = capture
        return dependencies
    return load


def run_replay(d2_root, manifest, config, output):
    from chapter3_bser.experiments.safe_search_v1 import run_paired
    from chapter3_bser.experiments.safe_search_v1.run_development import validate_child
    from chapter3_bser.experiments.safe_search_v1.provenance import framework_sources, verify_sources

    d2_root, manifest, config, output = (Path(p).resolve() for p in (d2_root, manifest, config, output))
    child = d2_root / f"scene_{INDEX:04d}"
    reference_path = child / f"episode_{INDEX:04d}" / VARIANT / "summary.json"
    parent = run_paired.read_json(d2_root / "identity.json")
    expected = next(item for item in parent["selected"] if item["original_episode_index"] == INDEX)
    records = validate_child(child, expected, parent)
    reference = next(row for row in records if row["variant"] == VARIANT)
    if (reference["physical_steps"] != 84 or reference["terminal"] is not True
            or reference["stop_reason"] != "obstacle_collision"):
        raise ValueError("the expected completed 84-step D2 scene-7/V4 reference is missing")
    child_identity = run_paired.read_json(child / "identity.json")
    input_hashes = {str(p): sha(p) for p in (config, manifest)}
    if input_hashes != child_identity["input_sha256"]:
        raise ValueError("replay must use exactly the same original config and manifest as D2")
    sources = framework_sources()
    if sources != parent["sources_before"] or sources != child_identity["sources_after"]:
        raise ValueError("current sealed sources do not match the completed D2 reference")
    script = Path(__file__).resolve()
    inputs = (manifest, config, reference_path, child / "identity.json", script)
    output = run_paired.output_directory(output, inputs)
    if d2_root == output or d2_root in output.parents:
        raise ValueError("supplemental replay cannot write inside the D2 evaluation")
    stable_hashes = {str(p): sha(p) for p in inputs}
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(schema="ch3.safe_search.public_map_replay_identity.v1",
        independent_supplement=True, included_in_D2_100_arms=False,
        training=False, checkpoint_loaded=False, privileged_truth_read=False,
        original_episode_index=INDEX, variant=VARIANT, baseline="B0_search_prior",
        max_physical_steps=BUDGET, unchanged_task_horizon=400,
        reference_summary=str(reference_path), reference_signature_sha256=reference["signature_sha256"],
        reference_physical_steps=reference["physical_steps"],
        wrapper_path=str(script), wrapper_sha256=sha(script), stable_input_sha256=stable_hashes,
        sources_before=sources,
        source_notes=["Public arrays match extraction in core/mapping/planning_state.py:extract_planning_state.",
            "Masks are maintained by core/mapping/path_planner.py:OnlineUnknownMapTaskPlanner._refresh_online_masks.",
            "Only run_paired.load_dependencies returned capture_runtime is wrapped; original capture runs once.",
            "No graph extraction, planner query, map update, truth obstacle access or target-state access."])
    run_paired.write_json(output / "replay_identity.json", metadata)
    original_loader = run_paired.load_dependencies
    loader = wrapped_loader(original_loader)
    try:
        run_paired.load_dependencies = loader
        run_paired.run_experiment(manifest, output / "run", config_path=config,
            baseline="B0_search_prior", variants=VARIANT, episode_indices=str(INDEX),
            steps=BUDGET, seed=12729, full_episodes=False, verify_v0=False, search_coverage=True)
        replay = run_paired.read_json(output / "run" / f"episode_{INDEX:04d}" / VARIANT / "summary.json")
        trace_path = output / "run" / f"episode_{INDEX:04d}" / VARIANT / "step_trace.jsonl"
        trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        continuity = (len(trace) == 84 and all(row["step_before"] == index and
                      row["step_after"] == index + 1 for index, row in enumerate(trace)))
        match = bool(continuity and replay["physical_steps"] == reference["physical_steps"]
            and replay["terminal"] == reference["terminal"]
            and replay["stop_reason"] == reference["stop_reason"]
            and replay["found_within_budget"] == reference["found_within_budget"]
            and replay["signature_sha256"] == reference["signature_sha256"])
        equivalence = dict(schema="ch3.safe_search.public_map_replay_equivalence.v1",
            reference_signature_sha256=reference["signature_sha256"],
            replay_signature_sha256=replay["signature_sha256"],
            equal_ordered_signature_digest=match, observed_transition_count=len(trace),
            compared_signature_count=85 if match else None,
            includes_initialization=True,
            comparison="SHA256 of the ordered physical-signature list: initialization plus each transition; "
                       "D2 stores the list digest, not individual signature plaintexts",
            signature_scope="agents, task, observations, reward/action record, guidance and Python/NumPy/Torch RNG",
            included_in_D2_100_arms=False)
        run_paired.write_json(output / "equivalence.json", equivalence)
        if not match:
            raise RuntimeError("supplemental observer did not exactly reproduce the D2 physical-signature digest")
        verify_sources(sources)
        if any(sha(path) != expected_hash for path, expected_hash in stable_hashes.items()):
            raise RuntimeError("wrapper/reference/input bytes changed during supplemental replay")
        metadata.update(sources_after=framework_sources(), source_and_input_verification_passed=True,
                        equal_ordered_signature_digest=True, supplemental_replay_complete=True)
        run_paired.write_json(output / "replay_identity.json", metadata)
        print("PUBLIC_MAP_REPLAY_COMPLETE physical_steps=84 signatures=85 equivalent=True", flush=True)
        return equivalence
    except BaseException as exc:
        run_paired.write_json(output / "failure.json", dict(exception_type=type(exc).__name__,
            message=str(exc), supplemental_replay_complete=False, included_in_D2_100_arms=False))
        raise
    finally:
        run_paired.load_dependencies = original_loader


def self_check():
    """Bounded pure-helper checks; no runtime, sealed gate, or simulator import."""
    import random
    lower, upper = [1, 1, 1], [2, 2, 2]
    assert segment_touches_box([0, 0, 0], [1, 1, 1], lower, upper)
    assert not segment_touches_box([0, 0, 0], [0.9, 0.9, 0.9], lower, upper)
    assert segment_touches_box([1, 1, 1], [1, 1, 1], lower, upper)
    assert not segment_touches_box([0, 1, 1], [0, 2, 2], lower, upper)
    assert touched_cells([0, 0, 0], [1, 0, 0], [[1.5, 0, 0], [4, 0, 0]], [1, 1, 1], 0) == [0]
    calls, token, marker = [], object(), object()
    rng = random.getstate()

    def original_capture(runtime, *, option):
        calls.append((runtime, option))
        return {"original": token}

    def original_loader(*args, **kwargs):
        calls.append((args, kwargs))
        return {"capture_runtime": original_capture, "unchanged": marker}

    def extra(runtime, row):
        row["public_extra"] = runtime
        return row

    dependencies = wrapped_loader(original_loader, extra)("config", observed=True)
    row = dependencies["capture_runtime"](token, option=marker)
    assert row["original"] is token and row["public_extra"] is token
    assert dependencies["unchanged"] is marker and len(calls) == 2
    assert calls[1] == (token, marker) and random.getstate() == rng
    print("PUBLIC_MAP_REPLAY_SELF_CHECK_PASS (geometry, original-call preservation, RNG)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d2-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        self_check()
        return
    if any(value is None for value in (args.d2_root, args.manifest, args.config, args.output_dir)):
        parser.error("replay requires --d2-root, --manifest, --config and --output-dir")
    run_replay(args.d2_root, args.manifest, args.config, args.output_dir)


if __name__ == "__main__":
    main()
