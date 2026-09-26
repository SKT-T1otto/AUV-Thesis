"""Read-only numerical audit of saved Phase2 JSON/NPZ; no production imports.

Run with NumPy from the repository root. Writes a NEW compact evidence JSON
beside this script; never edits experiment outputs or invokes any rollout.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
RUN = ROOT / "outputs/chapter3/hgr_phase2/collision_terminal/debug_gradient_efficiency_10scene"
RESULT = RUN / "phase2_results.json"
OUTPUT = Path(__file__).with_name("evidence.json")
METHODS = ("direct_new_mc", "old_mc", "hgr")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def summary(values):
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), min=float(a.min()), median=float(np.median(a)),
                mean=float(a.mean()), max=float(a.max())) if len(a) else dict(count=0)


def main(output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Retained analysis will not be overwritten; choose --output: {output}")
    original_sha = sha(RESULT)
    d = json.loads(RESULT.read_text(encoding="utf-8"))
    cfg = d["config"]
    assert d["status"] == d["frozen_policy_check"] == "PASS"
    assert d["policy_updates"] == 0 and cfg["scope"] == "fresh_main"
    n, repeats = cfg["snapshot_count"], cfg["repeat_count"]
    assert cfg["runtime_config"]["rl"]["gamma"] == .95 and cfg["runtime_config"]["max_steps"] == 400
    scene_ids = [x["snapshot_id"] for x in d["snapshot_manifest"]]
    assert len(set(scene_ids)) == n == 10 and repeats == 2
    assert len(d["samples"]) == 3 * n * repeats
    assert d["gradient_dimension"] == sum(x["size"] for x in d["gradient_parameter_layout"])

    pointers = collections.defaultdict(list)

    def collect_pointers(value):
        if isinstance(value, dict):
            if {"path", "key", "sha256", "shape", "dtype"} <= value.keys():
                pointers[value["path"]].append(value)
            else:
                for item in value.values():
                    collect_pointers(item)
        elif isinstance(value, list):
            for item in value:
                collect_pointers(item)

    collect_pointers(d)
    file_hashes = {}
    for relative, refs in pointers.items():
        path = (RUN / relative).resolve()
        assert path.is_relative_to(RUN.resolve())
        checksum = sha(path)
        assert all(ref["sha256"] == checksum for ref in refs), relative
        with np.load(path, allow_pickle=False) as arrays:
            for ref in refs:
                value = arrays[ref["key"]]
                assert list(value.shape) == ref["shape"] and str(value.dtype) == ref["dtype"]
                assert np.isfinite(value).all()
        file_hashes[relative] = checksum

    with np.load(RUN / d["reference_gradient"]["path"], allow_pickle=False) as arrays:
        gref = arrays["g_ref"]
        refmeans = arrays["case_reference_means"]
        refvar = arrays["reference_mean_variance"]
    close(refmeans.mean(axis=0), gref)
    close(np.linalg.norm(gref), d["reference_gradient_norm"])
    close(refvar.sum(), d["reference_mean_variance_trace"])
    references = [r for case in d["reference"] for r in case["samples"]]
    assert len(references) == cfg["reference_rollouts"] == 20
    assert all(case["rollouts"] == len(case["samples"]) == 2 for case in d["reference"])
    for index, case in enumerate(d["reference"]):
        assert case["snapshot_id"] == scene_ids[index]
        close(np.linalg.norm(refmeans[index]), case["gradient_norm"])
    # Recompute the reference variance TRACE independently from individual
    # gradient norms and case-mean norms (raw reference vectors were not saved).
    reftrace = sum((sum(r["gradient_norm"] ** 2 for r in case["samples"])
                    - 2 * case["gradient_norm"] ** 2) / 2 for case in d["reference"]) / n**2
    close(reftrace, refvar.sum())

    reports, per_scene, compact_rows, corrections, paired_branches = {}, [], [], [], []
    comparison_keys = [(r["snapshot_id"], r["method"], r["repeat_id"]) for r in d["samples"]]
    assert len(set(comparison_keys)) == len(comparison_keys)
    streams = [r["random_stream_id"] for r in d["samples"] + references]
    assert len(set(streams)) == len(streams)
    for scene in scene_ids:
        for repeat in range(repeats):
            ids = {r["pair_id"] for r in d["samples"]
                   if r["snapshot_id"] == scene and r["repeat_id"] == repeat}
            assert len(ids) == 1
    assert not {r["pair_id"] for r in references} & {r["pair_id"] for r in d["samples"]}

    for method in METHODS:
        rows = [r for r in d["samples"] if r["method"] == method]
        assert len(rows) == n * repeats
        vectors = []
        for row in rows:
            index = scene_ids.index(row["snapshot_id"])
            with np.load(RUN / row["gradient_vector"]["path"], allow_pickle=False) as arrays:
                g = arrays[row["gradient_vector"]["key"]]
                vectors.append(g.copy())
                close(np.linalg.norm(g), row["gradient_norm"])
                close(np.dot(g - gref, g - gref), row["gradient_mse"])
                close(np.dot(g - refmeans[index], g - refmeans[index]), row["case_reference_mse"])
                close(row["gradient_mse"] * row["environment_steps"], row["cost_normalized_error"])
                compact_rows.append({k: row[k] for k in (
                    "snapshot_id", "method", "repeat_id", "tau", "environment_steps",
                    "actual_trajectory_length", "gradient_norm", "gradient_mse", "case_reference_mse",
                    "cost_normalized_error", "K_actual", "K_requested")})
                if method == "hgr":
                    g0, delta, full = arrays["g0"], arrays["g_delta"], arrays["g_full"]
                    np.testing.assert_array_equal(g0 + delta, full)
                    qrows = row["correction_queries"]
                    assert len(qrows) == row["K_actual"] == 8
                    branch_cost = 0
                    for q in qrows:
                        if row["tau"] is None:
                            assert q["no_handoff"] and q["label"] == 0 and "old" not in q and "new" not in q
                            continue
                        old, new = q["old"], q["new"]
                        close(q["label"], new["G_plus"] - old["G_plus"])
                        assert old["policy_seed"] == new["policy_seed"]
                        assert old["environment_seed"] != new["environment_seed"]
                        assert old["pair_id"] == new["pair_id"] == q["pair_id"]
                        assert old["snapshot_hash"] == new["snapshot_hash"] == row["snapshot_hash"]
                        for branch in (old, new):
                            assert branch["terminal_step"] == row["tau"] + branch["steps"] <= 400
                            assert branch["original_deadline"] == 400
                            assert branch["pairing"] == "policy_crn"
                            branch_cost += branch["steps"]
                        paired_branches.append(dict(snapshot_id=row["snapshot_id"], repeat_id=row["repeat_id"],
                            pair_id=q["pair_id"], draw=q["draw"], label=q["label"],
                            old_return=old["G_plus"], new_return=new["G_plus"],
                            old_steps=old["steps"], new_steps=new["steps"],
                            old_reason=old["termination_reason"], new_reason=new["termination_reason"],
                            trajectory_hash_equal=old["trajectory_trace_sha256"] == new["trajectory_trace_sha256"]))
                    assert row["environment_steps"] == row["actual_trajectory_length"] + branch_cost
                    norm0, normdelta = float(np.linalg.norm(g0)), float(np.linalg.norm(delta))
                    mse0 = float(np.dot(g0 - gref, g0 - gref))
                    # Stable matched-gradient MSE change, avoiding cancellation.
                    mse_change = float(2 * np.dot(g0 - gref, delta) + np.dot(delta, delta))
                    corrections.append(dict(snapshot_id=row["snapshot_id"], repeat_id=row["repeat_id"],
                        tau=row["tau"], gamma_to_tau=None if row["tau"] is None else .95 ** row["tau"],
                        g0_norm=norm0, g_delta_norm=normdelta,
                        delta_to_g0_norm=normdelta / norm0 if norm0 else None,
                        g0_mse=mse0, full_mse=row["gradient_mse"], matched_mse_change=mse_change,
                        main_steps=row["actual_trajectory_length"], branch_steps=branch_cost,
                        g0_cost_error=mse0 * row["actual_trajectory_length"],
                        label_mean=float(np.mean([q["label"] for q in qrows]))))
        x = np.stack(vectors)
        mean, var = x.mean(axis=0), x.var(axis=0, ddof=1)
        observed = d["statistics"][method]
        close(np.linalg.norm(mean), observed["mean_gradient_norm"])
        close(var.sum(), observed["variance_trace"])
        close(np.mean([r["gradient_mse"] for r in rows]), observed["mean_gradient_mse"])
        close(np.mean([r["cost_normalized_error"] for r in rows]), observed["mean_cost_normalized_error"])
        assert sum(r["environment_steps"] for r in rows) == observed["total_environment_steps"]
        within = []
        with np.load(RUN / observed["mean"]["path"], allow_pickle=False) as stored:
            close(mean, stored["mean"])
            close(var, stored["per_dimension_variance"])
            for index, scene in enumerate(scene_ids):
                mask = [i for i, r in enumerate(rows) if r["snapshot_id"] == scene]
                assert len(mask) == repeats
                case_x = x[mask]
                within.append(case_x.var(axis=0, ddof=1))
                close(case_x.mean(axis=0), stored["case_means"][index])
                close(within[-1], stored["within_case_variance"][index])
                selected = [rows[i] for i in mask]
                per_scene.append(dict(snapshot_id=scene, method=method,
                    handoffs=sum(r["tau"] is not None for r in selected),
                    mean_mse=float(np.mean([r["gradient_mse"] for r in selected])),
                    mean_case_reference_mse=float(np.mean([r["case_reference_mse"] for r in selected])),
                    within_variance_trace=float(within[-1].sum()),
                    environment_steps=sum(r["environment_steps"] for r in selected),
                    mean_cost_error=float(np.mean([r["cost_normalized_error"] for r in selected]))))
        stratified_mean_var = float(np.sum(within) / repeats / n**2)
        mse_sum = sum(r["gradient_mse"] for r in rows)
        reports[method] = {k: v for k, v in observed.items() if not isinstance(v, dict)}
        reports[method].update(
            mean_estimator_mse=float(np.dot(mean - gref, mean - gref)),
            mean_case_reference_mse=float(np.mean([r["case_reference_mse"] for r in rows])),
            stratified_mean_variance_trace=stratified_mean_var,
            cosine_mean_to_reference=float(np.dot(mean, gref) / (np.linalg.norm(mean) * np.linalg.norm(gref))),
            handoff_count=sum(r["tau"] is not None for r in rows),
            environment_steps_distribution=summary([r["environment_steps"] for r in rows]),
            mse_distribution=summary([r["gradient_mse"] for r in rows]),
            largest_three_mse_share=sum(sorted([r["gradient_mse"] for r in rows], reverse=True)[:3]) / mse_sum)

    costs = d["costs"]
    assert sum(costs.values()) == d["total_environment_steps"]
    assert sum(r["environment_steps"] for r in references) == costs["reference_steps"]
    assert sum(r["main_steps"] for r in corrections) == costs["hgr_main_steps"]
    assert sum(q["old_steps"] for q in paired_branches) == costs["hgr_correction_old_steps"]
    assert sum(q["new_steps"] for q in paired_branches) == costs["hgr_correction_new_steps"]
    assert len({q["pair_id"] for q in paired_branches}) == len(paired_branches)
    changed_sources = [p for p, h in d["source_identity"]["files"].items() if sha(ROOT / p) != h]
    assert not changed_sources
    hgr, direct = reports["hgr"], reports["direct_new_mc"]
    actual_corrections = [r for r in corrections if r["tau"] is not None]
    audit_path = ROOT / "outputs/chapter3/hgr_phase2/policy_pair_source.audit.json"
    policy_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    packet = dict(
        schema="hgr.phase2.analysis_evidence.v1", result_path=str(RESULT), result_sha256=original_sha,
        result_mtime_local=__import__("datetime").datetime.fromtimestamp(RESULT.stat().st_mtime).isoformat(),
        artifact_hashes={name: sha(ROOT / "outputs/chapter3/hgr_phase2" / name) for name in
                         ("policy_pair_source.pt", "frozen_phase2_source.pt", "policy_pair_source.audit.json")},
        execution={k: d[k] for k in ("status", "frozen_policy_check", "handoff_correction_status",
                    "policy_updates", "performance_claims_supported", "formal_experiment", "scope",
                    "gradient_dimension", "gradient_samples_per_method", "total_comparison_gradient_samples",
                    "reference_rollouts", "reference_rollouts_per_case", "no_handoff_hgr_samples")},
        config=dict(snapshot_count=n, repeat_count=repeats, reference_rollouts=cfg["reference_rollouts"],
                    K=cfg["runtime_config"]["correction_draws_per_cycle"], gamma=cfg["runtime_config"]["rl"]["gamma"],
                    horizon=400, predictor="zero", pairing="policy_crn", profile=cfg["runtime_config"]["profile"]),
        validation=dict(vector_files_checked=len(file_hashes), finite_shapes_dtypes_hashes="PASS",
                        sample_mse_variance_costs_recomputed="PASS", gradient_addition="PASS",
                        stream_pairing_and_clocks="PASS", source_files_unchanged=True,
                        comparison_rows=len(d["samples"]), reference_rows=len(references),
                        reference_limit="Individual reference vectors not persisted; means verified from case arrays and variance TRACE from saved sample norms. Full per-dimension reference variance cannot be independently rebuilt."),
        costs=dict(costs, total_with_historical_preparation=d["total_environment_steps"],
                   new_environment_steps=d["total_environment_steps"] - costs["source_preparation_steps"]),
        methods=reports, per_scene=per_scene, comparison_samples=compact_rows,
        reference=dict(norm=float(np.linalg.norm(gref)), mean_variance_trace=float(refvar.sum()),
                       rms_standard_error=float(np.sqrt(refvar.sum())),
                       rms_standard_error_to_norm=float(np.sqrt(refvar.sum()) / np.linalg.norm(gref))),
        hgr_vs_direct_ratios={key: hgr[key] / direct[key] for key in
                              ("mean_gradient_mse", "variance_trace", "mean_cost_normalized_error", "total_environment_steps")},
        hgr_decomposition=dict(
            continuation_pair_count=len(paired_branches), continuation_rollout_count=2 * len(paired_branches),
            branch_cost_share=(costs["hgr_correction_old_steps"] + costs["hgr_correction_new_steps"]) / hgr["total_environment_steps"],
            paired_returns_delta=summary([q["label"] for q in paired_branches]),
            absolute_paired_returns_delta=summary([abs(q["label"]) for q in paired_branches]),
            delta_norm_all=summary([r["g_delta_norm"] for r in corrections]),
            delta_norm_handoff=summary([r["g_delta_norm"] for r in actual_corrections]),
            delta_to_g0_handoff=summary([r["delta_to_g0_norm"] for r in actual_corrections]),
            exactly_zero_delta_samples=sum(r["g_delta_norm"] == 0 for r in corrections),
            zero_label_pairs=sum(q["label"] == 0 for q in paired_branches),
            g0_mean_mse=float(np.mean([r["g0_mse"] for r in corrections])),
            matched_mean_mse_change=float(np.mean([r["matched_mse_change"] for r in corrections])),
            g0_mean_cost_error=float(np.mean([r["g0_cost_error"] for r in corrections])),
            branch_termination_counts=dict(collections.Counter(q[key] for q in paired_branches for key in ("old_reason", "new_reason"))),
            same_steps_pairs=sum(q["old_steps"] == q["new_steps"] for q in paired_branches),
            same_trace_pairs=sum(q["trajectory_hash_equal"] for q in paired_branches),
            rows=corrections, branches=paired_branches),
        policy_pair_diagnostic=policy_audit["diagnostic"],
        source_identity_sha256=d["source_identity"]["sha256"], git_head=d["checkout"]["head"],
        file_hashes=file_hashes,
    )
    assert sha(RESULT) == original_sha
    for relative, checksum in file_hashes.items():
        assert sha(RUN / relative) == checksum
    with output.open("x", encoding="utf-8") as handle:
        json.dump(packet, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: packet[k] for k in ("validation", "costs", "methods", "reference", "hgr_vs_direct_ratios")}, ensure_ascii=False))
    print(json.dumps({k: v for k, v in packet["hgr_decomposition"].items() if k not in ("rows", "branches")}, ensure_ascii=False))
    print(json.dumps(actual_corrections, ensure_ascii=False))
    print("EVIDENCE=" + str(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    main(parser.parse_args().output)
