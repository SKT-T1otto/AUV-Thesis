"""Generate the six bounded BSER-E1 thesis-candidate figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _rows(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save(fig, folder, name):
    fig.tight_layout(); fig.savefig(folder / name, dpi=180); plt.close(fig)


def generate_figures(output_dir):
    output = Path(output_dir); folder = output / "figures" / "generated"; folder.mkdir(parents=True, exist_ok=True)
    rows = _rows(output / "per_instance_results.csv"); scaling = _rows(output / "solver_scalability.csv")
    ratios = np.asarray([float(row["greedy_exact_ratio"]) for row in rows]); fig, ax = plt.subplots(); ax.hist(ratios, bins=20); ax.axvline(0.5, color="red", linestyle="--"); ax.set(xlabel="greedy / exact", ylabel="instances"); _save(fig, folder, "greedy_exact_ratio_distribution.png")
    fig, ax = plt.subplots();
    for solver, field in (("standard", "greedy_runtime_seconds"), ("lazy", "lazy_runtime_seconds")):
        grouped = {}
        for row in scaling: grouped.setdefault(int(row["k_search"]), []).append(float(row[field]))
        ax.plot(sorted(grouped), [np.median(grouped[key]) for key in sorted(grouped)], marker="o", label=solver)
    ax.set(xlabel="search candidates per agent", ylabel="median runtime (s)"); ax.legend(); _save(fig, folder, "candidate_scale_runtime.png")
    fig, ax = plt.subplots(); ax.scatter([float(row["greedy_runtime_seconds"]) for row in scaling], [float(row["lazy_runtime_seconds"]) for row in scaling], s=10); ax.set(xlabel="standard greedy runtime (s)", ylabel="lazy greedy runtime (s)"); _save(fig, folder, "standard_vs_lazy_runtime.png")
    fig, ax = plt.subplots(); left = [float(row["expected_response_time"]) for row in rows]; right = [float(row["search_only_expected_response_time"]) for row in rows]; ax.boxplot([left, right], labels=["BSER", "search-only"]); ax.set(ylabel="conditional expected response time"); _save(fig, folder, "bser_vs_search_only_response.png")
    profiles = sorted({row["profile"] for row in rows}); fig, ax = plt.subplots(); x = np.arange(len(profiles)); exact = [np.mean([float(row["exact_objective"]) for row in rows if row["profile"] == profile]) for profile in profiles]; greedy = [np.mean([float(row["greedy_objective"]) for row in rows if row["profile"] == profile]) for profile in profiles]; ax.bar(x - .18, exact, .36, label="exact"); ax.bar(x + .18, greedy, .36, label="greedy"); ax.set_xticks(x, profiles, rotation=20, ha="right"); ax.set(ylabel="objective"); ax.legend(); _save(fig, folder, "objective_by_profile.png")
    case = json.loads((output / "typical_case.json").read_text(encoding="utf-8")); centers = np.asarray(case["cell_centers"]); belief = np.asarray(case["belief"]); selected = np.asarray(case["selected_waypoints"]); standby = np.asarray(case["standby_waypoint"]); fig = plt.figure(); ax = fig.add_subplot(projection="3d"); cloud = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=belief, cmap="viridis", s=8 + 180 * belief / max(float(np.max(belief)), 1e-12)); fig.colorbar(cloud, ax=ax, shrink=.6, label="belief"); ax.scatter(selected[:, 0], selected[:, 1], selected[:, 2], marker="^", s=70, color="orange", label="search allocation"); ax.scatter([standby[0]], [standby[1]], [standby[2]], marker="s", s=80, color="red", label="executor standby"); ax.legend(); ax.set(xlabel="x", ylabel="y", zlabel="z"); _save(fig, folder, "m20_typical_belief_allocation_3d.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("output_dir", type=Path); generate_figures(parser.parse_args().output_dir)
