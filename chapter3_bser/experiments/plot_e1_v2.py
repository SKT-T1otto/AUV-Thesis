"""Generate the ten fixed E1-v2 descriptive figures (no cross-scale chart)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _rows(path):
    with Path(path).open(newline="",encoding="utf-8") as handle: return list(csv.DictReader(handle))


def _save(output,name,title,x,y,xlabel="",ylabel=""):
    fig,axis=plt.subplots(figsize=(6.4,4.2)); axis.plot(x,y,marker="o",linewidth=1.4); axis.set(title=title,xlabel=xlabel,ylabel=ylabel); axis.grid(alpha=.25); fig.tight_layout(); fig.savefig(Path(output)/name,dpi=180); plt.close(fig)


def generate_figures(output_dir):
    output=Path(output_dir); methods=_rows(output/"response_diagnostics_comparison.csv"); unique=_rows(output/"per_unique_state_results.csv"); profile=_rows(output/"aggregate_by_profile.csv"); step=_rows(output/"aggregate_by_snapshot_step.csv"); candidates=_rows(output/"candidate_statistics.csv"); scalability=_rows(output/"solver_scalability.csv")
    method_names=sorted({row["method"] for row in methods})
    def means(field):
        return [np.mean([float(row[field]) for row in methods if row["method"]==method and row.get(field,"")!=""]) for method in method_names]
    def bar(name,title,values,ylabel):
        fig,axis=plt.subplots(figsize=(8,4.5)); axis.bar(range(len(values)),values); axis.set_xticks(range(len(values)),method_names,rotation=25,ha="right"); axis.set(title=title,ylabel=ylabel); axis.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(output/name,dpi=180); plt.close(fig)
    bar("figure_01_joint_objective.png","Joint objective by method",means("joint_objective"),"joint objective")
    bar("figure_02_detection_probability.png","Detection probability by method",means("detection_probability"),"probability")
    bar("figure_03_conditional_response.png","Conditional reachable response time",means("conditional_reachable_response_time"),"physical time")
    bar("figure_04_unreachable_mass.png","Unreachable detected mass",means("unreachable_detected_mass"),"detected mass")
    ratios=sorted(float(row["greedy_exact_ratio"]) for row in unique); _save(output,"figure_05_greedy_exact_ratio.png","Greedy/exact ratio",range(len(ratios)),ratios,"unique state rank","ratio")
    runtimes=sorted(float(row["greedy_runtime_seconds"]) for row in unique); _save(output,"figure_06_greedy_runtime.png","Greedy runtime",range(len(runtimes)),runtimes,"unique state rank","seconds")
    counts=sorted(int(row["search_candidate_count"]) for row in candidates); _save(output,"figure_07_candidate_count.png","Search candidate count",range(len(counts)),counts,"unique state rank","candidates")
    _save(output,"figure_08_profile_objective.png","Scenario-weighted objective by profile",[row["profile"] for row in profile],[float(row["mean_joint_objective"]) for row in profile],"profile","objective")
    _save(output,"figure_09_snapshot_objective.png","Objective by requested snapshot step",[int(row["requested_snapshot_step"]) for row in step],[float(row["mean_joint_objective"]) for row in step],"actions executed","objective")
    typical=json.loads((output/"typical_case.json").read_text()); centers=np.asarray(typical["cell_centers"]); belief=np.asarray(typical["belief"]); fig,axis=plt.subplots(figsize=(6,5)); scatter=axis.scatter(centers[:,0],centers[:,1],c=belief,s=12,cmap="viridis"); selected=np.asarray(typical["selected_waypoints"]); axis.scatter(selected[:,0],selected[:,1],marker="x",s=80,c="red",label="search"); standby=np.asarray(typical["standby_waypoint"]); axis.scatter([standby[0]],[standby[1]],marker="*",s=120,c="orange",label="standby"); axis.set(title="M20 median typical allocation",xlabel="x",ylabel="y"); axis.legend(); fig.colorbar(scatter,ax=axis,label="belief"); fig.tight_layout(); fig.savefig(output/"figure_10_m20_typical_case.png",dpi=180); plt.close(fig)
