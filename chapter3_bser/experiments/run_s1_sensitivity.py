"""One-factor-at-a-time BSER sensitivity analysis on fixed-hash M20 states."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from chapter3_bser.candidate_generator import generate_candidates
from chapter3_bser.config import load_bser_phase1a1_config
from chapter3_bser.exact_solver import solve_joint_exact
from chapter3_bser.experiments.instance_builder import SkippedSnapshot, iter_e1_snapshots
from chapter3_bser.greedy_solver import solve_joint_greedy
from chapter3_bser.objective import build_objective_context, evaluate_objective, expected_detection_probability, response_diagnostics


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "experiments" / "chapter3" / "bser_s1_sensitivity"


def _write_csv(path, rows):
    fields=sorted({key for row in rows for key in row}) if rows else ["status"]
    with Path(path).open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _jaccard(left, right):
    a=set(left.selected_ids); b=set(right.selected_ids)
    return 1.0 if not a and not b else len(a & b)/len(a | b)


def _displacement(left, right):
    by_agent={candidate.agent_id:np.asarray(candidate.waypoint) for candidate in left.selected}
    values=[float(np.linalg.norm(np.asarray(candidate.waypoint)-by_agent[candidate.agent_id])) for candidate in right.selected if candidate.agent_id in by_agent]
    return statistics.fmean(values) if values else 0.0


def _collect_states():
    unique={}
    for item in iter_e1_snapshots():
        if not isinstance(item,SkippedSnapshot) and item.profile=="M20_MOVING_UNKNOWN_MULTI":
            unique.setdefault(item.unique_state_sha256,item)
    return [item for _,item in sorted(unique.items())]


def run(states=None, output_dir=DEFAULT_OUTPUT, *, smoke=False):
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    config=load_bser_phase1a1_config(); specification=config["sensitivity"]
    available=list(_collect_states() if states is None else states)
    selected=sorted(available,key=lambda item:item.unique_state_sha256)[:(1 if smoke else int(specification["fixed_hash_unique_state_count"]))]
    rows=[]
    for item in selected:
        baseline=None
        for variant in specification["variants"]:
            varied=copy.deepcopy(config)
            varied["detection_model"]["p_scale"]=float(variant["p_scale"])
            varied["detection_model"]["sigma_sensor_radius_multiplier"]=float(variant["sigma_multiplier"])
            varied["objective"]["tau_executor"]=float(variant["tau_executor"])
            generation=generate_candidates(item.state,varied)
            context=build_objective_context(item.state,generation.search_candidates,generation.standby_candidates,varied)
            greedy=solve_joint_greedy(generation.search_candidates,generation.standby_candidates,context)
            exact=solve_joint_exact(generation.search_candidates,generation.standby_candidates,context,combination_limit=varied["solver"]["exact_combination_limit"])
            diag=response_diagnostics(greedy.selected,greedy.standby,context)
            if variant["id"]=="baseline": baseline=greedy
            if baseline is None: raise RuntimeError("baseline sensitivity variant must be first")
            rows.append({
                "unique_state_sha256":item.unique_state_sha256,"variant_id":variant["id"],
                "p_scale":variant["p_scale"],"sigma_multiplier":variant["sigma_multiplier"],"tau_executor":variant["tau_executor"],
                "selected_candidate_ids":";".join(greedy.selected_ids),"standby_candidate_id":greedy.standby.candidate_id,
                "detection_probability":expected_detection_probability(greedy.selected,context),
                "joint_objective":evaluate_objective(greedy.selected,greedy.standby,context),
                "conditional_reachable_response_time":"" if not diag.response_defined else diag.conditional_reachable_response_time,
                "unreachable_detected_mass":diag.unreachable_detected_mass,
                "allocation_jaccard_vs_baseline":_jaccard(baseline,greedy),
                "mean_waypoint_displacement_vs_baseline":_displacement(baseline,greedy),
                "standby_waypoint_displacement_vs_baseline":float(np.linalg.norm(np.asarray(greedy.standby.waypoint)-np.asarray(baseline.standby.waypoint))),
                "greedy_exact_ratio":1.0 if exact.objective<=1e-12 else greedy.objective/exact.objective,
                "exact_status":exact.status,
            })
    aggregates=[]
    for variant in specification["variants"]:
        group=[row for row in rows if row["variant_id"]==variant["id"]]
        aggregates.append({"variant_id":variant["id"],"state_count":len(group),"mean_detection_probability":statistics.fmean(row["detection_probability"] for row in group) if group else "","mean_joint_objective":statistics.fmean(row["joint_objective"] for row in group) if group else "","mean_unreachable_detected_mass":statistics.fmean(row["unreachable_detected_mass"] for row in group) if group else "","mean_allocation_jaccard_vs_baseline":statistics.fmean(row["allocation_jaccard_vs_baseline"] for row in group) if group else ""})
    _write_csv(output/"sensitivity_results.csv",rows); _write_csv(output/"aggregate_by_variant.csv",aggregates)
    manifest={"schema":"bser.s1.experiment.v1","selection_rule":"first 20 valid unique M20 states in SHA-256 order","selected_state_hashes":[item.unique_state_sha256 for item in selected],"variants":specification["variants"],"ofat":True,"defaults_tuned_from_results":False,"formal_training":False,"smoke":bool(smoke)}
    (output/"experiment_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    core={"schema":"bser.s1.summary.v1","selected_state_count":len(selected),"variant_count":len(specification["variants"]),"result_count":len(rows),"defaults_tuned_from_results":False,"passed":len(selected)==(1 if smoke else 20) and len(rows)==len(selected)*7 and all(row["exact_status"]=="OK" for row in rows)}
    core["deterministic_summary_sha256"]=hashlib.sha256(json.dumps(core,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    (output/"s1_summary.json").write_text(json.dumps(core,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (output/"s1_summary.md").write_text(f"# BSER S1 sensitivity\n\nStatus: **{'PASS' if core['passed'] else 'FAIL'}**\n\nSeven OFAT variants were evaluated on {len(selected)} fixed-hash M20 states. Results did not alter defaults.\n",encoding="utf-8")
    (output/"test_report.md").write_text(f"# S1 test report\n\nStatus: **{'PASS' if core['passed'] else 'FAIL'}**\n",encoding="utf-8")
    return core


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT); parser.add_argument("--smoke",action="store_true"); args=parser.parse_args(); summary=run(output_dir=args.output_dir,smoke=args.smoke); print(json.dumps(summary,sort_keys=True)); raise SystemExit(0 if summary["passed"] else 1)


if __name__=="__main__": main()
