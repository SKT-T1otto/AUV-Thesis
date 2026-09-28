"""Bounded B0/B1 current-source diagnostic probe; no checkpoints or training.

Not a historical replay or formal evaluation. Uses an exported scene/config,
records source differences, and optionally proves observer noninterference by
running the same bounded prefix once without observers. Maximum 100 steps.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import dataclasses
import json
import os
from pathlib import Path
import random
import sys

from scripts.analyze_search_failures import ROOT, read_json, file_hash, digest
from scripts.search_diagnostic_observer import QueryTap, TransitionObserver, capture_runtime


def jsonable(value):
    if dataclasses.is_dataclass(value):return jsonable(dataclasses.asdict(value))
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    if hasattr(value,"detach"):return value.detach().cpu().tolist()
    if hasattr(value,"tolist"):return value.tolist()
    return value


def physical_signature(runtime,record):
    import numpy as np
    import torch
    value=dict(agents=jsonable(runtime.env.get_agent_state()),task=jsonable(runtime.env.get_task_state()),
               observations=jsonable(runtime.observations),reward_record=jsonable(record),
               guidance=jsonable(runtime.guidance),python_rng=repr(random.getstate()),
               numpy_rng=jsonable(np.random.get_state()),torch_rng=torch.get_rng_state().tolist())
    return digest(value)


def run_probe(run_dir,output,*,episode_index=0,steps=30,verify_noninterference=False,live_public_map=False):
    if steps<1 or steps>100:raise ValueError("diagnostic probe is bounded to 1..100 steps")
    if live_public_map and not verify_noninterference:
        raise ValueError("live public-map extraction requires a paired noninterference check")
    run_dir=Path(run_dir).resolve();output=Path(output).resolve()
    if output==run_dir or run_dir in output.parents or (ROOT/"3090结果") in output.parents or (ROOT/"outputs") in output.parents:
        raise ValueError("probe cannot write into retained input/output directories")
    if output.exists() and any(output.iterdir()):raise ValueError("output must be new or empty")
    config=read_json(run_dir/"resolved_config.json")
    if config.get("baseline") not in {"B0_search_prior","B1_bser_prior"}:
        raise ValueError("probe only supports zero-policy B0/B1; no checkpoint loading")
    manifest=read_json(run_dir/"evaluation_manifest.json")
    if not 0<=episode_index<len(manifest["scenarios"]):raise ValueError("episode index outside manifest")
    scenario=manifest["scenarios"][episode_index]
    seed=int(config["seed"])+episode_index
    expected_identity=read_json(run_dir/"identity.json")["sources_before"]["production"]
    # Import only after validating the explicitly bounded request.
    from tools.ch3_baselines.framework_provenance import framework_sources, verify_framework_sources
    from tools.ch3_baselines.basic_search_prior import BasicSearchPriorRuntime
    from tools.ch3_baselines.bser_prior import BSERPriorRuntime
    from core.mapping.travel_cost_service import TravelCostService
    from core.mapping.planning_state import extract_planning_state
    from chapter3_bser.experiments.hgr.provenance import fresh_source_identity
    import numpy as np
    import torch
    sources=framework_sources()  # Existing source gates are mandatory and unchanged.
    current=fresh_source_identity()
    changed=sorted(k for k in set(current["files"])|set(expected_identity["files"])
                   if current["files"].get(k)!=expected_identity["files"].get(k))
    runtime_class=BasicSearchPriorRuntime if config["baseline"]=="B0_search_prior" else BSERPriorRuntime
    input_files={n:file_hash(run_dir/n) for n in ("resolved_config.json","evaluation_manifest.json","identity.json")}
    output.mkdir(parents=True,exist_ok=True)
    def write(name,value):
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    identity=dict(schema="ch3.search_diagnostic_probe.v1",mode="current_source_mechanism_probe",
                  historical_trajectory_reproduction=False,formal_evaluation=False,training=False,
                  checkpoint_loaded=False,scenario_id=scenario["scenario_id"],scenario_seed=scenario["scenario_seed"],
                  environment_innovation_seed=seed,episode_index=episode_index,max_physical_steps=steps,
                  source_before=current,recorded_source_sha256=expected_identity["sha256"],
                  different_source_files=changed,input_files=input_files,
                  historical_source_bytes_match=not changed,live_public_map=live_public_map,
                  diagnostic_scripts={name:file_hash(ROOT/"scripts"/name) for name in
                      ("run_search_diagnostic_probe.py","search_diagnostic_observer.py","analyze_search_failures.py")},
                  runtime_environment=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,
                      torch_num_threads=torch.get_num_threads(),omp_num_threads=os.environ.get("OMP_NUM_THREADS")),
                  caveat="Source gate passed; source equality alone does not establish historical trajectory equality.")
    write("identity.json",identity)
    observer=TransitionObserver()
    def execute(observed):
        signatures=[];runtime=None;tap=QueryTap(TravelCostService) if observed else None
        try:
            with tap if observed else nullcontext():
                runtime=runtime_class(config["native_runtime_config"],scenario,seed=seed,episode_id=episode_index)
                signatures.append(physical_signature(runtime,None))
                if observed:write("initial_queries.json",tap.drain())
                with (output/"step_trace.jsonl").open("w",encoding="utf-8") if observed else nullcontext() as handle:
                    for _ in range(steps):
                        if runtime.terminal:break
                        before=capture_runtime(runtime,extract_planning_state(runtime.env) if live_public_map else None) if observed else None
                        record=runtime.advance()
                        if observed:
                            after=capture_runtime(runtime,extract_planning_state(runtime.env) if live_public_map else None)
                            result=runtime.env.get_episode_result() if runtime.terminal else None
                            row=observer.observe(before,after,tap.drain(),result)
                            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n");handle.flush()
                        signatures.append(physical_signature(runtime,record))
                        if runtime.step%10==0:print(f"DIAGNOSTIC_PROGRESS observed={observed} step={runtime.step}",flush=True)
                result=dict(physical_steps=runtime.step,terminal=runtime.terminal,found=runtime.found_step is not None,
                            found_step=runtime.found_step,controller=runtime.controller_diagnostics(),
                            episode_result=runtime.env.get_episode_result())
                return signatures,result
        finally:
            if runtime is not None:runtime.close()
    try:
        baseline_signatures=None
        if verify_noninterference:baseline_signatures,_=execute(False)
        signatures,result=execute(True)
        verification=dict(requested=verify_noninterference,
                          equal_at_every_step=signatures==baseline_signatures if verify_noninterference else None,
                          signature_covers="public physical state, observations, rewards, guidance, Python/NumPy/Torch RNG")
        if verify_noninterference and signatures!=baseline_signatures:
            raise RuntimeError("observer changed bounded trajectory or RNG")
        verify_framework_sources(sources)
        if fresh_source_identity()!=current:raise RuntimeError("production sources changed during probe")
        if input_files!={n:file_hash(run_dir/n) for n in input_files}:raise RuntimeError("input evidence changed")
        write("summary.json",dict(result,telemetry=observer.result(),observer_noninterference=verification,
                                   probe_completed=True,formal_evaluation=False,performance_passed=None))
        print(json.dumps(dict(output=str(output),steps=result["physical_steps"],noninterference=verification),ensure_ascii=True))
    except BaseException as error:
        write("failure.json",dict(type=type(error).__name__,message=str(error),probe_completed=False))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--episode-index",type=int,default=0)
    p.add_argument("--steps",type=int,default=30)
    p.add_argument("--current-source-probe",action="store_true",required=True,
                   help="Explicitly acknowledges a mechanism probe, not a historical replay")
    p.add_argument("--verify-noninterference",action="store_true")
    p.add_argument("--live-public-map",action="store_true",
                   help="Record fresh public occupancy for sensing-delay diagnosis; never supplied to the controller")
    a=p.parse_args();run_probe(a.run_dir,a.output_dir,episode_index=a.episode_index,steps=a.steps,
                             verify_noninterference=a.verify_noninterference,live_public_map=a.live_public_map)


if __name__=="__main__":main()
