"""Run one Phase 1B.1 pilot seed as independent condition episodes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from chapter3_bser.experiments.phase1b1_pilot.run_pilot import (
    METHODS,
    _worker,
    _write_json,
)
from chapter3_bser.online.config import load_phase1b1_config
from core.scenarios.ch3_generator_impl import build_scenario_manifests


def run_seed(seed: int, output_dir: Path, workers: int) -> Path:
    config = load_phase1b1_config()
    seeds = tuple(int(value) for value in config["experiment"]["scenario_seeds"])
    if seed not in seeds:
        raise ValueError(f"seed {seed} is not in the Phase 1B.1 pilot protocol")
    indices = tuple(int(value) for value in config["experiment"]["episode_indices"])
    max_steps = int(config["experiment"]["max_steps"])
    manifest = build_scenario_manifests(
        count=5,
        generator_seed=seeds[0],
        split="validation",
        profiles=("M20_MOVING_UNKNOWN_MULTI",),
    )["M20_MOVING_UNKNOWN_MULTI"]
    scenarios = {int(row["scenario_seed"]): row for row in manifest["scenarios"]}
    jobs = [
        (method, scenarios[seed], episode_index, max_steps)
        for episode_index in indices
        for method in METHODS
    ]
    metrics = []
    steps = []
    failures = []
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 4))) as executor:
        future_map = {
            executor.submit(_worker, job): (job[0], job[2])
            for job in jobs
        }
        for future in as_completed(future_map):
            method, episode_index = future_map[future]
            try:
                metric, diagnostics = future.result()
                metrics.append(metric)
                steps.extend(diagnostics)
            except Exception as exc:
                failures.append(
                    {
                        "method": method,
                        "scenario_seed": seed,
                        "episode_index": episode_index,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
    order = {method: index for index, method in enumerate(METHODS)}
    metrics.sort(key=lambda row: (order[row["method"]], int(row["episode_index"])))
    steps.sort(
        key=lambda row: (
            order[row["method"]],
            int(row["episode_index"]),
            int(row["step"]),
        )
    )
    checkpoint = {
        "schema": "bser.phase1b1.pilot.seed_checkpoint.v1",
        "scenario_seed": seed,
        "max_steps": max_steps,
        "episode_indices": list(indices),
        "methods": list(METHODS),
        "metrics": metrics,
        "steps": steps,
        "failures": failures,
    }
    path = Path(output_dir) / f"seed_{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, checkpoint)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    path = run_seed(args.seed, args.output_dir, args.workers)
    print(path)


if __name__ == "__main__":
    main()
