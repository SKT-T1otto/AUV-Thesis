"""Deterministic E0 manifest generation from the self-contained core builder."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict

from .manifest import sha256_json, write_manifest
from .registry import PROFILES


PROJECT_ROOT = Path(__file__).resolve().parents[2]

from .ch3_generator_impl import build_scenario_manifests


E0_SEEDS = (1729, 1730, 1731, 1732, 1733)


def build_e0_manifests() -> Dict[str, dict]:
    generated = build_scenario_manifests(
        count=5,
        generator_seed=1729,
        split="validation",
        profiles=tuple(PROFILES),
    )
    result = {}
    for profile, manifest in generated.items():
        value = deepcopy(manifest)
        seeds = tuple(int(item["scenario_seed"]) for item in value["scenarios"])
        if seeds != E0_SEEDS:
            raise RuntimeError(f"unexpected E0 seeds for {profile}: {seeds}")
        for scenario in value["scenarios"]:
            scenario["scenario_sha256"] = sha256_json(scenario)
        value["manifest_sha256"] = sha256_json(value)
        result[profile] = value
    if set(result) != set(PROFILES):
        raise RuntimeError("CH3 generator did not return all four E0 profiles")
    return result


def generate_e0_manifests(output_dir: Path | None = None) -> Dict[str, dict]:
    output_dir = output_dir or PROJECT_ROOT / "configs" / "scenarios" / "e0_equivalence"
    manifests = build_e0_manifests()
    for profile, value in manifests.items():
        write_manifest(output_dir / f"{profile}.json", value)
    index = {
        "experiment": "CH3-E0 Canonical Environment Migration Equivalence",
        "generator_seed": 1729,
        "scenario_seeds": list(E0_SEEDS),
        "profiles": {
            name: {
                "path": f"configs/scenarios/e0_equivalence/{name}.json",
                "manifest_sha256": value["manifest_sha256"],
                "scenario_count": len(value["scenarios"]),
            }
            for name, value in manifests.items()
        },
    }
    write_manifest(output_dir / "scenario_manifest_index.json", index)
    return manifests
