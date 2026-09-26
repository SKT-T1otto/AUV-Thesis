"""Read-only Phase2 preflight; never collect, train, create outputs, or resume.

The official Phase2 module already performs these checks before rollout. This
command exposes them without running an experiment and gives field-specific
errors. Keeping this launcher outside production packages preserves the exact
source identity of the existing frozen artifact; no source gate is bypassed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from chapter3_bser.experiments.hgr import phase2_gradient_efficiency as phase2

DEFAULT_CONFIG = phase2.ROOT / "configs/chapter3/hgr_phase2_gradient_efficiency.json"


class Phase2ConfigError(ValueError):
    """The experiment must not start until the stated input is corrected."""


def validate_phase2_config(config_path=DEFAULT_CONFIG):
    try:
        config = phase2.resolve_config(phase2.load_config(config_path))
    except Exception as exc:
        raise Phase2ConfigError(f"config/runtime_config invalid: {exc}") from exc
    runtime = config["runtime_config"]
    location = config.get("snapshot_source")
    if not isinstance(location, str) or not location.strip():
        raise Phase2ConfigError("snapshot_source must name an existing frozen-source file")
    source_path = Path(location).resolve()
    if not source_path.is_file():
        raise Phase2ConfigError(f"snapshot_source file does not exist: {source_path}")
    location = config.get("output_dir")
    if not isinstance(location, str) or not location.strip():
        raise Phase2ConfigError("output_dir must name a new or empty collision_terminal run directory")
    try:
        # Existing protocol gate, including refusal to reuse nonempty outputs.
        output = phase2.validated_output(runtime, location)
    except Exception as exc:
        raise Phase2ConfigError(
            f"output_dir invalid: {exc}. Use a new or empty directory such as "
            "outputs/chapter3/hgr_phase2/collision_terminal/<run_name>; "
            "resuming or appending to a previous run is not supported."
        ) from exc
    try:
        source = phase2.load_snapshot_source(source_path, runtime)
    except Exception as exc:
        raise Phase2ConfigError(
            f"snapshot_source rejected ({source_path}); requires {phase2.SOURCE_SCHEMA}, "
            f"matching source/runtime identities and valid policy payloads: {exc}"
        ) from exc
    if len(source.cases) != config["snapshot_count"]:
        raise Phase2ConfigError(
            f"snapshot_count={config['snapshot_count']} does not match source scenario count="
            f"{len(source.cases)}; use all predeclared scenarios, no runtime subsampling."
        )
    try:
        phase2._validate_source(source, config)
        # Constructor checks shared theta and policy identities on copies only.
        # It neither constructs MissionRuntime nor performs any rollout/update.
        phase2.Evaluation(config, source, output, phase2.RuntimeBackend())
    except Exception as exc:
        raise Phase2ConfigError(f"snapshot_source scenario/policy validation failed: {exc}") from exc
    return dict(
        status="PREFLIGHT_PASS", config=str(Path(config_path).resolve()),
        source_schema=phase2.SOURCE_SCHEMA, snapshot_source=str(source_path),
        snapshot_source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        source_identity_sha256=source.source_identity["sha256"],
        runtime_contract_sha256=phase2._digest(phase2.runtime_contract(runtime)),
        scope=config["scope"], snapshot_count=config["snapshot_count"],
        source_scenario_count=len(source.cases), repeat_count=config["repeat_count"],
        reference_rollouts=config["reference_rollouts"], output_dir=str(output),
        output_state="empty" if output.exists() else "not_created", existing_run_resume=False,
        environment_steps=0, experiment_started=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    try:
        report = validate_phase2_config(args.config)
    except Phase2ConfigError as exc:
        print(json.dumps(dict(status="PREFLIGHT_FAIL", error=str(exc), experiment_started=False),
                         ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
