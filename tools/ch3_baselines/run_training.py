"""Manual configuration/provenance entry for existing B2/B3 trainers.

--check-only validates a training plan without constructing a trainer or model.
No estimator, update, checkpoint schema or production source is implemented here.
"""
import argparse
import json
from pathlib import Path
import traceback

from chapter3_bser.experiments.hgr.train import Trainer, validated_output
from chapter3_bser.experiments.hgr.provenance import checkout_identity
from .framework_provenance import framework_sources, verify_framework_sources
from .provenance import file_sha256, write_json, digest
from .registry import training_config, task_conditions


def training_plan(baseline, *, reference_training_config=None, output_dir=None):
    spec, reference_path, config = training_config(baseline, reference_training_config=reference_training_config, output_dir=output_dir)
    output = validated_output(config, config["output_dir"])
    sources = framework_sources()
    return dict(baseline=baseline, method=spec["method"], runtime_method=spec["runtime_method"],
        algorithm=spec["algorithm"], planner_mode=spec["planner"], learning_mode=spec["learning_mode"],
        config=config, output_dir=str(output), reference_config_path=str(reference_path),
        reference_config_sha256=file_sha256(reference_path),
        reference_kind="run_config_reference" if "outputs" in reference_path.parts else "default_config_reference",
        common_task_conditions=task_conditions(config), sources_before=sources,
        checkout_before=checkout_identity(), trainer="chapter3_bser.experiments.hgr.train.Trainer",
        checkpoint_schema=config["checkpoint_schema"], training_started=False)


def train(baseline, *, reference_training_config=None, output_dir=None, check_only=False):
    plan = training_plan(baseline, reference_training_config=reference_training_config, output_dir=output_dir)
    if check_only:
        return plan
    # Reached only through an explicit manual training invocation; the unified
    # evaluation CLI has no path to this function.
    output = Path(plan["output_dir"])
    verify_framework_sources(plan["sources_before"])
    if file_sha256(plan["reference_config_path"]) != plan["reference_config_sha256"]:
        raise ValueError("reference config changed before training")
    trainer = Trainer(plan["config"], output)
    plan["training_started"] = True
    plan["start_identity_sha256"] = digest(plan)
    write_json(output/"baseline_training_identity.json", plan)
    try:
        result = trainer.run()
        verify_framework_sources(plan["sources_before"])
        if file_sha256(plan["reference_config_path"]) != plan["reference_config_sha256"]:
            raise ValueError("reference config changed during training")
        plan["sources_after"] = framework_sources()
        plan["checkout_after"] = checkout_identity()
        plan["completed_main_trajectories"] = result["completed_main_trajectories"]
        plan["requested_main_trajectories"] = plan["config"]["total_main_trajectories"]
        plan["training_complete"] = plan["completed_main_trajectories"] >= plan["requested_main_trajectories"]
        plan["status"] = "complete" if plan["training_complete"] else "stopped_before_requested_main_count"
        write_json(output/"baseline_training_identity.json", plan)
        return result
    except (Exception, KeyboardInterrupt) as exc:
        plan.update(training_complete=False, status="failed", checkout_after=checkout_identity())
        write_json(output/"baseline_training_identity.json", plan)
        write_json(output/"baseline_training_failure.json", dict(method=plan["method"], runtime_method=plan["runtime_method"],
            message=str(exc), exception_type=type(exc).__name__, traceback=traceback.format_exc(), training_complete=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, choices=("B2_direct_mc", "B3_direct_boundary"))
    parser.add_argument("--reference-training-config", help="Optional actual HGR config; all parameters except method/algorithm/output are retained")
    parser.add_argument("--output-dir", help="New/empty collision_terminal directory; default is isolated per baseline")
    parser.add_argument("--check-only", action="store_true", help="Validate/print without constructing models, training or writing outputs")
    result = train(**vars(parser.parse_args(argv)))
    if "config" in result:
        print(json.dumps({k: result[k] for k in ("baseline", "method", "runtime_method", "algorithm", "output_dir", "reference_kind", "training_started")}, indent=2))
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
