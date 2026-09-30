"""Read-only D2 source/config preflight. Never constructs a trainer or simulator."""
import json
from .provenance import framework_sources


def main():
    from tools.ch3_baselines.registry import training_config, load_reference, task_conditions
    sources = framework_sources()
    path, reference = load_reference()
    methods = {"HGR": dict(config=str(path), planner_protocol=reference["planner_protocol"])}
    for name in ("B2_direct_mc", "B3_direct_boundary"):
        spec, _, config = training_config(name)
        if task_conditions(config) != task_conditions(reference):
            raise ValueError("D2 common task mismatch")
        methods[name] = dict(config=spec["training_config"], planner_protocol=config["planner_protocol"])
    print(json.dumps(dict(source_inventory_sha256=sources["inventory"]["sha256"],
                          historical_records=sources["historical_record_count"], methods=methods,
                          formal_experiment_started=False), indent=2))


if __name__ == "__main__":
    main()
