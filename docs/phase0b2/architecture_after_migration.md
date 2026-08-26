# Architecture after migration

`core` is a normal Python package and does not modify `sys.path`. The public
`MissionCoreEnv` facade constructs `core.env.uav_env.UAVEnv` directly and
reports `implementation_source = "core"`.

```text
core/
|-- env/             mission facade, UAV environment, target motion, contracts
|-- config/          Chapter-3 constants and configuration builder
|-- mapping/         occupancy mapping and path planning
|-- communication/   fixed reliable Chapter-3 communication
|-- algorithms/      MADDPG, agents, networks, noise, shared helpers
|-- replay/          Chapter-3 replay buffer
|-- registry/        method and experiment registries
|-- runtime/         builder, engine, training helpers, metrics
|-- scenarios/       schema, registry, manifest, deterministic generator
|-- provenance/      source/run provenance helpers
|-- evaluation/      reserved public boundary
`-- wrappers/        reserved shared-wrapper boundary
```

`chapter3_bser`, `chapter4_rcag`, and `chapter5_vsgc` remain separate chapter
boundaries. They were not implemented or migrated in Phase 0B-2.
