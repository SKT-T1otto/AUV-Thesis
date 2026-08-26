# Import dependency report

Scope: recursive Python scan of `core`, chapter package boundaries, `tools`,
and `tests`, plus AST import resolution for every absolute `core.*` import.

## Result

- Production references to `legacy_adapters`: 0
- Production references to `ch3_snapshot`: 0
- Production references to `crk_core`: 0
- Production `sys.path.insert` or `sys.path.append`: 0
- Production references to sibling CH3/CH4/CH5 paths: 0
- Unresolved absolute `core.*` imports: 0
- `MissionCoreEnv` implementation module: `core.env.uav_env`
- Scenario generator implementation module: `core.scenarios.ch3_generator_impl`

Two tests intentionally contain historical directory names as forbidden tokens
or deletion assertions. They do not import, execute, or require those paths.
The production-code raw scan has zero matches.
