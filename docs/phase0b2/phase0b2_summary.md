# Phase 0B-2 summary

Final status: `PASS_CORE_SELF_CONTAINED_NO_LEGACY_DEPENDENCY`

`core` is now the sole executable source for the Chapter-3 mission environment,
scenario generation, mapping, communication, MADDPG, replay, registries,
runtime/training helpers, metrics, and provenance. All 27 authority-source
records were migrated to 27 tracked core targets without semantic changes.

The executable snapshot was removed. `legacy_adapters` contains only a README
and can be deleted without affecting imports, tests, training smoke, scenario
generation, or E0 replay.

## Acceptance results

- Legacy snapshot versus core E0: 60/60; maximum absolute difference 0.0;
  task-event mismatches 0.
- Core versus frozen golden data with legacy hidden: 60/60; maximum absolute
  difference 0.0; task-event mismatches 0.
- Current-worktree tests with legacy hidden: 17/17.
- Git-tracked isolated copy without legacy or sibling repositories: 17/17.
- Git archive without `.git`, legacy, siblings, data, or checkpoints: 17/17,
  explicit reset/step and scenario generation passed, 2x10 training smoke
  passed, and core-only E0 passed 60/60.
- Training closure: 20 transitions, replay sample, finite critic and actor
  updates, weights-only checkpoint roundtrip, and post-load step all passed.
- Static production dependency counts: legacy imports 0, `sys.path` injections
  0, sibling CH3/CH4/CH5 dependencies 0.
- Legacy repositories: CH3 369/369, CH4 103/103, CH5 312/312 files byte-equal
  to `WORKING_BASELINE_V2`.

`allow_bser_phase1a = true`. No BSER, RCAG, VSGC, CH4, or CH5 implementation
was performed in this phase.
