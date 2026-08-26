# Repository role

- CRK-Thesis-v2 is the only writable production repository.
- `core` is the sole shared executable implementation.
- Chapter-specific algorithms belong in their chapter namespaces.
- CH3, CH4, and CH5 are read-only legacy references outside this repository.
- Runtime dependencies on legacy adapters or sibling repositories are forbidden.

# Current status

- Phase 0B-2 is a completed historical migration/equivalence baseline.
- Chapter 3 Phase 1A through Phase 1B.3a implementations and evidence exist.
- The independent Phase 1C method and trainer exist as
  `ch3_bser_rmaddpg_phase1c`.
- Phase 1C is **WIP / short training interrupted / resume-ready**.
- The planned 1000-episode short run stopped after episode 128; episode 100 is
  the last complete recovery checkpoint, so episodes 101-128 must be replayed.
- Phase 1C training, convergence, and formal comparisons are not complete.
- Chapter 4 RCAG and Chapter 5 VSGC remain placeholders in this repository.

# Hard restrictions

- Do not automatically start long training or restore a checkpoint.
- Do not mark Phase 1C, the 1000-episode run, or formal thesis evaluation
  complete.
- Do not modify frozen Phase 1B behavior or its historical experiment evidence.
- Preserve reward, observation, action, target, obstacle, success, handoff,
  dynamics, planner, actor, critic, replay, and network-dimension contracts.
- Do not place chapter-specific algorithms inside `core` unless they are true
  shared infrastructure.
- Do not reintroduce `sys.path` injection or legacy imports.
- Do not modify or delete user-retained files under `outputs`.
- Do not commit checkpoints, models, raw data, or long-run outputs.
- Do not run `git commit` or `git push` unless the user explicitly requests it.
- Existing Phase 0B-2 golden E0 data and historical phase evidence must not be
  overwritten or rewritten as current results.

# Provenance policy

- `docs/provenance/ch3_to_core_migration_manifest.json` is a frozen historical
  Phase 0B-2 baseline. Do not rewrite its hashes to match later code.
- All 27 historical provenance records must remain present and validated.
- The only currently permitted post-Phase-0B-2 provenance evolution is
  `core/registry/experiment_registry.py`, where the independent
  `ch3_bser_rmaddpg_phase1c` method was registered.
- That permitted evolution must preserve the seven legacy
  `ACTIVE_CH3_FINAL_EXPERIMENT_MODES`, remain outside that legacy active tuple,
  and be validated through the independent registered-method path.
- `tests/test_repository_metadata.py` pins both the historical Phase 0B-2 hash
  and the reviewed current hash for that one permitted evolution.
- Any additional provenance mismatch is a failure until explicitly reviewed.
- Never delete, skip, broadly exempt, or weaken provenance checks merely to
  make a test pass.

# Required verification

- For documentation-only or narrowly scoped changes, run task-scoped tests.
- For shared-core, environment-contract, training-runtime, or release-level
  changes, run the complete relevant regression suite.
- When shared core changes, preserve and run the frozen Phase 0B-2 E0
  equivalence checks where applicable.
- Keep the 28D observation, 3D action, and 124D centralized-critic contracts.
- Preserve all 27 source-provenance records and their historical hashes.
- Keep formal compact evidence visible to Git and checkpoint artifacts ignored.
- Use clean-clone verification for repository-closure or release checkpoints.
- Describe local test results as latest recorded local verification, not as CI.
- Do not launch formal long training merely as a verification step.
- Before reporting a Phase 1C run complete, require the planned terminal
  artifacts and explicit user-approved experiment execution.

# Phase 1C resume gate

Before treating the interrupted Phase 1C run as ready to resume:

- the working tree must be clean;
- the intended code commit must be identified;
- `tests.test_phase1c_guidance` must pass;
- `tests.test_observation_28d_contract` and
  `tests.test_phase1b2_path_tracking` must pass;
- `tests.test_repository_metadata` must pass under the historical-baseline plus
  explicit-permitted-evolution provenance contract;
- the episode 100 checkpoint must exist locally and load successfully;
- the user must explicitly choose to start or resume the long-running
  experiment.
