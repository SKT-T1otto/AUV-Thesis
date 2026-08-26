# Phase 0B-2 test report

## Baseline and migration equivalence

- Pre-migration Phase 0B-1 tests: 6/6 passed.
- Pre-migration E0: 60/60, maximum difference 0.0, event mismatches 0.
- Frozen snapshot versus migrated core E0: 60/60, maximum difference 0.0,
  event mismatches 0.
- Core versus frozen golden hashes with legacy hidden: 60/60, maximum
  difference 0.0, event mismatches 0.

## Unit and isolation tests

- Current worktree with legacy hidden: 17/17 passed.
- Git-tracked isolated copy without legacy: 17/17 passed.
- Fresh Git archive without `.git` or legacy: 17/17 passed.
- Required nine Phase 0B-2 tests are present and passed.
- Source provenance: 27/27 target paths and SHA256 values passed.
- Core Python Git tracking: 40/40.

## Training closure

- Device: CPU
- Episodes/steps: 2 x 10
- Transitions: 20; sampled batch: 8
- Critic and actor updates: passed and finite
- Safe checkpoint load: weights-only path used
- Checkpoint roundtrip and post-load step: passed
- Repository checkpoint persistence: none

One sandboxed test invocation initially failed because Windows denied access to
the AUV environment `_ctypes` DLL; the identical authorized command passed.
One initial Git archive run found EOL-sensitive provenance hashes; the LF
contract fix was followed by a fresh 17/17 archive pass and 60/60 E0 pass.
