# Standalone clone and archive test report

## Method B: tracked-file isolated copy

- Source: `git ls-files` from commit `6116ec5`
- Copied: 150 of 151 tracked entries
- Excluded: the complete `legacy_adapters` directory
- Sibling repositories present: no
- PYTHONPATH: temporary repository only
- Result: 17/17 tests passed

## Method C: Git archive

- Source: `git archive` from validation HEAD `6094ad8`
- `.git` present: no
- `legacy_adapters` present during execution: no
- Sibling repositories/data/checkpoints present: no
- Complete tests: 17/17 passed
- Explicit `MissionCoreEnv` reset/step: passed
- Four-profile scenario regeneration and frozen-hash comparison: passed
- Explicit CPU training smoke: 2 episodes x 10 steps, passed
- Checkpoint roundtrip and post-load step: passed
- Core-only E0: 60/60, maximum absolute difference 0.0, event mismatches 0

The first archive attempt exposed line-ending-dependent provenance hashes in
one test (the other 16 passed). Production imports and runtime behavior were
already valid. The issue was fixed by freezing `core` Python files to LF and
updating the seven affected target hashes. A fresh archive then passed the
provenance test and the complete validation above.
