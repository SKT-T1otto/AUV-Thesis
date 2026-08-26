# GitHub self-containment report

- Branch validated: `phase0b2-core-self-contained`
- Validation HEAD: `6094ad88bf1051e9fa42891c998832259c174902`
- Core Python files present: 40
- Core Python files tracked: 40
- Untracked core Python files: 0
- Executable legacy Python files: 0
- Checkpoints added by Phase 0B-2: 0
- Formal training outputs added: 0
- `.gitattributes` freezes `core` Python files to LF so provenance hashes are
  identical in Windows worktrees and `git archive` exports.

A Git archive of the validated HEAD, with the optional legacy README directory
removed, passed all 17 tests, environment reset/step, scenario regeneration,
2x10 training smoke, checkpoint roundtrip, and 60-trajectory core-only E0.
The archive had no `.git` directory and no access to sibling repositories.
