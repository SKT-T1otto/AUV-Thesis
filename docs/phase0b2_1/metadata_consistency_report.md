# Metadata consistency report

## Updated metadata

- Root `README.md` now states Phase 0B-2 complete and Phase 0B-2.1 closure.
- `AGENTS.md` now freezes the current repository role and verification rules.
- `SOURCE_MANIFEST.json` uses portable schema v2, phase 0B-2.1, and no machine
  path as a runtime dependency.
- Root `.gitattributes` enforces LF for production/chapter/test/tool Python and
  provenance JSON, while preserving CRLF for Windows scripts.
- Root `.gitignore` excludes caches, local environments, IDE state, models,
  checkpoints, runtime output, and local data without ignoring formal evidence.
- Core and chapter README files describe implemented and reserved boundaries
  without claiming BSER, RCAG, or VSGC completion.

## Verification

- Metadata unit tests: 3/3 passed.
- Formal summary/manifest ignore probes: all not ignored.
- Temporary `.pt` probe: ignored by `*.pt`.
- `core/env` future source probe: not ignored.
- Required `git check-attr` probes: `text=set`, `eol=lf` locally and in the
  GitHub clone.
- Source provenance after metadata changes: 27/27.
