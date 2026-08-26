# Legacy removal report

Removed after the legacy-vs-core 60/60 E0 gate and the legacy-hidden 17/17 test
gate:

- `legacy_adapters/ch3_snapshot/`: 30 files, including 29 Python files and the
  source-hash metadata file.
- Obsolete one-time snapshot creation, baseline capture, migration, golden
  freezing, Phase 0B-1 finalization, and dual-runtime E0 scripts.

The source-hash evidence remains in
`docs/provenance/ch3_authority_source_hashes_v1.json`, and all 27 migration
records remain in `docs/provenance/ch3_to_core_migration_manifest.json`.

Final `legacy_adapters` contents: `README.md` only. Python code count: 0.
The directory can be deleted wholesale without affecting runtime or tests.
Removed files remain recoverable from earlier commits on the Phase 0B-2 branch.
