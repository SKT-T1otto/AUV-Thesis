# B2/B3 provenance missing-file repair — 2026-09-22

The explicitly requested `docs/provenance/baseline_maddpg_evolution.json` is now
present. It records the independent B2/B3 algorithm, architecture and trainer
identities, B3's boundary encoder schema, exact hashes for the 14 added baseline
production files, and the 9 framework changes required by the existing gate.
Historical production/protected manifests and all 27 migration records remain
unchanged. No `core/` or existing `chapter3_bser/` code changed in this repair.

The new provenance test requires the file to exist, validates its aggregate
hash through the production loader, checks the requested B2/B3 identities, and
compares algorithm/architecture fields with the actual training configurations.
Existing tests continue to reject modified, missing and additional unreviewed
sources. The earlier blocked verification record is retained in
`implementation_status.md` as historical evidence.

Latest recorded local verification (not CI):

| Command | Result |
| --- | --- |
| `python -B -m unittest -v tests.test_ch3_baseline_validation tests.test_ch3_baseline_provenance` | **PASS**, 26 tests in 419.182 seconds |
| `python -B -m unittest -v tests.test_repository_metadata` | **PASS**, 6 tests in 0.605 seconds |
| `git diff --check` | **PASS** |

Tests used `D:\anaconda\anaconda\envs\AUV\python.exe` with
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. The requested suite's complete output
is saved in [provenance_fix.log](provenance_fix.log). Its B0/B1 missions are
synthetic, bounded interface checks; all generated runtime artifacts were
temporary. No formal B2/B3 training or thesis performance evaluation ran.

This resolves the missing-manifest/source-gate failure. It does not turn earlier
unrun training/evaluation experiments into completed results.
