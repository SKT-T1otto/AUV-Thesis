# tests

Current behavior: `python -B -m tools.verify_collision_terminal --suite current --output-dir docs/collision_terminal/maintenance/logs/NEW_CURRENT`.

Complete local acceptance: use `--suite all` with a new directory. It runs current,
fixed legacy compatibility, historical evidence, then frozen golden verification.
`--suite regression` remains an alias for current, not for historical evidence.

Fixed legacy reference and exact comparison fields are in `legacy_baseline.json`.
The historical source and current source run in separate temporary directories
and Python processes. Fixtures are test-only; no user checkpoint is loaded.

Small planner tests are grouped in `test_bser_objective_properties.py` and
`test_bser_planner_equivalence.py`. The current offline smoke and protocol checks
are in `test_bser_offline_protocol.py`. Original TestCase/method bodies remain.
Fixed report/sample/overlay checks live outside recursive tests discovery in
`historical_verification/`; use `--suite historical` to run that suite explicitly.

The cleanup method map and archive index are in
`docs/collision_terminal/maintenance/cleanup_manifest.json`.
