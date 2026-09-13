# Historical evidence checks

Run from the repository root:

`python -B -m tools.verify_collision_terminal --suite historical --output-dir docs/collision_terminal/maintenance/logs/NEW_HISTORICAL`

This group preserves the fixed Phase 1B.3a case list, Phase 1A-v1 delivery blob
hashes, CH3 E0 delivery report assertions, and v2 overlay manifest boundary.
Original test methods are mapped in the maintenance cleanup manifest. The v1
blob reference is pinned to `93a9c8fb53857051390265e3035061bf05402e25`.

Missing original inputs are reported as `blocked_missing_historical_input`,
with precise paths and nonzero exit status, separately from current behavior.
Restore real files from an identified backup and verify their frozen identities;
do not generate replacements or update historical hashes. File-based E0/overlay
checks retain their original paths below the repository root. Blob checks need
the actual historical Git object, not an unrelated current CSV.

Frozen trajectory execution is separate: use `--suite golden`. `--suite all`
does not succeed when any required historical or golden input is blocked.
