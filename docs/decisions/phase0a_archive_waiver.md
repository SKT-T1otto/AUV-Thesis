# Phase 0A archive verification waiver

Recorded: `2026-08-02T09:31:09.178625+00:00`

The user accepts the current CH3/CH4/CH5 workspace snapshot as the development starting point for Phase 0B. External historical archive verification is waived; missing historical outputs will not be restored, and the user states that historical data has an independent backup.

This decision does **not** assert that the historical archive was completely verified. `LEGACY_BASELINE_V2_CANDIDATE.json` remains non-authoritative and must not be rewritten as a verified baseline.

Accepted missing paths:

- `CH3/data/chapter3/S_profiles_200ep_analysis_bundle.zip`
- `CH5/docs/code_directory_tree.md`
- All user-confirmed manually archived historical output roots recorded in the working baseline.

Required baseline labels:

- `USER_ACCEPTED_CURRENT_SNAPSHOT`
- `ARCHIVE_NOT_REVERIFIED`
- `NOT_A_COMPLETE_HISTORICAL_ARCHIVE`

This waiver authorizes new development only in `CRK-Thesis-v2`; CH3, CH4, CH5 and all Phase 0A evidence remain read-only.
