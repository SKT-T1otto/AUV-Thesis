# Latest recorded local verification — 2026-09-22

Historical pre-manifest record: the missing-file blocker below is superseded by
the explicitly authorized fix in [provenance_fix.md](provenance_fix.md). The
earlier test outcomes are retained here unchanged.

Status: implementation written; framework activation and complete acceptance
remain blocked on explicit approval of the additional provenance evolution.
This is local verification, not CI or formal experiment evidence.

| Check | Recorded result |
| --- | --- |
| Independent B2/B3 real training, boundary features, checkpoint identity, unchanged priors | 5 checks passed in 160.857 s; B2/B3 each completed one synthetic four-step mission and three optimizer batches |
| Current-tree configuration, method isolation, task fairness, moved B0/B1 guard | 5 checks passed in 0.196 s; one repeats the prior B0/B1 check, so 9 distinct checks across these two runs |
| Additional direct trainer smoke | Each baseline completed one two-step temporary mission; actual actor updates and exact model/optimizer state round-trip verified |
| Requested validation + provenance + repository metadata command | 21 tests ran in 18.850 s; 25 error reports, all due to the intentionally absent unapproved `baseline_maddpg_evolution.json`; repository metadata checks passed |
| Wider environment/reward/runtime/HGR regressions | 40 individual checks recorded as passed before the additional run was manually interrupted while framework approval remained pending; remaining checks were not completed, so this is not a full-suite pass. See `shared_regression.log` |
| Whitespace check | `git diff --check` passed |
| Protected legacy code and evidence | No diff under `core`, HGR experiment/model directories, retained `outputs`, or historical production/provenance manifests |

The actual learning tests exercised uniform core replay; updates for all four
actors and both critics per agent; B3 encoder gradients; next-state boundary
conditioning in target policies; full checkpoint model/optimizer restoration;
and HGR, cross-baseline and missing optimizer-state rejection. Test checkpoints
were temporary and removed by the tests.

`test_real_checkpoints_evaluate_without_optimizer_updates` is added, unskipped,
and still pending execution after source-gate approval. Positive unified
400-step evaluation acceptance is also pending. Check-only and other framework
entry tests cannot reach their intended checks while provenance activation is
blocked; this report does not claim they passed.

Automatic approval review rejected creating the active provenance evolution
manifest because it changes the hashes accepted by integrity checks, and
`AGENTS.md` requires explicit review for additional provenance mismatches. The
requested approval is limited to the 14 new baseline production files and 9
necessary framework/configuration files. All historical records and hashes stay
frozen. No gate was bypassed and the active evolution manifest was not written.

No formal 1000-episode training, performance evaluation, checkpoint restoration
of a retained user run, commit or push was performed.
