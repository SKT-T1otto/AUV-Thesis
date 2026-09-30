# D2 four-arm orchestration source review

Authorization: 2026-09-30 user request to implement unified D2, D2+B2,
D2+B3 and D2+HGR configurations, launchers and result summaries. Formal
experiments remain manual; local validation uses bounded synthetic smoke only.

Two existing executable files change: the exact successor dispatch in
`safe_search_v1/provenance.py`, and the baseline training-manifest protocol
validator, which now accepts the generator's canonical `ch3_unknown_map_v1`
alongside the retained historical fixture spelling `CH3_UNKNOWN_MAP_V1`.
It does not transform scenes or accept other protocols. Ten new executable/configuration files are
enumerated in `d2_suite_v1/provenance.py`. No core, planner, dynamics, reward,
network, replay or training-algorithm changes are included. The prior D2
seal, frozen reference and all preceding historical seals remain unchanged.
The successor validates every complete byte profile, exact before/after hashes,
parent manifest bytes and this review. No provenance mismatch is exempted.

The preparation layer freezes scene copies, training configurations, seeds,
task conditions, evaluation policy modes and source identity. It rejects
train/evaluation overlap by ID, seed or identity-stripped full scene content.
It cannot establish whether a person previously tuned on an external scene.
Generation requires an explicit flag. Training uses native from-scratch
trainers and counts all native environment-step categories including HGR
branches. Started episodes/cycles finish before stopping; overshoot is reported.

The evaluation layer invokes the existing native runtimes with immutable
checkpoint parameters or a zero-residual D2 policy. Public state is read for
kinematic search diagnostics; this adds no control or reward transformations.
All episodes must reach authoritative full-task terminal outcomes. A collision
at the discovery step counts as pre-Found. Missing Found receives horizon 400
in the penalized-time metric. Conditional discovery times are secondary.

Execution requires `--execute`; existing unfinished output directories are not
overwritten or resumed. Completed receipts bind artifacts and final checkpoint
bytes. Summaries reject mixed scenario order, seeds and identities. Partial
jobs receive no rates. The single D2 reference is not counted as independent
replicates when reused against three training seeds. No performance conclusion
or formal-run completion is established by these interface checks.
