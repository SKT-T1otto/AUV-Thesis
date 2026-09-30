# SafeSearch v1 source review

Review date: 2026-09-28. Historical comparison: B0/B1 commit
`a66c88defc55f0342588cbefecbb838ffb6eec5a`; present pre-repair checkout:
`44c86cd`. This review authorizes a new opt-in experiment inventory only. It
does not rewrite the original experiment identity or permit the old baseline
gate to accept modified source.

## Existing HGR evolution

The nine production differences were checked against the original Git content,
the fixed hashes in `experiment_plan.json`, and the changes introduced by
`89165a4` and `16ae663`. They are retained and pinned individually.

| File under Chapter 3 | Reviewed change | Effect on B0/B1 zero-policy evaluation |
| --- | --- | --- |
| `experiments/hgr/build_phase2_source.py` | Explicit policy-pair source builder; validates supplied source, policy and runtime identity. No implicit training. | Separate entry; not invoked. |
| `experiments/hgr/phase1_acceptance.py` | Manual bounded acceptance for opt-in phase1, including branch replay and spawned-process comparison. | Separate entry; not invoked. |
| `experiments/hgr/phase2_gradient_efficiency.py` | Frozen-policy Monte Carlo/score-gradient diagnostic with explicit source and step accounting. | Separate entry; not invoked. |
| `experiments/hgr/runtime.py` | Adds opt-in phase1 validation, RNG isolation, addressed noise, branch identity/trace metadata and CPU-only seed mode. | Imported and used. With no `phase1` config, seeding retains `torch.manual_seed`, supplied noise is absent, and the same policy/action/physics path executes. Requires disabled-variant trajectory comparison; source similarity alone is not the verification. |
| `experiments/hgr/train.py` | Adds opt-in phase1 checkpoint revisions, named streams, stable predictor, branch pairing and no-update proof. | Trainer is not constructed. No optimizer/checkpoint execution in SafeSearch. |
| `models/hgr/estimator.py` | Optional, proof-validated identical-suffix bypass. Default `bypass=None` keeps previous positive-K and loss behavior. | No estimator or optimizer calls. |
| `models/hgr/phase1.py` | Explicit config contracts, behavior identity, no-update proof, addressed noise and RNG audits. Import only defines objects. | `phase1_options` returns `None`; additional behavior checks and branch noise are inactive. |
| `models/hgr/policy.py` | Optional `standard_noise` input, shape/finite validation; default uses the same generator-based normal sampling. | B0/B1 use deterministic zero residuals, not HandoffPolicy sampling. |
| `models/hgr/stable_predictor.py` | Opt-in 153D float64 zero/ridge predictor; independent of 124D PRRAC critic. | Not constructed. |

The complete protected-baseline inventory also contains one prior addition
beyond those nine production differences:
`scripts/linux/run_hgr_phase1_acceptance.sh` (`89165a4`). It dispatches only the
manual acceptance module, rejects existing logs/nonempty output, preserves exit
status through `pipefail`, and is not invoked by SafeSearch. Its exact hash is
included; the scripts directory is never excluded.

`07a4b36` preserves six historical CRLF/mixed production files through explicit
`.gitattributes` entries. The new gate accepts complete reviewed raw-byte
inventories, rather than normalizing bytes at runtime: historical Git/LF,
existing Windows, and fresh Git with preserved production bytes and LF JSON
metadata. The third profile is required by the current attributes. Runtime
verification does not choose an allowed hash independently for each file.

## New repair scope

Three existing files expose default-equivalent construction/assembly hooks:

- `tools/ch3_baselines/basic_search_prior.py`: `_build_prior_controller` returns
  `SearchPriorController(phase, allocator=allocator)` exactly as before; the
  bridge factory is inherited from `MissionRuntime`.
- `chapter3_bser/experiments/hgr/runtime.py`: `_build_online_controller` returns
  `build_prrac_online_controller(phase_config, config)` and
  `_build_guidance_bridge` returns `RMADDPGGuidanceBridge()` with unchanged
  construction order, arguments and call count.
- `chapter3_bser/online/allocator.py`: `_generate_candidates` returns
  `generate_candidates(state, self.config)`; `allocate` uses that result once
  and preserves the existing observer and objective/solver logic.

Only the explicit SafeSearch subclass selects alternative components. No
shared-core source, reward/action/observation/dynamics contract or historical
manifest is modified. The allocator's historical mixed endings are retained;
the exact modified bytes receive a distinct new hash.

The exact new module/config/launcher names and before/after hashes are sealed
in `docs/provenance/safe_search_v1_evolution.json` after implementation review.
All `core` and Chapter 3 Python sources, baseline tools, all JSON config files,
all Python/shell scripts, original manifests, plan and this review are checked.
The 69 pre-existing additional inputs outside the former baseline inventory
were also compared with `44c86cd`: each matches its Git blob exactly or differs
only by the recorded CRLF-to-LF checkout conversion. Their complete hashes are
newly protected rather than treated as unrelated, exempt configuration.
Unexpected additions, deletions, byte changes or mixed source profiles fail.
Runtime preflight is read-only and never refreshes hashes. The candidate builder
returns an unaccepted in-memory record for review; it cannot bless arbitrary
changes to existing production sources.

All 27 Phase 0B-2 records remain present, with their historical hashes unchanged.
The original baseline entry continues to reject the evolved checkout. Test
results and bounded trajectory evidence are recorded separately; this review
does not claim improvement in Found or completion of a formal experiment.

## Follow-up after the first bounded probe

The first probe exposed two independent missing-assignment cases, reproduced
with synthetic tests before repair. The new `SearchController` now includes
unfinished searchers absent from the installed assignment in recovery pending;
an accepted replan clears only IDs that were actually installed. The opt-in
`SafeJointAllocator` preserves the original partial candidate, solver,
objective, standby and frozen-assignment rules, but merges over old IDs union
affected IDs when a searcher was previously absent. Ordinary partial replans
still call the unchanged inherited implementation. Both repairs remain inside
the independent SafeSearch namespace; the three legacy hooks are unchanged.
The first probe is incomplete evidence, not a completed experiment or a
performance comparison. Its source identity remains separate from the next
explicitly reviewed seal.

## Fixed-20 development evaluation and read-only coverage diagnostics

The subsequent 2026-09-28 user request explicitly authorizes the frozen D2
development split: B0, 20 original scenarios, all V0--V4 variants, and the
unchanged 400-step task horizon. It does not authorize training or execution
of the remaining 80 scenarios. The D1 source seal is retained verbatim in
`source_evolution_d1.json`; historical manifests and D1 evidence stay frozen.

This reviewed evolution adds three independent modules: `search_coverage.py`,
`run_development.py`, and `analyze_development.py`. The only existing SafeSearch
module changed is `run_paired.py`, to attach the optional coverage observer and
include its diagnostics in each trace and terminal record. The control,
guidance, geometry and recovery implementations, and all three legacy hooks,
retain their exact D1 bytes.

The coverage observer calls the original target negative-evidence update once
with unchanged arguments and returns its result unchanged. It reads actual
target-observation timestamps after that update, never hidden target or obstacle
truth. Optional per-searcher attribution uses the active planner's tensor
geometry and is published only when its union equals those timestamps. It
does not sample RNG, update beliefs twice, or supply data to any controller.
Coverage is a discrete evidence footprint, not a detection probability; effective
observation steps are explicitly labelled as a new/aged-evidence proxy.

The development launcher isolates scenarios in subprocesses, preserves original
indices and seeds, and requires 100 unique full task terminals with matching
source and input identities. A program error or missing arm prevents a complete
stage verdict. Parallel wall times cannot resolve the plan's serial timing tie
break. The offline analyzer distinguishes discovery, mission success, collision
roles, and timeout; paired uncertainty uses 20 scenario pairs, not 100 arms.
This remains development evidence and does not establish `performance_passed`.

## D2 observation-only repair after the interrupted first batch

The first full-episode batch exposed an observer IndexError in scenario index
10 / V3 after the last persisted transition at step 280. The original recorder
used a retained PathTracker index against the current guidance path, although
hold, route replacement, and completed-search guidance can legitimately retain
a different tracker route. This is a diagnostic program failure, not a task
collision or timeout. The first batch and its exact seal are preserved in
`source_evolution_d2_v1.json`; none of its arms are substituted into the rerun.

The new independent `telemetry.py` reads the existing state without resetting,
advancing, or modifying the tracker. It keeps the original remaining-path-length
measurement used by the motion-stall proxy. Segment geometry is reported only
when an active guidance route and its tracker route agree; otherwise it is null
with an explicit diagnostic reason. Indices are never clamped to invent a
segment. `run_paired.py` uses this observer in place of the historical recorder;
the historical script stays unchanged. All control, geometry, recovery, physics,
sensor and task implementations retain their exact D1 and first-D2 bytes.

The repaired recorder requires full-episode V3 observed/unobserved trajectory
equivalence on the failing scenario before a new complete 100-arm batch. Old
completed arms may verify overlapping trajectory signatures, but do not enter
the new batch's denominator. A change in worker count affects only independent
scene scheduling; no parallel wall-time comparison is a performance claim.
