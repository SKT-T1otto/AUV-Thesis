# D2 performance implementation review — 2026-10-08

Authorization: the user's Windows D2 Suite performance request permits
production performance repairs and new source identities while retaining every
algorithm, safety, environment, reward, scenario and training-budget contract.
Old Linux checkpoints and prepared plans are not migration targets.

## Reviewed boundaries

- Deterministic target forecasts reuse at most 16384 previously computed
  physical substeps. Every miss calls the original `advance_target_state`;
  actual target advancement, collision/reflection integration and prediction
  horizons are unchanged. Keys include exact position/velocity bytes, sample
  and reflection counts, dt, ordered obstacle geometry, bounds, clearance,
  reflection limit and epsilon. Values are immutable and returned arrays are
  copied. Metadata and layout identity do not enter physical integration and
  are preserved from the normalized caller state. Stateful/custom metadata,
  obstacle iterators, unusual numeric inputs and patched physical primitives
  retain the original loop. No predicted step is approximated or omitted.

- D2 installs `D2PlanningViews` in its existing assembly hook. The live safety
  projection copies current public occupancy and agent state without building
  a route graph. It intentionally has no `planning_graph`. The authoritative
  provider, periodic and forced refresh decisions, candidate generation and
  safety retries remain on the original path. Historical SafeSearch runtime
  construction retains its original full `PlanningViews` implementation.
- The projection key includes step, exact occupied-mask bytes, grid shape,
  spacing and center bytes, and current agent fields. It does not trust a map
  revision counter alone. Arrays use immutable buffers; no hidden target or
  ground-truth obstacle object is exposed.
- Shared online-map sampling batches the original points and floor/round/clip
  cell conversion. The same linspace endpoints, sample count, sample spacing,
  world bounds, per-ray deduplication, free/occupied votes, update order,
  saturation and map revisions remain. The sampled planner predicate and
  SafeSearch's closed-box predicate retain their distinct original meanings.
- Online connected components are cached by exact occupied/valid masks, cell
  coordinates, grid dimensions, world/z bounds, spacing, epsilon, clearance and
  neighbor offsets. Probability-dependent costs, A* results and endpoints are
  not cached by this geometry key. The cache has at most 16 entries, returns
  copies and applies only to the exact online planner class. No A* ordering,
  endpoint snapping, clearance or route feasibility rule changes.
- Repeated online sampled-segment queries use an LRU of at most 32768 answers.
  Every request keys the exact endpoints, freshly read occupied-mask bytes,
  grid dimensions, spacing, z origin, epsilon and world bounds. Probability
  revisions cannot invalidate this purely geometric predicate; occupied-cell
  changes invalidate it even without a revision increment. Endpoints are not
  rounded. Role and clearance were not inputs to the original online predicate.
- Original segment sample indices have a separate immutable 8192-entry cache,
  allowing occupancy changes to reuse sampling coordinates while checking the
  new occupied bytes. CPU physical edge/heuristic times have a 32768-entry cache
  keyed by exact Torch-computed displacement, dtype and role. The original Torch
  norm and scalar arithmetic run on each miss; CUDA planners retain the original
  method. Component-cache misses isolate the parent's revision-only memo so
  fresh content cannot accidentally read a stale inner cache.
- B2/B3 metrics and completed episode rows use compact append-only JSONL.
  Checkpoints retain full necessary episode metadata. Final saving materializes
  the original JSON arrays once through atomic temporary files. Update metrics
  no longer accumulate in memory. The reader ignores only an incomplete final
  line after interruption; corrupt complete records fail. Each append closes
  and flushes Python buffers; this does not claim power-loss durability.
- D2 HGR appends cycle/episode/branch journals and materializes final arrays at
  normal completion. Its checkpoint metadata is retained. Non-D2 HGR logging
  keeps its historical path. No checkpoint validation is removed.
- Compute settings are explicit `performance.learner_device` and
  `performance.cpu_threads`, defaulting to CPU and one thread. B2/B3 keep models
  resident on the chosen device for rollout and learning; the environment stays
  on CPU. Unavailable CUDA requests fail. HGR retains its CPU stochastic policy.
  Prepare-time CLI flags freeze choices into job configs and source-bound plans.
- The CUDA baseline adapter retains OU noise state, recurrence and random draws
  on CPU. The core agent's existing noise-to-action-device copy is reused; no
  core actor, critic, optimizer, RNG implementation or network is modified.
  CUDA arithmetic can still differ numerically and requires its own measured
  evaluation before any formal device choice.

## Unchanged experimental definition

No edits to reward, gamma, actor/critic networks, replay sampling or batch size,
warmup, update cadence, number of updates, 400000-step budget, 28D observation,
3D action, 124D critic, collision/team protocols, BSER candidates/objective,
V4 safety/recovery decisions, Found/handoff/mission behavior, physical dynamics,
random events or scenario distributions. No safety check, environment step or
learning update is skipped.

## Provenance and verification

The exact new evolution follows `d2_suite_pipeline_v1_evolution.json`. Historical
parent manifests and results stay frozen. The two shared core evolutions are
separately pinned against the Phase 0B-2 `core/mapping/path_planner.py` and
`core/env/target_motion.py` hashes;
all original 27 records remain validated. No directory exemption, ignored
mismatch, permissive normalization or checkpoint compatibility bypass is added.

Validation covers scalar-reference mapping, cache invalidation, interrupted
journals, runtime/source guards, fixed-seed reward/trajectory/model signatures,
relevant regression and frozen E0 comparisons. Opt-in instrumentation separates
rollout, environment, snapshots, extraction, graph/endpoint construction,
controller, geometry, route/A*, guidance, replay, learner, logging and checkpoint
validation. Timers are never imported by formal training. Measured results and
limitations are recorded separately; this review is not thesis performance
acceptance or an assertion that a throughput threshold has been met.

## Operational boundary

Retain `linux_pipeline_01` with its original source/checkpoint identity. Prepare
`linux_pipeline_02` from the same frozen scene inputs and train its formal seeds
from scratch after human review. No formal experiment, automatic resume, commit
or push belongs to this maintenance task. Serial launching remains the default;
single-seed tests do not establish multi-seed concurrency on the Linux 3090 host.
