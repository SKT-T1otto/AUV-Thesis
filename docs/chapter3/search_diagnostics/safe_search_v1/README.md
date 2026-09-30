# Safe-search v1: B0/B1 paired experiments

For the fixed-20 B0 development launcher, coverage diagnostics, and the current
execution record, see [development_runbook.md](development_runbook.md). The D1
implementation report and its evidence remain the historical bounded checks.

The 2026-09-28 local fixed-20 evaluation is complete: 100 terminal episodes,
27,790 physical steps, and zero program failures. See the
[full results and diagnosis](development_results_20260928/report.md) and
[compact evidence](development_results_20260928/compact_evidence.json).
The registered engineering gate selects V3, with Found 35% and pre-Found
collision 25% versus V0's 25% and 45%; substantial stall remains. V1/V4 reach
40% Found, while V2 fails the collision gate. These are development findings,
not statistical superiority or formal performance acceptance.

This entry evaluates an opt-in Chapter 3 runtime. It loads no checkpoint, performs
no learning, and preserves the original 400-step collision-terminal task. It
does not require `3090结果` or any historical HGR run directory. Supply the full
validation manifest already prepared on the experiment machine.

This implementation is restricted to B0/B1's deterministic zero-residual prior
controllers, so neither method needs retraining for this comparison. B2, B3
and HGR are not wired into this new entry. Their old training results or
checkpoints cannot be used as results of this repair. A later integration and
fixed-weight evaluation are needed before deciding whether to retrain those
learned methods.

The five variants are V0 (disabled), V1 (planning-state consistency), V2
(consistency plus failure recovery), V3 (path safety), and V4 (all interventions).
V0 is compared with a direct original B0/B1 runtime when `--verify-v0` is supplied.
The comparison covers each step's physical state, observations, rewards,
guidance and Python/NumPy/Torch CPU RNG state, including initialization.

## Linux: bounded mechanism regression first

From the new repository checkout and the existing AUV Python environment:

The pasted Linux transcript confirms the checkout directory below, but does
not contain the manifest's filename or location. Set `MANIFEST` to your existing
full validation manifest before running this block. The configuration path
below is the checked-in runtime reference, not an old HGR result directory.
Keep an externally prepared manifest outside the repository or under `runs/`.
Adding a new JSON file under the protected `configs/` tree correctly fails the
source inventory check; do not edit expected hashes to accommodate it.

```bash
cd /home/legion/AUV-Thesis/AUV-Thesis-SafeSearch
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate AUV
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

# MANIFEST is the existing scene file prepared on this Linux machine.
: "${MANIFEST:?Set MANIFEST to your existing full validation manifest}"
CONFIG=/home/legion/AUV-Thesis/AUV-Thesis-SafeSearch/configs/chapter3/hgr_train.json
RUN_ID=$(date +%Y%m%d_%H%M%S)

python -B -m chapter3_bser.experiments.safe_search_v1.run_paired \
  --manifest "$MANIFEST" \
  --config "$CONFIG" \
  --baseline B0_search_prior \
  --variants V0,V1,V2,V3,V4 \
  --episode-indices 0,3,28 \
  --steps 100 --seed 12729 --verify-v0 \
  --output-dir "runs/safe_search_v1/mechanism_B0_${RUN_ID}"
```

The indices `0,3,28` identify scenarios 1, 4 and 29 only when this is the same
ordered manifest used in the diagnosis. Check `identity.json`: it records the
scenario id, scene hash, scenario seed, original index and innovation seed.
Different manifests remain valid experiments, but cannot be labeled replay of
those original failure cases.

The scenario seed is retained. The environment innovation seed is exactly
`12729 + original_manifest_index`, never the index within a selected subset.
The command checks the full manifest, including unselected scenarios, for
validation labels, unique identities, profile and task horizon.

## Full-episode evaluation after mechanism checks

The fixed 20 development indices below come from
`docs/chapter3/search_diagnostics/3090_20260928/experiment_plan.json`. They are
original zero-based indices, selected before this repair's evaluation; the
other 80 remain reserved for the locked comparison. These indices apply only
to that plan's original 100-scene order. A differently ordered manifest must
not be silently assigned the same development/held-out labels.

```bash
python -B -m chapter3_bser.experiments.safe_search_v1.run_development \
  --manifest "$MANIFEST" \
  --config "$CONFIG" \
  --workers 4 --execute \
  --output-dir "runs/safe_search_v1/development_B0_${RUN_ID}"
```

This launcher checks the exact frozen scenario content, runs all five variants
with the original seeds and 400-step horizon, and requires all 100 terminal
records. Its per-scene subdirectories contain the paired-run outputs below.
Coverage observation is enabled automatically. Do not interpret partial counts
as the completed development success rate.

Only after locking one variant on development scenes, compare `V0` against
that variant on the reserved original indices. Repeat the locked comparison
with `--baseline B1_bser_prior`. The prior validation set is not a newly
independent thesis test set. `--full-episodes` explicitly requests full task
evaluation; it never starts training.

## A+C development addition (V5)

The user-requested V5=A+C addition completed all 20 fixed development scenes
on 2026-09-28: 6,315 physical steps and zero program failures. Found was 8/20
and pre-Found collision was 5/20. The separate A+C development gate passed;
the original V0–V4 registered selection remains V3. See the
[A+C report](ac_results_20260928/report.md) and
[reproduction runbook](ac_runbook.md). This is a post-hoc development
comparison, not independent test evidence or formal performance acceptance.

## B1 paired transfer (V0/V3/V5)

The B1 transfer evaluation completed all 60 terminal runs on the same 20
development scenes on 2026-09-29: 17,216 physical steps and zero program
failures. Found was 5/20, 5/20, and 7/20; pre-Found collision was 9/20,
4/20, and 10/20 for V0, V3, and V5 respectively. V5 reduced movement
stalls but failed the preregistered safety gate against V3.

See the [B1 result and diagnosis](b1_results_20260928/report.md),
[frozen B1 plan](b1_experiment_plan.json), and
[CLI / platform prerequisites](b1_runbook.md). Long Hold drift was reconciled
against recorded dynamics, and post-Found start-connector failures were
audited separately. No training, HGR run, or remaining-80 evaluation was run.

## Outputs and interpretation

- `identity.json`: source gate, unchanged input hashes, versions, complete task
  conditions, source baseline and selected original indices/seeds.
- `resolved_config.json`, `evaluation_manifest.json`: exact resolved native
  configuration and complete original manifest for reproducibility.
- `episode_NNNN/VN/step_trace.jsonl`: public positions, velocities, installed
  guidance, actual query failure reasons and motion-stall proxies. No truth
  obstacle geometry is supplied to online control by this observer.
- `episode_NNNN/VN/summary.json`: terminal result or explicit `budget_cutoff`,
  controller diagnostics and optional V0 exact-equality result.
- `episodes.json`, `summary.json`: per-arm counts and V0 paired differences.
  `failure.json` is written if an experiment stops on an error.

A nonterminal 100-step prefix is **budget cutoff**, not timeout. Its
`found_within_budget_fraction` is descriptive; `found_rate` and full-evaluation
collision rate remain null. Full Found rates appear only after every requested
scenario of that arm terminates in a `--steps 400 --full-episodes` run. Partial
runs never silently become complete evaluation populations.

Motion-stall and hold fractions use three searchers' pre-found agent-steps as
the denominator. Increased hold may explain a lower collision count; a safer
method that remains parked is not demonstrated to improve search. Optional
`--search-coverage` records the actual target-evidence footprint and counts
steps with new or sufficiently aged observations. This is an evidence-renewal
proxy, not detection probability; absent or invalid diagnostics remain null.
Query failure counts are not
map unreachability fractions. Formal performance acceptance stays unset; the
paired summary does not automatically declare a winning variant.

The runner accepts only a new/empty output directory, rejects retained
`outputs` and imported result directories, and checks immutable source and
input identities before and after each runtime. Never bypass a failed source
gate by editing its expected hashes on the experiment machine.
