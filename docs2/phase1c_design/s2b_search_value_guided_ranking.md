# S2B Phase2: Search Value Guided BSER Candidate Ranking

This is an opt-in, inference-only experiment, not a completed performance result.
No new training, reward, actor/critic architecture, execution control or S2A1
recovery behavior is introduced. The 28D / 3D / 124D contracts and existing
checkpoint schemas are unchanged.

## Integration

`candidate_generator.py` supplies the unchanged reachable search/standby pools.
`BSEROnlineAllocator._solve_candidates` defaults to the original joint greedy
solver. The PRRAC factory injects `SearchValueGuidedBSERAllocator` only when
`search_value_guidance.enabled` is true and `weight > 0`.

The guided allocator first runs original joint BSER and retains its executor
standby choice. For that fixed standby it reranks positive-marginal-gain search
candidates in each greedy round:

```text
final_score = original_bser_marginal_gain + weight * clipped_probability
```

No score normalization or replacement of the BSER objective occurs. The default
weight is 0.1 and supported weights are [0, 0.1]; probability clips must lie in
[0, 1]. Thus a per-round original score advantage greater than 0.1 cannot be
overturned. This is a bounded auxiliary term, not a claim that its relative
contribution is always small when original marginal gains approach zero.
Zero original gain cannot become eligible via the bonus. Partition constraints,
deterministic candidate-key ties and original objective reporting are retained.
Existing controller event scheduling, acceptance and hysteresis still decide
whether the resulting allocation is used. Both full and partial allocations use
the hook; unaffected assignments remain frozen by the existing partial allocator.
After target discovery the auxiliary ranking is bypassed.

## Head inputs and compatibility

The evaluator sends the checkpoint's Head weights to workers as NumPy snapshots.
Workers create a separate CPU, eval-mode, gradient-disabled copy without changing
the actor RNG stream. There are no updates to the Head or checkpoint.

The trained Head expects 34D input: a copy of the existing 28D observation plus
the existing six public search statistics. To distinguish candidates, an isolated
instance of the existing `PathTracker` previews the initial tracking waypoint of
each candidate's existing path. Only navigation delta, direction, distance and
closing speed are replaced in the Head's private feature copy (indices 6:12, 15,
17). Actor observations and the live tracker are never changed by scoring. Current
coverage, collision history and belief statistics are preserved; no hypothetical
future observations or hidden target/obstacle information are synthesized.

This is a counterfactual local-guidance surrogate. A Head trained on observed
states is not automatically calibrated for every alternative candidate. Candidates
with identical local navigation previews can legitimately receive equal values.
High classification accuracy alone does not establish better candidate ranking or
Found Rate; a paired pilot is required.

Old configs omit the option and follow the original solver, without Head inference
or new episode fields. Explicit `enabled=false` and `weight=0` also bypass inference.
Legacy PRRAC checkpoints without a Head continue to load with guidance disabled.
Active guidance fails clearly if Head weights are absent; it never ranks using an
untrained random fallback. Present weights must match the existing Head exactly.
This experiment does not activate the separate low-value re-evaluation controller.

## Diagnostics

Explicit guidance configs write `search_value_guidance_metrics.json` and store
episode statistics in `episode_evaluation.csv`. The new JSON includes per-episode
checkpoint/scenario/mode identifiers and weighted aggregate statistics:

- `candidate_count`: candidates evaluated once per allocation proposal.
- `mean_search_value`: candidate-weighted mean prediction.
- `mean_selected_candidate_value`: mean value of solver-selected candidates.
- `selected_value_rank`: mean one-based value rank within each selected searcher's
  candidate pool (ties share rank).
- `ranking_changed_count`: greedy rounds with ordering different from original
  marginal-gain order at the same selected prefix.
- `allocation_changed_count`: changed search proposals versus original joint BSER.
- `accepted_search_change_count`: changed proposals actually returned by the
  controller; rejected proposals do not count as installed changes.

The first two change counters describe proposals, not proof of execution changes.
Check the accepted counter and Found Rate together. Existing resolved-config hashes
include the option, preventing a resume with different ranking parameters; old
configs are not augmented with defaults, preserving their resume hashes.

## Latest recorded local verification (2026-09-04; not CI)

Runtime: the existing AUV Python environment on Windows. Test checkpoints and
evaluation artifacts were generated only in temporary test directories.

- `python -m compileall -q chapter3_bser scripts tests`: passed.
- `python -m unittest tests.test_search_value_head`: 9 passed.
- `python -m unittest tests.test_search_value_guided_ranking`: final 17 passed
  (127.263 seconds), including acceptance-flag accounting.
- Existing 28D/provenance tests and native B1 train/evaluation equivalence passed.
- `git diff --check`: passed.
- `python -m unittest discover -s tests`: first run completed 335 tests in
  2453.340 seconds, with 2 failures and 4 errors. Two integration compatibility
  issues were then confirmed fixed: preserve the original disabled controller
  constructor call and keep the optional guidance artifact outside the legacy
  mandatory artifact list. All six failing cases were rerun on the corrected
  code: those two passed; the following four historical checks still failed.

Remaining failures (neither their tests nor historical evidence were modified):

1. `tests.test_bser_event_detection`: the test expects 8 events, whereas the
   unchanged `HEAD` event enum contains 9.
2. `tests.test_bser_v1_artifacts_frozen`: `HEAD` lacks
   `experiments/chapter3/bser_e1_offline/aggregate_by_profile.csv`.
3. `tests.test_ch3_e0_equivalence.E0DeliveryTests.test_full_e0_passed`: missing
   `experiments/chapter3/e0_equivalence/equivalence_summary.json`.
4. `tests.test_phase1c_v2_isolation.Phase1CV2IsolationTests.test_overlay_manifest_contains_no_frozen_or_core_files`:
   missing `docs2/phase1c_v2_design/overlay_manifest.json`.

The entire discovery suite was not repeated after the targeted fixes; this is
not a claim of a clean full-suite pass. The final six-case rerun had one failure
and three errors, all listed above. No frozen checks were weakened or skipped.
No existing checkpoints, `outputs`, S2A1 results, commits or pushes were changed.
Resolve/review these regression blockers before treating the experiment gate as
passed. No pilot, formal training or performance comparison was launched.

## Next pilot (not executed; regression gate not yet passed)

Use `configs/chapter3/bser_phase1c_search_value_guided.json` with an explicitly
chosen, already-trained native B1 checkpoint containing the Head. It is an
evaluation config (`training_update=false`), with 10 validation scenarios, 400
steps and unchanged S2A1 C2 local recovery. From the repository root:

```powershell
python -m chapter3_bser.experiments.phase1c_prrac.evaluate_prrac_checkpoints `
  --config configs/chapter3/bser_phase1c_search_value_guided.json `
  --checkpoint '<existing-trained-search-value-checkpoint.pt>' `
  --episodes 10 --workers 1 --device cpu `
  --output-dir '<new-empty-guided-pilot-directory>'
```

For the paired baseline, copy this config to a new local config and change only
`search_value_guidance.enabled` to false. Use the same checkpoint, seed, scenarios,
mode and recovery variant, and a separate new output directory. Do not reuse or
overwrite historical S2A1 outputs. Pilot authorization and checkpoint choice are
still required; these commands have not been run.
