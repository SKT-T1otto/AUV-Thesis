# CH3-final experiment provenance

Reviewed starting point: `7b3767ca58b547bb34d0105ecdc9b8189ed151e7`, 2026-09-21.
This revision changes provenance, metadata, tests and documentation only. It does
not start training, resume a checkpoint, establish performance, or authorize a
100-episode evaluation. The new experiment chain is **B0 → B1 → B2 → B3 → HGR**.

## Current production identity

The new `production_source_sha256` is:

```text
3bf6035001e28efbe5e5cd3db7b43bc43a11e0bf83ed2037a7ac29c5c518b4bd
```

It identifies all **196 Python files** in `core/` and `chapter3_bser/` as stored
in the current Git/LF source tree. The complete inventory is pinned in
[`final_production_source.json`](../chapter3/baselines/final_production_source.json).
Production file-set additions, removals and unreviewed byte changes fail. The
baseline gate never automatically refreshes a manifest or consults a checkpoint.

The inspected Windows working tree still has the raw aggregate
`7a8dda612fb68c0a63bcc38b04ee0973ac80488bddd6c9fffe1fbf9b2f23f491`.
Comparison against every Git blob established that the entire difference is
CRLF/mixed versus LF line endings in these six files:

- `chapter3_bser/controllers/action_adapter.py`
- `chapter3_bser/experiments/phase1b1_pilot/run_pilot.py`
- `chapter3_bser/hysteresis/policy.py`
- `chapter3_bser/online/allocator.py`
- `chapter3_bser/online/config.py`
- `chapter3_bser/online/controller.py`

Those working files are **unchanged**, including their bytes. The new pin records
two exact, complete reviewed inventories: `git_lf` and `windows_existing`.
Runtime validation hashes actual bytes and requires equality with one entire
inventory. It does not normalize newlines, accept arbitrary CRLF changes, combine
per-file alternatives, skip a directory, or disable hash checks. An unreviewed
newline change in another production file also fails.

The returned `production_source_sha256` names the portable final experiment.
`production_checkout_profile` records the matched checkout, while `production`
retains HGR's native **raw-byte** identity. On a Git/LF Linux checkout both hashes
are `3bf603…b4bd`; in the retained Windows worktree the native hash is `7a8dda…f491`.
This distinction preserves the existing checkpoint comparisons without modifying
HGR. It does not make Windows raw-byte checkpoints portable to a different LF
source inventory. Final Linux B2/B3/HGR checkpoints must be created under the
new LF source version and keep their original identities.

## Historical HGR checkpoint provenance

Historical HGR checkpoint provenance
`7a8dda612fb68c0a63bcc38b04ee0973ac80488bddd6c9fffe1fbf9b2f23f491`
is **ended** for the new experimental chain. The user reports the checkpoint is
no longer available; it is not a resume target or an evaluation prerequisite.
The hash's occurrence in the retained Windows source profile describes observed
source bytes, not authorization to resume that historical run.

The old `frozen_production_source.json` and `framework_protected_b0.json` remain
byte-for-byte intact as historical records. The previous active baseline manifest
is preserved byte-for-byte as
[`ch3_baseline_protected_pre_final.json`](ch3_baseline_protected_pre_final.json).
The Phase 0B-2 migration manifest, all 27 records, collision/HGR evolution records
and checkpoint identities are not rewritten. Active B0 and unified baseline
entries now use the new final gate; the old files are not runtime prerequisites.

## Baseline protected inventory

[`framework_protected_baseline.json`](../chapter3/baselines/framework_protected_baseline.json)
protects **36 files**: nine baseline Python modules, six baseline JSON files,
and every one of the 21 shell scripts under `scripts/`. Discovery is recursive;
new shell scripts are included regardless of their command text. This covers the
HGR Linux entry as well as the four baseline entries. All baseline and shell
files are hashed as exact bytes. Windows `.bat`, `.cmd`, `.ps1` and `.psm1` files
are excluded from the runtime inventory and remain available for execution.

```text
configs/chapter3/baselines/baseline_registry.json
configs/chapter3/baselines/basic_search_prior_v1.json
configs/chapter3/baselines/bser_prior_eval.json
configs/chapter3/baselines/direct_boundary_train.json
configs/chapter3/baselines/direct_mc_train.json
configs/chapter3/baselines/search_prior_eval.json
scripts/linux/_search_value_audit_common.sh
scripts/linux/bundle_search_value_audits.sh
scripts/linux/env_preflight.sh
scripts/linux/run_ch3_baseline_eval.sh
scripts/linux/run_ch3_basic_prior_eval.sh
scripts/linux/run_ch3_learning.sh
scripts/linux/run_collision_terminal.sh
scripts/linux/run_phase1c_prrac_eval.sh
scripts/linux/run_phase1c_prrac_execution_ablation.sh
scripts/linux/run_phase1c_prrac_s1_search_diag.sh
scripts/linux/run_phase1c_prrac_s1_train.sh
scripts/linux/run_phase1c_prrac_s2a1_local_connector_ablation.sh
scripts/linux/run_phase1c_prrac_s2a_collision_ablation.sh
scripts/linux/run_phase1c_prrac_train.sh
scripts/linux/run_phase1c_v2_1_train.sh
scripts/linux/run_phase1c_v2_diagnostic_eval.sh
scripts/linux/run_phase1c_v2_train.sh
scripts/linux/run_search_value_d1_audit.sh
scripts/linux/run_search_value_d2_audit.sh
scripts/linux/train_ch3_direct_boundary.sh
scripts/linux/train_ch3_direct_mc.sh
tools/ch3_baselines/__init__.py
tools/ch3_baselines/basic_search_prior.py
tools/ch3_baselines/bser_prior.py
tools/ch3_baselines/evaluate.py
tools/ch3_baselines/framework_provenance.py
tools/ch3_baselines/provenance.py
tools/ch3_baselines/registry.py
tools/ch3_baselines/run_baseline.py
tools/ch3_baselines/run_training.py
```

Documentation, README and Markdown bytes do not participate in identity. The
two new/current JSON pins are read as the authority for source inventories, but
their file bytes are not hashed into execution identity. Changing editorial
metadata or JSON formatting does not change identity; changing their pinned
inventories still requires a consistent digest and must match actual sources.

## OpenMP exception retained

`historical_openmp_assignment_retained = true`

The original assignment at `core/runtime/engine.py:18` remains exact, with file
SHA256 `1998051dc1651ea9413a4e18124d40dce0e56cc58aeb5e345c4a552a46e698cc`.
The existing regression guard checks it against its historical frozen commit,
and scans all other Python sources for new assignments. This project still
contains that historical Python environment assignment.

Launchers keep explicit user opt-in, stderr environment-value records and risk
warnings. They do not default the OpenMP bypass to TRUE. The bypass can crash or
silently yield incorrect results; numerical reliability is not guaranteed.
The B0/B1 and independent HGR entries do not import `core.runtime.engine` in the
existing fresh-process probes. The legacy path
`core.runtime.training → core.runtime.__init__ → core.runtime.builder → core.runtime.engine`
still overwrites the variable at line 18. No frozen core fix is included here.

## Linux update and bounded smoke commands

Run from the Linux repository root after this revision has been committed and
pushed to `origin/main`. This task does not itself commit or push. Activate the
existing Python experiment environment first. Use a checkout whose local work
has been retained and reviewed; the following pull is fast-forward only:

```bash
git switch main
git pull --ff-only origin main
```

If intentionally aligning a reviewed local branch directly to the remote tip,
the reset alternative preserves local edits or stops on a conflict:

```bash
git fetch origin
git reset --keep origin/main
```

Invoke the gate, rather than only importing its function:

```bash
python -B -c "from tools.ch3_baselines.framework_provenance import framework_sources; s=framework_sources(); print(s['production_source_sha256']); print(s['production']['sha256']); print('PASS')"
python -B -m unittest -v -f tests.test_ch3_baseline_validation
```

The checked-in `tests/fixtures/ch3_final/smoke_manifest.json` is the existing
synthetic handoff fixture with validation labels and the unchanged 400-step task
horizon, exactly matching the validation suite. It is interface verification,
not formal performance data. Each invocation below runs **one** episode with no
checkpoint and no training; a fresh temporary directory prevents overwriting
retained experiment outputs:

```bash
smoke_root="$(mktemp -d -t ch3-final-smoke.XXXXXX)"
bash scripts/linux/run_ch3_baseline_eval.sh \
  --baseline B0_search_prior \
  --manifest tests/fixtures/ch3_final/smoke_manifest.json \
  --episodes 1 --seed 12729 \
  --output-dir "$smoke_root/collision_terminal/B0"
bash scripts/linux/run_ch3_baseline_eval.sh \
  --baseline B1_bser_prior \
  --manifest tests/fixtures/ch3_final/smoke_manifest.json \
  --episodes 1 --seed 12729 \
  --output-dir "$smoke_root/collision_terminal/B1"
```

B2/B3/HGR training and later evaluation need new manually authorized runs and
their own resulting checkpoints. There is no dependency on the lost HGR file.

## Verification

Required local command:

```powershell
python -B -m unittest -v -f tests.test_openmp_runtime_env tests.test_ch3_baseline_provenance tests.test_repository_metadata tests.test_ch3_baseline_validation
```

The provenance suite includes `test_linux_provenance_platform_independent`: it
builds a temporary LF source checkout, simulates CRLF/LF Windows launchers,
invokes the entire gate in a fresh Python process without historical manifests
or checkpoints, and requires the final Linux production hash. Separate tests
change, add and remove real fixture files in every protected directory. Native
Windows BAT help and PowerShell dispatch tests remain in place.

Latest recorded local verification, 2026-09-21: **42/42 PASS**, no skips, in
**1016.916 seconds**, using `D:\anaconda\anaconda\envs\AUV\python.exe` on Windows.
The OpenMP bypass variable was unset when the suite started. This is local
verification, not CI or a native Linux/PyTorch run.

| Suite | Passed |
| --- | --- |
| `tests.test_openmp_runtime_env` | 12/12 |
| `tests.test_ch3_baseline_provenance` | 8/8 |
| `tests.test_repository_metadata` | 6/6 |
| `tests.test_ch3_baseline_validation` | 16/16 |

B0's single synthetic episode completed successfully at step 161 (318.41 s).
B1 completed with a normal 400-step timeout (514.06 s), `program_failure=false`.
Both had complete output schemas and passed zero-residual/no-learning assertions.
Neither result is a formal performance claim. No training or 100-episode
evaluation ran, and all simulator outputs were temporary test artifacts.

All 196 production file bytes match the pre-change snapshot. Historical records
remain unchanged, including the exact legacy engine assignment. All 36 current
baseline-protected files have LF bytes, and their aggregate is
`ea3777d6e51e0aa002f12669e4d431d3d8ffed0105511c14051fadebe2023d41`.
The archived pre-final baseline manifest matches its original Git blob exactly.

Changed implementation files are only `tools/ch3_baselines/provenance.py` and
`tools/ch3_baselines/framework_provenance.py`. Other changes are the three scoped
test modules, the new synthetic smoke JSON fixture, the new/current provenance
JSON records, this report, and links identifying the older documentation as
historical. No launcher, baseline algorithm/configuration, HGR, model, reward,
collision or environment implementation was changed.
