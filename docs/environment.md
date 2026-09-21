# Runtime environment

## OpenMP compatibility: explicit opt-in only

Some conda environments combining Intel MKL, NumPy and PyTorch may load duplicate
Intel OpenMP runtimes (`libiomp5md.dll` / `libiomp5.so`) and stop with `OMP: Error #15`.
The updated launchers **never set `KMP_DUPLICATE_LIB_OK`**. Unset, empty and existing
user values are preserved. The launchers themselves do not enable the bypass by
default; the retained legacy Python exception is documented below.

A user may explicitly supply `KMP_DUPLICATE_LIB_OK=TRUE` as a temporary workaround.
Bypassing the duplicate-runtime check is unsafe: it may crash or silently produce
incorrect results. **Numerical reliability is not guaranteed.** This patch changes
launcher reporting only; unchanged algorithm/model/seed configuration does not
establish numerical equivalence or validate scientific results obtained with the
bypass enabled.

Every updated launcher prints `OPENMP_RUNTIME_ENV` and the inherited environment
value to stderr before Python starts; unset is recorded as `<unset>`. A nonempty
value other than `FALSE`, `0`, `NO` or `OFF` also emits a conservative risk warning.
The launcher does not reinterpret or rewrite user values. Keep stderr in the run
log so the setting and warning remain attached to the invocation. This is a
launcher-time record, not a guarantee that a later conda activation, imported
module or user code will leave the environment unchanged.

Normal Windows PowerShell usage (no bypass enabled by these commands):

```powershell
python -B -m unittest tests.test_openmp_runtime_env -v
python -B -m unittest tests.test_ch3_baseline_validation -v
.\scripts\run_ch3_basic_prior_eval.bat --help
```

Explicit, temporary Windows opt-in in a separate PowerShell session:

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
.\scripts\run_ch3_basic_prior_eval.bat --help 2>&1 | Tee-Object openmp-launch.log
Remove-Item Env:KMP_DUPLICATE_LIB_OK
```

Normal Linux usage:

```bash
python -B -m unittest tests.test_openmp_runtime_env -v
python -B -m unittest tests.test_ch3_baseline_validation -v
bash scripts/linux/run_ch3_basic_prior_eval.sh --help
```

Explicit Linux opt-in for one command only:

```bash
KMP_DUPLICATE_LIB_OK=TRUE bash scripts/linux/run_ch3_basic_prior_eval.sh --help >openmp-launch.log 2>&1
```

The help examples do not start experiments. Existing inherited shell values are
respected; clear the variable yourself when you intend to run without it.

## Updated launchers

The reporting block follows `setlocal` in batch, `$ErrorActionPreference` in
PowerShell and the initial `set` flags in Bash, before Python executes. The sourced
`_search_value_audit_common.sh` helper does not itself launch Python; both D1/D2
entrypoints report the environment before sourcing it. Python commands, argument
order and exit-code handling are retained.

Windows:

- `scripts/run_ch3_basic_prior_eval.bat`
- `scripts/run_ch3_baseline_eval.bat`
- `scripts/train_ch3_direct_mc.bat`
- `scripts/train_ch3_direct_boundary.bat`
- `scripts/run_ch3_learning.ps1`

Linux (all 20 Python launchers in `scripts/linux/`):

- `bundle_search_value_audits.sh`
- `env_preflight.sh`
- `run_ch3_baseline_eval.sh`
- `run_ch3_basic_prior_eval.sh`
- `run_ch3_learning.sh`
- `run_collision_terminal.sh`
- `run_phase1c_prrac_eval.sh`
- `run_phase1c_prrac_execution_ablation.sh`
- `run_phase1c_prrac_s1_search_diag.sh`
- `run_phase1c_prrac_s1_train.sh`
- `run_phase1c_prrac_s2a_collision_ablation.sh`
- `run_phase1c_prrac_s2a1_local_connector_ablation.sh`
- `run_phase1c_prrac_train.sh`
- `run_phase1c_v2_1_train.sh`
- `run_phase1c_v2_diagnostic_eval.sh`
- `run_phase1c_v2_train.sh`
- `run_search_value_d1_audit.sh`
- `run_search_value_d2_audit.sh`
- `train_ch3_direct_boundary.sh`
- `train_ch3_direct_mc.sh`

## Exact retained historical assignment

`historical_openmp_assignment_retained = true`

The machine-readable record is [openmp_runtime_compatibility.json](openmp_runtime_compatibility.json).
The sole exception is `core/runtime/engine.py:18`:

```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

The entire file must remain byte-identical to commit
`a51a234e4ca9575c9707409c6b75a26380ad2a5c` and its existing entry in
`docs/chapter3/baselines/frozen_production_source.json`:

```text
1998051dc1651ea9413a4e18124d40dce0e56cc58aeb5e345c4a552a46e698cc
```

Tests verify the full file hash, frozen Git bytes, exact line and assignment AST.
No entire directory is exempted. The full 196-file production inventory is checked,
and Python write checks cover tracked/untracked repository sources (including
tests), plus all Python files in `core`, `chapter3_bser`, `tools` and `scripts`.
Added assignments, mutator calls and any change to the historical file fail.
Mutation cases verify these failures without editing production files. The AST
regression guard covers literal and named constant keys and common environment
mutators; it does not claim to detect every possible computed Python program.

This repository still contains a Python environment assignment. Importing the
historical engine executes it and can overwrite a user value, including `FALSE`.
The exception is retained by explicit user instruction; it is not a project-wide
claim that Python no longer modifies OpenMP settings.

## Entry import inspection

Fresh-process tests inspect each entry with the variable unset, `FALSE` and `TRUE`.
They observe the environment before/after import, `sys.modules`, and import audit
stack frames for the historical engine. Imports construct no trainer, restore no
checkpoint and run no evaluation. The legacy `core.runtime.training` import is a
positive control to prove that the trace detects the actual engine path and its
overwrite. The verified local import results are:

| Entry | Initial environment values | Old engine imported | Environment after import |
| --- | --- | --- | --- |
| `tools.ch3_baselines.evaluate` (B0) | unset / FALSE / TRUE | no | unchanged |
| `tools.ch3_baselines.run_baseline` (B0/B1) | unset / FALSE / TRUE | no | unchanged |
| `chapter3_bser.experiments.hgr.cli` (HGR) | unset / FALSE / TRUE | no | unchanged |
| `core.runtime.training` (legacy positive control) | FALSE | yes | TRUE |

The observed legacy path is `core/runtime/__init__.py:3` importing `.builder`,
then `core/runtime/builder.py:11` importing `core.runtime.engine`, which executes
its retained assignment at line 18. Import traces and before/after values are in
the JSON report. The scope is entry imports in the local AUV environment, not a
claim about every possible later dynamic import or checkpoint execution path.

## Verification and provenance

`tests/test_openmp_runtime_env.py` checks default non-enablement, unchanged explicit
values, stderr records and risk warnings, Bash syntax, argument forwarding and
exit codes using a fake Python executable. It also checks source provenance,
negative mutation cases, and independent process imports. Probe environment
values are assigned only by child shells, never by new Python assignments.

The baseline source gate now protects Linux execution inputs only; see
[Linux baseline provenance](chapter3/baselines/linux_provenance.md) for the current
19-file inventory and verification. Windows launcher bytes are excluded from
every active source gate. Their original hashes remain historical metadata, and
their commands/functionality are checked separately. The OpenMP evolution check
retains exact Linux bytes; Windows command comparisons allow CRLF/LF checkout
conversion. Historical production manifests, their hashes and checkpoint
identities are unchanged. No algorithm, model, baseline or HGR logic is changed.

The earlier 2026-09-20 results (16/16 baseline, 7/8 OpenMP, 10/10 provenance/metadata)
belong to the superseded default-on patch and old acceptance scope. They are not
the verification result for this explicit-opt-in revision.

## Earlier local verification: OpenMP opt-in revision (2026-09-21)

The later Linux provenance revision and its 38-test rerun are recorded in the
[Linux provenance report](chapter3/baselines/linux_provenance.md).

| Test module | Result |
| --- | --- |
| `tests.test_openmp_runtime_env` | 12/12 PASS, no skips, 87.231 seconds |
| `tests.test_ch3_baseline_validation` | 16/16 PASS, 739.154 seconds |
| `tests.test_ch3_baseline_provenance` + `tests.test_repository_metadata` | 10/10 PASS, 1.511 seconds |

The baseline suite was launched with the OpenMP variable unset. It ran only its
two temporary synthetic validation episodes: B0 ended successfully at step 161;
B1 reached the normal 400-step timeout, with no program failure. These outcomes
validate the bounded test fixtures and output contracts, not formal performance.

The 196 production Python files retain their frozen hashes. No historical
production hash or checkpoint identity was rewritten. No formal training or
100-episode evaluation ran. These are Windows local results; Bash probes used
Git Bash, not a native Linux/PyTorch installation. Passing these checks does not
certify numerical reliability when the explicit bypass is enabled.
