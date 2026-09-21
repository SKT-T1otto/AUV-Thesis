# Linux baseline provenance

The baseline source gate uses the same Linux execution scope on every host.
Windows launchers remain available and are checked for functionality separately;
their checkout bytes do not participate in baseline runtime identity.

## Protected inventory

`framework_protected_baseline.json` currently protects 19 files:

- All nine Python files under `tools/ch3_baselines/`.
- All six JSON configurations under `configs/chapter3/baselines/`.
- Four Bash entrypoints under `scripts/linux/`: `run_ch3_basic_prior_eval.sh`,
  `run_ch3_baseline_eval.sh`, `train_ch3_direct_mc.sh`, and
  `train_ch3_direct_boundary.sh`.

The scanner recursively discovers `.sh` files under `scripts/` referencing the
baseline namespace, including new entrypoints. Additions, removals, content
changes and removal of the namespace reference invalidate the reviewed inventory.
Unrelated experiment scripts remain outside this baseline-specific scope.

Windows `.bat`, `.cmd`, `.ps1` and `.psm1` launchers are not read or hashed. This
also applies to the older B0 gate, `capture_sources()` / `baseline_source_identity()`
and `framework_entry_scripts`; excluding them only from the main scanner would
leave indirect Windows-byte dependencies in the source snapshot.

The B0 manifest retains excluded Windows hashes and the previous provenance-module
hash as historical review metadata under `linux_runtime_evolution`. They are not
active launcher pins. Existing B0 algorithm/config hashes and all production
source hashes remain unchanged.

## Checkout bytes

Protected files are hashed as actual bytes. Verification does not normalize line
endings or accept alternate hashes. Existing `.gitattributes` specifies LF for
Python, JSON and Bash, while Windows launchers use CRLF. Windows launcher CRLF/LF
conversion must not affect the Linux source gate.

Five baseline JSON files had stale local CRLF bytes despite their existing LF
attributes: `baseline_registry.json`, `bser_prior_eval.json`,
`direct_boundary_train.json`, `direct_mc_train.json`, and `search_prior_eval.json`.
Their local bytes were restored to the exact existing Git LF blobs before the
manifest refresh. Their JSON values and Git contents did not change. This prevents
the reviewed JSON hashes from immediately failing on a fresh Linux checkout.

## Verification

`tests/test_ch3_baseline_provenance.py` retains four test methods with expanded
cases: the entire source gate must succeed when Windows-byte reads are forbidden;
Windows CRLF/LF and content changes do not affect scanner identity; Bash changes
and new entrypoints remain detectable; manifest integrity and the frozen
production binding remain enforced.

Windows launchers are still checked for existence and expected dispatch. On a
Windows host, all four real BAT entrypoints execute `--help` with the selected
Python. The PowerShell learning entrypoint runs against an inert conda stub to
verify argument forwarding and exit status without training. Linux hosts do not
attempt native Windows execution. The separate OpenMP evolution check compares
Windows command text independently of line endings; it is not a byte pin in Linux
provenance. Linux launcher hashes remain exact.

The requested 38-test suite is:

```bash
python -B -m unittest tests.test_ch3_baseline_validation tests.test_openmp_runtime_env tests.test_ch3_baseline_provenance tests.test_repository_metadata -v
```

It contains 16 baseline validation, 12 OpenMP, four baseline provenance and six
repository metadata tests. Only the validation suite's two temporary synthetic
episodes run; no formal training or 100-episode evaluation is authorized.

Production identity remains:

```text
196 Python files in core/ and chapter3_bser/
7a8dda612fb68c0a63bcc38b04ee0973ac80488bddd6c9fffe1fbf9b2f23f491
```

Historical production manifests, checkpoint identities, baseline algorithms and
HGR code are unchanged. The existing OpenMP opt-in policy and exact historical
engine exception are unchanged (`historical_openmp_assignment_retained = true`).

## Latest recorded local result (2026-09-21)

The complete command above passed **38/38 tests**, with no skips, in **897.328
seconds** on Windows using the AUV conda interpreter. The OpenMP bypass variable
was unset when the suite was launched. Bash checks used Git Bash; the Windows-byte
independence cases and LF/CRLF fixtures exercise the Linux provenance contract,
but this is not a native Linux/PyTorch test run or CI result.

| Module | Passed |
| --- | --- |
| `tests.test_ch3_baseline_validation` | 16/16 |
| `tests.test_openmp_runtime_env` | 12/12 |
| `tests.test_ch3_baseline_provenance` | 4/4 |
| `tests.test_repository_metadata` | 6/6 |

B0's temporary synthetic episode ended successfully at step 161. B1 reached the
normal 400-step timeout with no program failure. All output/schema assertions
passed. These are bounded validation outcomes, not formal performance evidence.
No formal training or 100-episode evaluation ran.
