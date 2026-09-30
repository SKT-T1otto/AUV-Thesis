# D2 Linux launcher source review

Authorization: 2026-09-30 user request to prepare Linux training for the
four-arm D2 suite. D2 needs no training; D2_B2, D2_B3 and D2_HGR start from
scratch. Formal execution remains manual.

The exact successor adds linux.py, linux_provenance.py and two Bash launchers.
The only changed protected file is safe_search_v1/provenance.py, dispatching
this successor. All parent manifests and 27 historical records stay frozen.
Every complete byte profile, parent bytes, review bytes and before/after
transition is checked. No core, native trainer, algorithm, device selection,
budget, task protocol, planner or network contract changes are included.

The wrapper checks source/assets, rejects relocated absolute input/output
paths and imports native dependencies before calling the existing executor.
Dependency checks construct no policy or simulator. CPU remains the actual
training device; CUDA availability is reported separately, without enabling it.
Training accepts only the three learning arms; evaluation accepts all four.
No --execute means read-only preview. Explicit execution holds an exclusive
wrapper lock, tees Python stdout/stderr to a unique launch directory and
records completion/failure. The generic launcher does not share this lock;
mixing concurrent launchers is unsupported. Abrupt process death may leave a
running status and stale lock; neither means successful completion. Locks are
never automatically taken over. The underlying executor still refuses to
overwrite unfinished jobs or automatically resume a checkpoint.

Plans are prepared on the execution machine. Training and evaluation run on
that same machine and source checkout. A byte-preserving complete run copy
may be summarized elsewhere using the existing offline summary verifier.
Historical verification evidence remains historical, not current performance.
