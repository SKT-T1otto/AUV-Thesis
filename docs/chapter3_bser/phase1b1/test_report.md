# BSER Phase 1B.1 test report

- Authoritative environment: AUV Python, CPU, no training checkpoint.
- Complete active suite: 78/78 passed in 384.409 seconds; 82 discovered and 4 old E0/blob/metadata checks superseded by the task instructions.
- Final Phase 1B.1-specific suite: 13/13 passed in 258.657 seconds, including the bounded four-method pilot smoke.
- A later complete-suite rerun reached the 15-minute command window without a new result and was stopped after the timeout left a non-computing process; it does not replace the completed 78/78 report.
- Formal pilot: 80/80 condition-episodes completed, 0 failure cases, no formal training and no oracle access.
- Static leak scan and `test_phase1b1_no_target_truth.py`: no forbidden target-truth or hidden obstacle access found.
- Repository scope: `git diff --check` passed; `core/`, `chapter4_rcag/`, and `chapter5_vsgc` changed-file count is zero.

Machine-readable complete-suite evidence is in `active_test_report.json`.
