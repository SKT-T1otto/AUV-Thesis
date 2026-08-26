# Phase 0B-2.1 test report

## Local candidate

- Complete unittest discovery: 20/20 passed (original 17 plus 3 metadata tests)
- Required focused regression set: 4/4 passed
- Source provenance: 27/27 passed
- Core-only E0: 60/60, maximum absolute difference 0.0, event mismatches 0
- CPU training smoke: 2x10 steps, passed
- Actor/critic updates: completed and finite
- Checkpoint roundtrip and post-load step: passed

## GitHub clean clone

- Initial status: clean
- HEAD matched pushed candidate: yes
- Complete unittest discovery: 20/20 passed
- Source provenance: 27/27 passed
- Core-only E0: 60/60, maximum absolute difference 0.0, event mismatches 0
- CPU training smoke: 2x10 steps, passed
- Checkpoint roundtrip and post-load step: passed
- LF attributes: effective for both representative core sources and provenance
- Historical snapshot/sibling repositories/checkpoints/raw data: absent

The disposable remote clone's E0 execution modified only three derived result
files. Its frozen golden trace manifest remained byte-identical.
