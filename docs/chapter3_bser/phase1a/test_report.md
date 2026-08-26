# Phase 1A test report

- Preflight baseline unittest: 20/20 passed.
- Final local unittest: 37/37 passed.
- Formal E1: 180 valid, 60 terminated skips, all strict properties passed.
- Final E0: 60/60, maximum absolute difference 0.0, zero event mismatches.
- Final CPU training smoke: PASS; finite actor/critic updates, checkpoint
  roundtrip and post-load step passed; no repository checkpoint written.
- Existing core freeze: 40/40 unchanged by byte and AST hash.
- Fresh remote clone: 37/37 tests, E0 60/60, training smoke and full E1 passed;
  all 180 deterministic instance hashes matched local results.

The training smoke is closure validation only and is not a formal training or
algorithm-performance result.
