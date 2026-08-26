# BSER-E1 results

The frozen 4-profile × 5-scenario × 3-trace × 4-snapshot protocol produced 180
valid instances and 60 `SKIPPED_TERMINATED` records. No replacement snapshots
were introduced. All 180 valid instances passed nonnegativity, monotonicity,
submodularity, partition feasibility, greedy-bound and lazy-equivalence checks.

- Minimum greedy/exact ratio: 0.983549219825.
- Mean greedy/exact ratio: 0.99877892802445.
- Mean BSER minus search-only conditional response time: -0.12075546778151668.
- PSE read-only snapshot available: 180/180 valid instances.
- Deterministic summary SHA-256: `d3e4ffd9eb05e13406de622931e9810df8a7a0647a43ea532dc8b53d2c09bfe7`.

The negative response-time difference means the finite BSER allocation had a
lower conditional response diagnostic on average in this offline matrix. It is
not evidence of improved mission success, capture rate, learned-policy quality,
or completed online reallocation.
