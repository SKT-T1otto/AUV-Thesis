# BSER Phase 1A summary

Phase 1A implements a defensive read-only planning state, pure travel-cost
queries, deterministic search/standby candidates, the coupled monotone
submodular BSER objective, exact/standard/lazy solvers, and three offline
comparators. No existing core Python source changed and no formal training ran.

Formal E1 retained 180 valid instances and 60 terminated skips. Every valid
instance passed nonnegativity, monotonicity, submodularity, partition,
approximation-bound and lazy-equivalence checks. Minimum/mean greedy-exact
ratios were 0.983549219825/0.99877892802445. PSE snapshots were available for
all valid instances. BSER minus search-only conditional response time averaged
-0.12075546778151668.

Local verification passed 37/37 tests, E0 60/60 with zero numeric/event drift,
the CPU training smoke, and the 40-file core freeze. This phase makes no online
task-success or trained-policy claim and does not start Phase 1B.

A fresh GitHub clone independently reproduced 37/37 tests, E0 60/60, the CPU
training smoke, 180 E1 instance hashes and deterministic summary SHA-256. The
verified remote branch is `chapter3-bser-phase1a`; the annotated verification
tag is `chapter3-bser-phase1a-e1-verified-v1`.
