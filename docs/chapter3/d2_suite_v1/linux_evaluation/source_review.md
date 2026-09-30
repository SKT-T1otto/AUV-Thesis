# Dedicated Linux evaluation entrypoints review

Authorization: 2026-09-30 user request to finish D2 evaluation and the three
post-training evaluations, with the entire experiment conducted on Linux.
Formal execution remains manual. This is orchestration, not an algorithm or
evaluation-metric change.

Five additions are closed-listed in evaluation_provenance.py: evaluation_cli,
its provenance verifier, and Bash reference, trained and summary entrypoints.
The sole changed protected file is safe_search_v1/provenance.py dispatching
the new exact successor. Previous Linux and suite seals, review records and
all historical provenance remain byte-for-byte frozen. Every profile, parent,
review and exact before/after change is verified with no exemptions.

D2 evaluation selects only the untrained reference and accepts no training
seed or arm overrides. The trained entrypoint defaults to all three learning
arms and all prepared training seeds. Optional arm/seed subsets are explicit.
Every selected training receipt and its final checkpoint bytes must validate
before any requested evaluation starts. Missing receipts appear in read-only
preview; explicit execution rejects them. Corrupt receipts or files fail
closed. No checkpoint is selected by evaluation performance, loaded by this
readiness check, or resumed for training.

Actual evaluation delegates to the existing Linux logger/lock and native
full-terminal evaluator. Default previews start no simulator. Training,
networks, rewards, D2 planning, scene order and metrics stay unchanged. Summary
delegates to the existing offline verifier; partial plans retain null rates
and suite_complete=false, and are never reported as complete automatically.
Local synthetic smoke uses temporary inputs and bounded forced collisions;
it is not formal performance evidence or a native Linux platform run.
