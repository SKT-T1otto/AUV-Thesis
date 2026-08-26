# Public target-route design

The corrected mechanism distinguishes target discovery from executor receipt.

1. `TARGET_FOUND` records that a searcher has found the target, but the controller returns `WAIT_PUBLIC_TARGET_HANDOFF` while the public task state says the executor does not know it.
2. `EXECUTOR_TARGET_RECEIVED` fires only when `executor_knows_target` changes from false to true in consecutive public mission contexts.
3. On receipt, the executor route source priority is:
   - `PUBLIC_HANDOFF_TARGET` when `executor_navigation_target` is present and reachable;
   - `CURRENT_VALID_ROUTE` when an installed route remains valid;
   - `REACHABLE_BELIEF_FALLBACK` as the last public-information fallback.

The corrected controller never reads simulator target truth or a hidden target position. Its mission context is immutable and assembled only from the public task state, public search execution state, and planning view.
