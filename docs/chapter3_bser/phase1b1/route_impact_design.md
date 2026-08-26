# Route-impact design

For every active searcher route and the executor route, the evaluator compares consecutive public occupancy/planning views and records:

- new occupied cells intersecting the exact path;
- new occupied cells intersecting a one-cell 3-D path corridor;
- positive occupancy-probability mass added inside that corridor;
- route reachability under the current planning graph;
- connected-component changes at path endpoints;
- relative planning-cost increase from the installed route cost.

A route is impacted if any direct/corridor intersection exists, it becomes unreachable, its component changes, its cost increase reaches 0.15, or corridor probability mass reaches 0.20. Off-route obstacle events are retained in diagnostics but do not call the optimizer. Impacted searchers and the executor form the local replanning scope; unaffected assignments are copied byte-for-byte into the candidate allocation.
