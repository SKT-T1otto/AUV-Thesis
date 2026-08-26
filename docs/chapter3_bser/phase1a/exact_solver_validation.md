# Exact solver validation

For each fixed standby point the exact solver enumerates the Cartesian product
of one `None-or-candidate` choice from each search-agent partition. Joint exact
then enumerates all standby candidates. The reported combination count is
`|Y| product_i (|E_i|+1)` and the solver refuses only cases above the frozen
100000 cap.

In formal E1, all 180 valid K=4 instances completed exact enumeration. The
240 scalability rows contained 180 completed exact cases and 60 explicit
`EXACT_SKIPPED_COMBINATORIAL_LIMIT` cases. No skipped exact case was counted as
a failure or silently replaced by a smaller instance.
