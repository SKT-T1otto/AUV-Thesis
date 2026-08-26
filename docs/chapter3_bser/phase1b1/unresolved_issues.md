# Unresolved issues

1. Corrected success is 0/20 versus static 4/20, so the preregistered success noninferiority gate fails.
2. Atomic partial replanning rejected 1,234 decisions because at least one affected searcher lacked a valid replacement route.
3. Another 1,720 corrected decisions reached a candidate but produced no assignment change, indicating substantial optimizer work without control impact.
4. Corrected diagnostics contain 7,608 `EXECUTOR_INVALID` and 7,729 `WAYPOINT_STALE` observations. Their persistence should be diagnosed before changing thresholds.
5. Corrected public target receipt occurred only 5 times, and route-source samples include 1,789 `NO_VALID_ROUTE` states; executor reachability remains a primary failure mode.
6. The final complete-suite rerun exceeded the 15-minute command window; the earlier complete 78/78 report and the final 13/13 targeted suite pass remain the available test evidence.

Do not expand validation, enter Phase 1C, or train BSER-RMADDPG until mission-success performance is recovered under a separately approved diagnostic phase.
