# Candidate model

Search candidates are constructed deterministically from four available
sources: belief local maxima, high-belief uncovered cells, unknown/free
frontiers, and known-free high-belief cells. The source lists are round-robin
merged, deduplicated by cell, filtered by minimum spatial separation, current
known occupancy, read-only reachability, and a frozen travel-time budget. Each
search agent receives up to K distinct candidates; shortages are recorded and
never filled by duplicating a waypoint.

Standby candidates include the executor's current position, current PSE wait
point, reachable belief peak, belief-weighted representative cell, and further
high-probability reachable cells. IDs and tie breaks are deterministic.

Exact E1 uses K_search=4 and K_standby=4. Scalability uses search sizes
4/8/16/32 and standby sizes 4/8/16. The 100000 exact-combination cap is fixed;
larger cases are explicitly skipped for exact only.
