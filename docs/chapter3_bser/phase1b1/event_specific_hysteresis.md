# Event-specific hysteresis

The corrected event priority is `EXECUTOR_TARGET_RECEIVED`, `EXECUTOR_INVALID`, `OBSTACLE_DISCOVERED`, `WAYPOINT_STALE`, `BELIEF_SHIFT`, then `PERIODIC_REFRESH`.

Configured cooldowns are 20 steps for belief/periodic events, 5 for obstacle events, 0 for public target receipt, 0 for executor invalidation, and 5 for stale waypoints. Target receipt and executor invalidation are accepted immediately; obstacle and stale events are accepted after their event-local cooldown. Belief and periodic candidates additionally require objective gain strictly above 1% of the prior objective magnitude.

Rejected optimizer attempts mark the corresponding event cooldown, preventing repeated expensive calls on an unchanged condition. After policy acceptance, waypoint stabilization preserves unaffected agents and rejects affected-agent switches shorter than 1.0 m. Candidate allocations are committed atomically only when all required affected assignments are valid.
