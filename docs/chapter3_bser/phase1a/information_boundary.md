# Information boundary

BSER uses current agent positions, velocities, role labels, public speed and
sensor limits, navigation targets, the current target-belief grid, the current
planner occupancy/free/occupied/unknown masks, map revision, mission step,
scenario profile, layout identity, and current read-only reachability.

It does not use the true target coordinate, target future trajectory, task
target tensor, future observations, final episode result, training labels,
physical obstacle objects, hidden obstacle masks, or undiscovered obstacles.
The extractor never calls the target-state or mapping-metrics facade methods;
the latter internally compute truth-based audit metrics unsuitable for BSER.

For M10/M20, occupancy is copied only from the online mapper's current
probability and classification arrays. A runtime guard test raises on attempted
access to privileged target/obstacle attributes. For M90, the protocol declares
the planner's current oracle validity map observable, so its current valid mask
is copied as deterministic occupancy. This does not grant target truth.

The PSE snapshot baseline reads only the current four navigation targets and
constructs read-only paths from the same planning snapshot. It neither changes
PSE nor projects its continuous waypoints into the BSER candidate set. PSE and
BSER share the evaluation model but have different feasible waypoint spaces.

E1 validates finite objective properties, approximation behavior and solver
efficiency. It observes neither mission success rate nor an online intervention,
so it cannot replace formal online performance or multi-seed training studies.
