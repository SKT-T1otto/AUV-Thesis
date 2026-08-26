# CH3 local observation contract (28 dimensions)

Authority: the byte-identical snapshot of `CH3/env.py`, specifically `_get_obs()` and `get_observation_layout()`. This document names no inferred fields and does not change numeric behavior.

| Slice | Field | Normalization | Unknown-target behavior | Role behavior |
|---|---|---|---|---|
| 0:3 | position | none | unchanged | same computation |
| 3:6 | velocity | none | unchanged | same computation |
| 6:9 | navigation_target_delta | none | search waypoint/executor wait point | role-specific target |
| 9:12 | navigation_target_direction | divide by clamped delta norm | follows navigation target | role-specific target |
| 12:15 | known_target_delta | none | exactly zero | populated only for agents whose target-known flag is true |
| 15:16 | navigation_distance | clip(norm/10, 0, 1) | follows navigation target | role-specific target |
| 16:17 | speed | clip(norm/(role v_xy_max+1e-6), 0, 1) | unchanged | role-specific speed limit |
| 17:18 | closing_speed | tanh(dot(v,dir)/(role v_xy_max+1e-6)) | follows navigation target | role-specific speed limit |
| 18:19 | nearest_obstacle_distance | clip(distance/10, 0, 1) | unchanged | same computation |
| 19:20 | waypoint_progress | reached/max(1,total) | unchanged | per-agent counts |
| 20:21 | agent_finished | boolean cast to float | unchanged | per-agent status |
| 21:22 | hold_progress | clip(counter/max(1,role hold steps),0,1) | unchanged | search vs executor hold steps |
| 22:26 | role_onehot | one-hot | unchanged | `search_fast`, `search_balanced`, `search_precise`, `executor` |
| 26:28 | target_knowledge_phase | binary pair | `[1,0]` unknown | `[0,1]` known for that agent |

The local contract contains no CH4/CH5 39-dimensional communication observation. Communication remains CH3's fixed reliable handoff. Target belief and occupancy are internal task/mapping state and are deliberately not inserted into the 28-dimensional local vector; the obstacle-related local field is only the nearest-obstacle distance at slice 18:19.
