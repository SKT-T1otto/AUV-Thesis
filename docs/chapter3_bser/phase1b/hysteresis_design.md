# Replanning hysteresis design

The hysteresis policy prevents ordinary snapshot changes from invoking the optimizer every environment step.

- `TARGET_FOUND` and `OBSTACLE_DISCOVERED` are critical and bypass cooldown/gain checks.
- Other events require cooldown expiry and an objective improvement strictly greater than `0.01`.
- The configured cooldown is 20 environment steps.
- No event means no allocation call.
- The no-hysteresis ablation preserves the event detector but bypasses cooldown and gain filtering.

The policy records the last replan step and returns an explicit decision reason, objective gain, and remaining cooldown. Thresholds are configuration values, not learned or E2-tuned values.

E2 supports the intended suppression behavior: the complete Event-BSER method averaged 8.0 replans per episode with a 29.48-step mean interval. Removing hysteresis increased this to 268.3 replans per episode with a 1.00-step mean interval. These are descriptive measurements from the frozen five-seed protocol, not a significance claim.
