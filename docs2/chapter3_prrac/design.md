# Chapter 3 PRRAC design

PRRAC v1 combines the existing BSER navigation assignment with a new
phase-routed residual Actor-Critic. It does not implement BEHSP and does not
modify any Chapter 4 algorithm.

## Information and control boundary

Each Actor receives only the frozen local 28D observation. A shared encoder
retains the role one-hot fields, a deterministic soft router mixes three
independent residual experts (search, intercept, hold), and a trust gate scales
the mixed residual. The Actor output is still a 3D residual action in `[-1,1]`.
The Actor never adds the waypoint prior: the environment remains the sole
owner of `prior + scaled residual` composition.

The trust gate uses the shared embedding, router probabilities and these
public observation slices: navigation direction `9:12`, navigation distance
`15:16`, closing speed `17:18`, nearest-obstacle distance `18:19`, waypoint
progress `19:20`, hold progress `21:22`, and target-knowledge phase `26:28`.
Its alignment coefficient is `softplus(raw_alignment_scale)`, so gate strength
is monotonically non-decreasing in residual/navigation cosine alignment when
the base logit is fixed. A zero navigation vector maps to finite zero
alignment.

## Stage and critic semantics

The fixed stage mapping is PRE_FOUND to SEARCH, POST_FOUND to INTERCEPT, and
CONTACT/HOLD/SUCCESS to HOLD. Stage labels are replay metadata and supervision;
they are not Actor features and are not appended to the Critic input.

Each independent twin Critic consumes exactly the frozen 124D joint
observation-action vector and emits three Q heads. Current Q gathers with
`stage_before`; bootstrap Q gathers with `stage_after`. This preserves phase
changes across the transition without changing the 124D contract.

## Isolation

PRRAC composes the frozen PhaseAwareReplayBuffer and Phase1CV2TrainingEnv.
It has its own replay metadata, algorithm state, checkpoints, output namespace,
scripts and diagnostics. Phase 1C-v1/v2 checkpoints are rejected explicitly.
The implementation uses no privileged target, obstacle, future-state or
diagnostic input in the Actor.
