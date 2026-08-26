"""Independent Phase 1C PRRAC experiment namespace."""

METHOD = "ch3_bser_rmaddpg_phase1c"
IMPLEMENTATION_VERSION = "bser.phase1c.prrac_v1"
ARCHITECTURE_VERSION = "prrac.phase_routed_residual.v1"
CHECKPOINT_SCHEMA = "bser.phase1c.prrac.training_state.v1"
REPLAY_SCHEMA = "bser.phase1c.prrac_replay.v1"

__all__ = (
    "ARCHITECTURE_VERSION",
    "CHECKPOINT_SCHEMA",
    "IMPLEMENTATION_VERSION",
    "METHOD",
    "REPLAY_SCHEMA",
)
