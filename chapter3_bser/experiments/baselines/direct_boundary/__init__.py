"""B3 boundary-conditioned MADDPG baseline."""
from .train import B3Trainer, DirectBoundaryTrainer

__all__ = ("B3Trainer", "DirectBoundaryTrainer")
