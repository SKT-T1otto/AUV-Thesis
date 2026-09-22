"""B3: BSER plus boundary-conditioned, independent MADDPG training."""
from ..common.train import BaselineTrainer


class DirectBoundaryTrainer(BaselineTrainer):
    baseline = "B3_direct_boundary"


B3Trainer = DirectBoundaryTrainer
