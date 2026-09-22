"""B2: BSER plus direct, independent MADDPG training."""
from ..common.train import BaselineTrainer


class DirectMCTrainer(BaselineTrainer):
    baseline = "B2_direct_mc"


B2Trainer = DirectMCTrainer
