"""Reuse v1 physical/event repairs with an explicit successor identity."""
from collections import Counter
from chapter3_bser.experiments.baselines.common.runtime import BaselineMissionRuntime
from chapter3_bser.experiments.d2_repair_v1.runtime import RepairedMissionRuntime as ParentRuntime, execution_gates
from .config import VERSION, validate


class RepairedMissionRuntime(ParentRuntime):
    def __init__(self, config, *args, **kwargs):
        self.repair_config = validate(config)
        self.repair_counts = Counter()
        self.raw_components = {}
        self.phase_returns = Counter()
        BaselineMissionRuntime.__init__(self,self.repair_config,*args,**kwargs)

    def summary(self):
        row = super().summary()
        row["repair_version"] = VERSION
        return row
