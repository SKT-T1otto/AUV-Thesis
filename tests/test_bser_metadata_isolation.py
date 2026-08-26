import unittest
from dataclasses import fields
from core.mapping.planning_state import PlanningStateView
from chapter3_bser.experiments.instance_builder import PlanningSnapshotMetadata

class MetadataIsolationTest(unittest.TestCase):
    def test_experiment_fields_exist_only_in_metadata(self):
        state={field.name for field in fields(PlanningStateView)}; metadata={field.name for field in fields(PlanningSnapshotMetadata)}
        self.assertFalse(state & {"profile","scenario_id","scenario_seed","action_trace","obstacle_layout_id"}); self.assertTrue({"profile","scenario_id","action_trace","obstacle_layout_id"}<=metadata)
