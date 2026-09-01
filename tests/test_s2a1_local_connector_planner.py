from __future__ import annotations

from dataclasses import replace
import unittest

from chapter3_bser.experiments.phase1c_prrac.search_collision_recovery import (
    LastCollisionFreeState,
    canonical_path_hash,
    plan_local_connector,
)
from tests.bser_test_utils import synthetic_state


class LocalConnectorPlannerTests(unittest.TestCase):
    def test_last_safe_tier_precedes_graph_cells(self):
        state = synthetic_state()
        safe = LastCollisionFreeState((0.5, 1.5, 1.0), (0.0, 0.0, 0.0), 0, 1, (1.5, 1.5, 1.0))
        plan, audit, failure = plan_local_connector(
            state, 0, "candidate-0", (2.5, 2.5, 1.0), (1.5, 1.5, 1.0),
            ((1.5, 1.5, 1.0),), safe, attempt_id=1,
        )
        self.assertIsNone(failure)
        self.assertEqual(plan.endpoint_tier, 0)
        self.assertNotEqual(plan.local_endpoint, state.agents[0].position)
        self.assertNotEqual(plan.local_endpoint, (1.5, 1.5, 1.0))
        self.assertNotEqual(plan.base_path_hash, plan.overlay_path_hash)
        self.assertEqual(plan.overlay_path_hash, canonical_path_hash(plan.sampled_local_path))
        self.assertEqual(audit.selected_tier, 0)

    def test_plan_does_not_need_current_start_connector(self):
        state = synthetic_state()
        agent = replace(state.agents[0], position=(0.75, 0.75, 1.0))
        state = replace(state, agents=(agent, *state.agents[1:]))
        left, _, _ = plan_local_connector(state, 0, "candidate-0", (2.5, 2.5, 1.0),
                                          (1.5, 1.5, 1.0), ((1.5, 1.5, 1.0),), None,
                                          failed_direction=(1.0, 0.0, 0.0), attempt_id=1)
        right, _, _ = plan_local_connector(state, 0, "candidate-0", (2.5, 2.5, 1.0),
                                           (1.5, 1.5, 1.0), ((1.5, 1.5, 1.0),), None,
                                           failed_direction=(1.0, 0.0, 0.0), attempt_id=1)
        self.assertIsNotNone(left)
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
