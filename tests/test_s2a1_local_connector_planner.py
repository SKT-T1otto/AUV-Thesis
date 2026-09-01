from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from chapter3_bser.experiments.phase1c_prrac.search_collision_recovery import (
    LastCollisionFreeState,
    canonical_path_hash,
    plan_local_connector,
)
from tests.bser_test_utils import locked, synthetic_state


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

    def test_candidate_tiers_one_two_and_three_are_public_and_ordered(self):
        state = synthetic_state()
        tier1, _, _ = plan_local_connector(
            state, 0, "candidate-0", (2.5, 2.5, 1.0), (1.5, 1.5, 1.0),
            ((1.5, 1.5, 1.0),), None, attempt_id=1,
        )
        self.assertEqual(tier1.endpoint_tier, 1)

        no_free = replace(
            state.occupancy,
            free_mask=locked(np.zeros_like(state.occupancy.free_mask), np.bool_),
        )
        tier2_state = replace(state, occupancy=no_free)
        tier2, _, _ = plan_local_connector(
            tier2_state, 0, "candidate-0", (2.5, 2.5, 1.0), (1.5, 1.5, 1.0),
            ((1.5, 1.5, 1.0),), None, attempt_id=1,
        )
        self.assertEqual(tier2.endpoint_tier, 2)
        self.assertEqual(tier2.source, "LOCAL_CONNECTOR_UNKNOWN_FALLBACK")

        no_valid = replace(
            state.planning_graph,
            valid_mask=locked(np.zeros_like(state.planning_graph.valid_mask), np.bool_),
        )
        tier3_state = replace(tier2_state, planning_graph=no_valid)
        tier3, _, _ = plan_local_connector(
            tier3_state, 0, "candidate-0", (2.5, 2.5, 1.0), (1.5, 1.5, 1.0),
            ((1.5, 1.5, 1.0),), None,
            failed_direction=(1.0, 0.0, 0.0), attempt_id=1,
        )
        self.assertEqual(tier3.endpoint_tier, 3)
        self.assertEqual(tier3.source, "DETERMINISTIC_REVERSE_DIRECTION")

    def test_same_position_anchor_and_base_tracking_endpoint_are_rejected(self):
        state = synthetic_state()
        current = state.agents[0].position
        safe = LastCollisionFreeState(current, (0.0, 0.0, 0.0), 0, 0, current)
        plan, audit, failure = plan_local_connector(
            state, 0, "candidate-0", (2.5, 2.5, 1.0), (1.5, 1.5, 1.0),
            ((1.5, 1.5, 1.0),), safe, attempt_id=1,
        )
        self.assertIsNone(failure)
        self.assertIsNotNone(plan)
        self.assertGreaterEqual(audit.rejected_same_position_count, 2)
        self.assertNotEqual(plan.local_endpoint, current)
        self.assertNotEqual(plan.local_endpoint, (1.5, 1.5, 1.0))


if __name__ == "__main__":
    unittest.main()
