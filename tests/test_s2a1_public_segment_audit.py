from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from chapter3_bser.experiments.phase1c_prrac.search_collision_recovery import audit_public_segment
from core.mapping.planning_state import OccupancyBeliefView
from tests.bser_test_utils import locked, synthetic_state


class PublicSegmentAuditTests(unittest.TestCase):
    def test_clear_segment_is_deterministic_and_spacing_bounded(self):
        state = synthetic_state()
        audit = audit_public_segment(state, (0.5, 0.5, 1.0), (1.5, 0.5, 1.0))
        self.assertTrue(audit.accepted)
        self.assertGreaterEqual(audit.sample_count, 3)
        self.assertEqual(audit, audit_public_segment(state, audit.start, audit.endpoint))

    def test_occupied_endpoint_and_out_of_bounds_are_rejected(self):
        state = synthetic_state()
        occupied = np.asarray(state.occupancy.occupied_mask).copy(); occupied[3] = True
        occupancy = replace(state.occupancy, occupied_mask=locked(occupied, np.bool_))
        state = replace(state, occupancy=occupancy)
        self.assertFalse(audit_public_segment(state, (0.5, 0.5, 1.0), (1.5, 0.5, 1.0)).accepted)
        self.assertFalse(audit_public_segment(state, (0.5, 0.5, 1.0), (-1.0, 0.5, 1.0)).accepted)


if __name__ == "__main__":
    unittest.main()
