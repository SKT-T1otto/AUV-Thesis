from __future__ import annotations

import unittest

from chapter3_bser.experiments.phase1c_prrac.search_collision_recovery import CollisionEdgeDetector


class DetectorTests(unittest.TestCase):
    def test_one_edge_per_streak_and_rearm(self):
        d=CollisionEdgeDetector(); self.assertTrue(d.observe(0,True,search_active=True)); self.assertFalse(d.observe(0,True,search_active=True)); self.assertFalse(d.observe(0,False,search_active=True)); self.assertTrue(d.observe(0,True,search_active=True))


if __name__ == "__main__": unittest.main()
