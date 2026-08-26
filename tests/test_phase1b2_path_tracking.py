import unittest

from chapter3_bser.controllers.path_tracker import PathTracker


class Phase1B2PathTrackingTest(unittest.TestCase):
    def test_tracks_path_in_order_without_jumping_to_final_waypoint(self):
        tracker = PathTracker(threshold=0.5)
        path = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        final = (3.0, 0.0, 0.0)

        first = tracker.tracking_target(0, (0.0, 0.0, 0.0), path, final)
        self.assertEqual(first, path[1])
        self.assertNotEqual(first, final)

        unchanged = tracker.tracking_target(0, (0.4, 0.0, 0.0), path, final)
        self.assertEqual(unchanged, path[1])

        second = tracker.tracking_target(0, (1.0, 0.0, 0.0), path, final)
        self.assertEqual(second, path[2])
        completed = tracker.tracking_target(0, (2.0, 0.0, 0.0), path, final)
        self.assertEqual(completed, final)


if __name__ == "__main__":
    unittest.main()
