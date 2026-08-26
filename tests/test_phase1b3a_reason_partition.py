import unittest

from chapter3_bser.diagnostics.event_semantics import (
    classify_executor_invalid,
    classify_waypoint_stale,
)


class Phase1B3AReasonPartitionTest(unittest.TestCase):
    def test_executor_primary_partition_and_priority(self):
        cases = (
            (("INSTALLED_ASSIGNMENT_REACHABLE_FALSE", "CURRENT_QUERY_UNREACHABLE"), "INSTALLED_ASSIGNMENT_INVALID"),
            (("CURRENT_QUERY_UNREACHABLE", "RELATIVE_COST_INCREASE"), "HARD_ROUTE_INVALID"),
            (("PATH_CELL_BECAME_OCCUPIED",), "HARD_ROUTE_INVALID"),
            (("COMPONENT_CHANGED",), "HARD_ROUTE_INVALID"),
            (("RELATIVE_COST_INCREASE",), "SOFT_RESPONSE_DEGRADED_ONLY"),
            (("NONFINITE_COST",), "DIAGNOSTIC_INCONSISTENCY"),
        )
        for flags, expected in cases:
            with self.subTest(flags=flags):
                self.assertEqual(classify_executor_invalid(flags), expected)

    def test_waypoint_primary_partition_and_priority(self):
        cases = (
            (("FINAL_WAYPOINT_REACHED", "WAYPOINT_QUERY_UNREACHABLE"), "FINAL_WAYPOINT_REACHED"),
            (("WAYPOINT_QUERY_UNREACHABLE",), "HARD_PATH_INVALID"),
            (("REMAINING_PATH_INVALIDATED",), "HARD_PATH_INVALID"),
            (("COMPONENT_CHANGED",), "HARD_PATH_INVALID"),
            (("NO_ACTIVE_ASSIGNMENT",), "NO_ACTIVE_ASSIGNMENT"),
            (("LOCAL_TRACKING_POINT_REACHED",), "DIAGNOSTIC_INCONSISTENCY"),
        )
        for flags, expected in cases:
            with self.subTest(flags=flags):
                self.assertEqual(classify_waypoint_stale(flags), expected)


if __name__ == "__main__":
    unittest.main()
