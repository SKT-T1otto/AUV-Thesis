import unittest

from chapter3_bser.diagnostics.event_semantics import (
    AssignmentVersionTracker,
    diagnostic_event_key,
)
from chapter3_bser.experiments.phase1b3a_diagnosis.run_diagnosis import _duplicate_summary


class Phase1B3ADuplicateEventKeysTest(unittest.TestCase):
    def test_versions_change_only_with_allocation_hash(self):
        tracker = AssignmentVersionTracker()
        self.assertEqual(tracker.observe("allocation-a"), 1)
        self.assertEqual(tracker.observe("allocation-a"), 1)
        self.assertEqual(tracker.observe("allocation-b"), 2)
        self.assertEqual(tracker.observe("allocation-b"), 2)

    def test_key_is_stable_and_never_suppresses_duplicate_events(self):
        first = diagnostic_event_key("EXECUTOR_INVALID", 3, 1, 7, "SEARCH")
        second = diagnostic_event_key("EXECUTOR_INVALID", 3, 1, 7, "SEARCH")
        self.assertEqual(first, second)
        detected_events = [first, second]
        self.assertEqual(len(detected_events), 2)

    def test_aggregate_keys_are_namespaced_by_episode(self):
        key = diagnostic_event_key("EXECUTOR_INVALID", 3, 1, 7, "SEARCH")
        rows = [
            {
                "event_type": "EXECUTOR_INVALID",
                "scenario_seed": seed,
                "episode_index": episode,
                "agent_id": 3,
                "step": 1,
                "assignment_version": 1,
                "map_revision": 7,
                "mission_phase": "SEARCH",
                "event_key": str(key),
            }
            for seed, episode in ((2729, 2), (2731, 1))
        ]
        summary = _duplicate_summary(rows)[0]
        self.assertEqual(summary["unique_event_key_count"], 2)
        self.assertEqual(summary["repeated_same_key_count"], 0)


if __name__ == "__main__":
    unittest.main()
