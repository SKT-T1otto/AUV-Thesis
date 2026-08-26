import unittest

from core.env.observation_contract import FIELDS, OBSERVATION_DIM, ROLE_ORDER, validate_contract


class ObservationContractTests(unittest.TestCase):
    def test_exact_layout(self):
        validate_contract()
        self.assertEqual(OBSERVATION_DIM, 28)
        self.assertEqual(ROLE_ORDER, ("search_fast", "search_balanced", "search_precise", "executor"))
        self.assertEqual(
            [(field.name, field.start, field.end) for field in FIELDS],
            [
                ("position", 0, 3), ("velocity", 3, 6), ("navigation_target_delta", 6, 9),
                ("navigation_target_direction", 9, 12), ("known_target_delta", 12, 15),
                ("navigation_distance", 15, 16), ("speed", 16, 17), ("closing_speed", 17, 18),
                ("nearest_obstacle_distance", 18, 19), ("waypoint_progress", 19, 20),
                ("agent_finished", 20, 21), ("hold_progress", 21, 22),
                ("role_onehot", 22, 26), ("target_knowledge_phase", 26, 28),
            ],
        )
        known_delta = next(field for field in FIELDS if field.name == "known_target_delta")
        self.assertEqual(known_delta.target_unknown_behavior, "exact zeros")


if __name__ == "__main__":
    unittest.main()
