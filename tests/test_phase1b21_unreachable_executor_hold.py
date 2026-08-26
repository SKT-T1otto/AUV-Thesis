import unittest
from dataclasses import replace

import numpy as np

from chapter3_bser.controllers.action_adapter import assignment_to_fixed_actions
from chapter3_bser.online.allocator import BSEROnlineAllocator
from tests.bser_online_test_utils import state_at


class Phase1B21UnreachableExecutorHoldTest(unittest.TestCase):
    def test_unreachable_executor_assignment_does_not_use_straight_line_fallback(self):
        state = state_at(0)
        allocation = BSEROnlineAllocator().allocate(state)
        executor = replace(
            allocation.executor_assignment,
            reachable=False,
            path=(),
            target_region=(99.0, 99.0, 99.0),
        )
        allocation = replace(allocation, executor_assignment=executor)
        actions = assignment_to_fixed_actions(state, allocation)
        expected = np.clip(
            -0.18 * np.asarray(state.agents[state.executor_id].velocity),
            -1.0,
            1.0,
        )
        np.testing.assert_allclose(actions[state.executor_id], expected, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
