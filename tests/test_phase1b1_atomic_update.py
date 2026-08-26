import unittest
from dataclasses import replace

from chapter3_bser.online.allocator import BSEROnlineAllocator
from tests.bser_online_test_utils import state_at


class MissingRouteAllocator(BSEROnlineAllocator):
    def _partial_search_candidates(self, state, affected):
        return ()


class AtomicUpdateTest(unittest.TestCase):
    def test_missing_affected_route_preserves_whole_allocation(self):
        base = BSEROnlineAllocator().allocate(state_at(0))
        result, ok, reason = MissingRouteAllocator().allocate_partial(state_at(1), base, affected_searcher_ids=(0,), trigger_reason="TEST")
        self.assertFalse(ok)
        self.assertEqual(result, base)
        self.assertEqual(reason, "ATOMIC_REJECT_MISSING_SEARCH_ROUTE")


if __name__ == "__main__": unittest.main()
