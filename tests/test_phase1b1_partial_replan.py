import unittest

from chapter3_bser.online.allocator import BSEROnlineAllocator
from tests.bser_online_test_utils import shifted_belief, state_at


class PartialReplanTest(unittest.TestCase):
    def test_only_affected_searcher_is_replaced(self):
        allocator = BSEROnlineAllocator()
        old = allocator.allocate(state_at(0))
        new, ok, _ = allocator.allocate_partial(shifted_belief(), old, affected_searcher_ids=(0,), trigger_reason="TEST")
        self.assertTrue(ok)
        old_map = {item.agent_id:item for item in old.search_assignments}
        new_map = {item.agent_id:item for item in new.search_assignments}
        self.assertEqual(new_map[1], old_map[1])
        self.assertEqual(new_map[2], old_map[2])


if __name__ == "__main__": unittest.main()
