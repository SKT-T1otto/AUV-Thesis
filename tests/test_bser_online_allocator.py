import unittest

from chapter3_bser.online.allocator import BSEROnlineAllocator
from tests.bser_online_test_utils import state_at


class OnlineAllocatorTest(unittest.TestCase):
    def test_same_state_has_same_allocation_hash(self):
        allocator = BSEROnlineAllocator()
        first = allocator.allocate(state_at(0))
        second = allocator.allocate(state_at(0))
        self.assertEqual(first.allocation_sha256, second.allocation_sha256)
        self.assertGreaterEqual(first.objective_value, 0.0)


if __name__ == "__main__": unittest.main()
