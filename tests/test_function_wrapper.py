import unittest
from slurmcmc.slurm_utils import FunctionWrapper

class test_function_wrapper(unittest.TestCase):

    def test_function_wrapper(self):
        foo = lambda x: x + 10
        wrapped_foo = FunctionWrapper(foo)
        foo_values = [wrapped_foo(x) for x in [1, 2, 3]]
        self.assertEqual(foo_values, [11, 12, 13])
        self.assertEqual(wrapped_foo.num_calls, 3)
        self.assertEqual(wrapped_foo.points_history, [1, 2, 3])
        self.assertEqual(wrapped_foo.values_history, [11, 12, 13])


if __name__ == '__main__':
    unittest.main()