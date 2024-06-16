import os
import unittest

import numpy as np

from slurmcmc.slurm_utils import SlurmPool


class test_slurmpool_local(unittest.TestCase):
    def setUp(self):
        self.work_dir = os.path.dirname(__file__)
        self.slurmpool = SlurmPool(self.work_dir, cluster='local-map')

    def test_slurmpool_map(self):
        fun = lambda x: x ** 2
        points = [2, 3, 4]
        res_expected = [fun(point) for point in points]
        res = self.slurmpool.map(fun, points)
        self.assertEqual(res, res_expected)


    def test_slurmpool_history(self):
        fun = lambda x: x ** 2
        points_1 = [2, 3, 4]
        res_1 = self.slurmpool.map(fun, points_1)
        points_2 = [5, 6, 7]
        res_2 = self.slurmpool.map(fun, points_2)
        np.testing.assert_array_equal(self.slurmpool.points_history, np.array([points_1, points_2]))
        np.testing.assert_array_equal(self.slurmpool.values_history, np.array([res_1, res_2]))

    def test_slurmpool_history_with_failed_points(self):
        fun = lambda x: x ** 2
        fun_that_fails = lambda x: None
        points_1 = [2, 3, 4]
        res_1 = self.slurmpool.map(fun, points_1)
        points_2 = [5, 6, 7]
        res_2 = self.slurmpool.map(fun_that_fails, points_2)

        # print('\n##########')
        # print('self.slurmpool.points_history', self.slurmpool.points_history)
        # print('self.slurmpool.failed_points_history', self.slurmpool.failed_points_history)
        # print('self.slurmpool.values_history', self.slurmpool.values_history)


        # np.testing.assert_array_equal(self.slurmpool.points_history, np.array([points_1]))
        # np.testing.assert_array_equal(self.slurmpool.failed_points_history, np.array([points_2]))
        # np.testing.assert_array_equal(self.slurmpool.values_history, np.array([res_1]))

        np.testing.assert_array_equal(self.slurmpool.points_history, points_1)
        np.testing.assert_array_equal(self.slurmpool.values_history, res_1)
        np.testing.assert_array_equal(res_2, None)
        np.testing.assert_array_equal(self.slurmpool.failed_points_history, points_2)


if __name__ == '__main__':
    unittest.main()
