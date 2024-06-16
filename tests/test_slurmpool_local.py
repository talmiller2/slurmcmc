import os, shutil
import unittest

import numpy as np

from slurmcmc.slurm_utils import SlurmPool


class test_slurmpool_local(unittest.TestCase):
    def setUp(self):
        self.work_dir = os.path.dirname(__file__) + '/test_work_dir'

    def tearDown(self):
        if os.path.isdir(self.work_dir):
            shutil.rmtree(self.work_dir)
        pass

    def test_slurmpool_local_map(self):
        self.slurmpool = SlurmPool(self.work_dir, cluster='local-map')
        fun = lambda x: x ** 2
        points = [2, 3, 4]
        res_expected = [fun(point) for point in points]
        res = self.slurmpool.map(fun, points)
        self.assertEqual(res, res_expected)

    def test_slurmpool_localmap_history(self):
        self.slurmpool = SlurmPool(self.work_dir, cluster='local-map')
        fun = lambda x: x ** 2
        points_1 = [2, 3, 4]
        res_1 = self.slurmpool.map(fun, points_1)
        points_2 = [5, 6, 7]
        res_2 = self.slurmpool.map(fun, points_2)
        np.testing.assert_array_equal(self.slurmpool.points_history, np.array([points_1, points_2]))
        np.testing.assert_array_equal(self.slurmpool.values_history, np.array([res_1, res_2]))

    def test_slurmpool_localmap_history_with_failed_points(self):
        self.slurmpool = SlurmPool(self.work_dir, cluster='local-map')
        fun = lambda x: x ** 2
        fun_that_fails = lambda x: None
        points_1 = [2, 3, 4]
        res_1 = self.slurmpool.map(fun, points_1)
        points_2 = [5, 6, 7]
        res_2 = self.slurmpool.map(fun_that_fails, points_2)

        np.testing.assert_array_equal(self.slurmpool.points_history, points_1)
        np.testing.assert_array_equal(self.slurmpool.values_history, res_1)
        np.testing.assert_array_equal(res_2, None)
        np.testing.assert_array_equal(self.slurmpool.failed_points_history, points_2)


    def test_slurmpool_localmap_2params(self):
        self.slurmpool = SlurmPool(self.work_dir, cluster='local-map')
        fun = lambda x: x[0] ** 2 + x[1] ** 2
        points = [[2, 3], [3, 4], [4, 5]]
        res_expected = [fun(point) for point in points]
        res = self.slurmpool.map(fun, points)
        self.assertEqual(res, res_expected)

    def test_slurmpool_local(self):
        self.slurmpool = SlurmPool(self.work_dir, cluster='local')
        fun = lambda x: x ** 2
        points = [2, 3, 4]
        res_expected = [fun(point) for point in points]
        res = self.slurmpool.map(fun, points)
        self.assertEqual(res, res_expected)

    def test_slurmpool_local_2params(self):
        self.slurmpool = SlurmPool(self.work_dir, cluster='local')
        fun = lambda x: x[0] ** 2 + x[1] ** 2
        points = [[2, 3], [3, 4], [4, 5]]
        res_expected = [fun(point) for point in points]
        res = self.slurmpool.map(fun, points)
        self.assertEqual(res, res_expected)


if __name__ == '__main__':
    unittest.main()
