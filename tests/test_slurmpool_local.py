import os
import shutil
import unittest

import numpy as np

from slurmcmc.slurm_utils import SlurmPool


class test_slurmpool_local(unittest.TestCase):
    def setUp(self):
        self.work_dir = os.path.dirname(__file__) + '/test_work_dir'
        self.verbosity = 1

    def tearDown(self):
        if os.path.isdir(self.work_dir):
            shutil.rmtree(self.work_dir)
        pass

    def test_slurmpool_local_map(self):
        slurm_pool = SlurmPool(self.work_dir, cluster='local-map', verbosity=self.verbosity)
        fun = lambda x: x ** 2
        points = [2, 3, 4]
        res_expected = [fun(point) for point in points]
        res = slurm_pool.map(fun, points)
        self.assertEqual(res, res_expected)
        self.assertEqual(slurm_pool.num_calls, 1)

    def test_slurmpool_localmap_history(self):
        slurm_pool = SlurmPool(self.work_dir, cluster='local-map', verbosity=self.verbosity)
        fun = lambda x: x ** 2
        points_1 = [2, 3, 4]
        res_1 = slurm_pool.map(fun, points_1)
        points_2 = [5, 6, 7]
        res_2 = slurm_pool.map(fun, points_2)
        points_history_expected = np.append(points_1, points_2).reshape(-1, 1)
        values_history_expected = np.append(res_1, res_2).reshape(-1, 1)
        np.testing.assert_array_equal(slurm_pool.points_history, points_history_expected)
        np.testing.assert_array_equal(slurm_pool.values_history, values_history_expected)

    def test_slurmpool_localmap_history_with_failed_points(self):
        slurm_pool = SlurmPool(self.work_dir, cluster='local-map', verbosity=self.verbosity)
        fun = lambda x: x ** 2
        fun_that_fails = lambda x: None
        points_1 = [2, 3, 4]
        res_1 = slurm_pool.map(fun, points_1)
        points_2 = [5, 6, 7]
        res_2 = slurm_pool.map(fun_that_fails, points_2)

        np.testing.assert_array_equal(slurm_pool.points_history, np.array(points_1).reshape(-1, 1))
        np.testing.assert_array_equal(slurm_pool.values_history, np.array(res_1).reshape(-1, 1))
        np.testing.assert_array_equal(res_2, None)
        np.testing.assert_array_equal(slurm_pool.failed_points_history, np.array(points_2).reshape(-1, 1))

    def test_slurmpool_localmap_2params(self):
        slurm_pool = SlurmPool(self.work_dir, cluster='local-map', verbosity=self.verbosity)
        fun = lambda x: x[0] ** 2 + x[1] ** 2
        points = [[2, 3], [3, 4], [4, 5]]
        res_expected = [fun(point) for point in points]
        res = slurm_pool.map(fun, points)
        self.assertEqual(res, res_expected)

    def test_slurmpool_localmap_history_2params(self):
        slurm_pool = SlurmPool(self.work_dir, cluster='local-map', verbosity=self.verbosity)
        fun = lambda x: x[0] ** 2 + x[1] ** 2
        points_1 = [[2, 3], [3, 4], [4, 5]]
        res_1 = slurm_pool.map(fun, points_1)
        points_2 = [[5, 6], [7, 8], [8, 9]]
        res_2 = slurm_pool.map(fun, points_2)
        points_history_expected = np.vstack([np.array(points_1), np.array(points_2)])
        values_history_expected = np.append(res_1, res_2).reshape(-1, 1)
        np.testing.assert_array_equal(slurm_pool.points_history, points_history_expected)
        np.testing.assert_array_equal(slurm_pool.values_history, values_history_expected)

    def test_slurmpool_localmap_with_budget(self):
        slurm_pool = SlurmPool(self.work_dir, cluster='local-map', verbosity=self.verbosity, budget=2)
        fun = lambda x: x ** 2
        points = [2, 3, 4, 5, 6]
        res_expected = [fun(point) for point in points]
        res = slurm_pool.map(fun, points)
        self.assertEqual(res, res_expected)
        self.assertEqual(slurm_pool.num_calls, 3)

    def test_slurmpool_local(self):
        slurm_pool = SlurmPool(self.work_dir, cluster='local', verbosity=self.verbosity)
        fun = lambda x: x ** 2
        points = [2, 3, 4]
        res_expected = [fun(point) for point in points]
        res = slurm_pool.map(fun, points)
        self.assertEqual(res, res_expected)

    def test_slurmpool_local_2params(self):
        slurm_pool = SlurmPool(self.work_dir, cluster='local', verbosity=self.verbosity)
        fun = lambda x: x[0] ** 2 + x[1] ** 2
        points = [[2, 3], [3, 4], [4, 5]]
        res_expected = [fun(point) for point in points]
        res = slurm_pool.map(fun, points)
        self.assertEqual(res, res_expected)

    def test_slurmpool_local_2params_with_log_file(self):
        slurm_pool = SlurmPool(self.work_dir, cluster='local', verbosity=self.verbosity, log_file='log_file.txt')
        fun = lambda x: x[0] ** 2 + x[1] ** 2
        points = [[2, 3], [3, 4], [4, 5]]
        slurm_pool.map(fun, points)

    def test_slurmpool_fail_on_existing_work_dir(self):
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.work_dir + '/0', exist_ok=True)
        with self.assertRaises(ValueError):
            SlurmPool(self.work_dir, cluster='local', verbosity=self.verbosity)

    def test_slurmpool_localmap_2params_3outputs(self):
        slurm_pool = SlurmPool(self.work_dir, cluster='local-map', verbosity=self.verbosity)
        fun = lambda x: [x[0] ** 2 + x[1] ** 2, 10 * x[0], 10 * x[1]]
        points = [[2, 3], [3, 4], [4, 5]]
        res_expected = [fun(point) for point in points]
        res = slurm_pool.map(fun, points)
        self.assertEqual(res, res_expected)

    def test_slurmpool_localmap_2params_3outputs_history_with_failed_points(self):
        slurm_pool = SlurmPool(self.work_dir, cluster='local-map', verbosity=self.verbosity)
        fun = lambda x: [2 * x[0], 3 * x[0], 4 * x[1]]
        fun_that_partially_fails_1 = lambda x: [2 * x[0], None, 4 * x[1]]
        fun_that_partially_fails_2 = lambda x: [2 * x[0], 3 * x[0], np.nan]
        points = [[2, 3], [3, 4], [4, 5]]
        res = slurm_pool.map(fun, points)
        _ = slurm_pool.map(fun_that_partially_fails_1, points)
        _ = slurm_pool.map(fun_that_partially_fails_2, points)

        np.testing.assert_array_equal(slurm_pool.points_history, np.array(points))
        np.testing.assert_array_equal(slurm_pool.values_history, np.array(res).reshape(-1, 1))
        np.testing.assert_array_equal(slurm_pool.failed_points_history, np.vstack([points, points]))


if __name__ == '__main__':
    unittest.main()
