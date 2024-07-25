import os
import unittest

from slurmcmc.general_utils import delete_directory
from slurmcmc.slurm_utils import SlurmPool


class test_map_slurm(unittest.TestCase):
    def setUp(self):
        self.work_dir = os.path.dirname(__file__) + '/test_work_dir_' + self._testMethodName

    def tearDown(self):
        delete_directory(self.work_dir)

    def test_slurmpool_slurm(self):
        slurm_pool = SlurmPool(self.work_dir, job_name='test_slurmpool',
                               # cluster='slurm', slurm_partition='socket',
                               cluster='slurm', slurm_constraint='serial',
                               )
        fun = lambda x: x ** 2
        points = [2, 3, 4]
        res_expected = [fun(point) for point in points]
        res = slurm_pool.map(fun, points)
        self.assertEqual(res, res_expected)


if __name__ == '__main__':
    unittest.main()
