import os, shutil
import unittest

from slurmcmc.slurm_utils import SlurmPool


class test_slurmpool_slurm(unittest.TestCase):
    def setUp(self):
        self.work_dir = os.path.dirname(__file__) + '/test_work_dir'

    def tearDown(self):
        if os.path.isdir(self.work_dir):
            shutil.rmtree(self.work_dir)
        pass

    def test_slurmpool_slurm(self):
        self.slurmpool = SlurmPool(self.work_dir, job_name='test_slurmpool', cluster='slurm',
                                   slurm_partition='testSocket', slurm_time=10)
        fun = lambda x: x ** 2
        points = [2, 3, 4]
        res_expected = [fun(point) for point in points]
        res = self.slurmpool.map(fun, points)
        self.assertEqual(res, res_expected)

if __name__ == '__main__':
    unittest.main()
