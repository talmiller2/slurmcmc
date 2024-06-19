import os
import shutil
import unittest

import numpy as np
from scipy.optimize import rosen

from slurmcmc.mcmc import slurm_mcmc


def log_prob_fun(x):
    return -rosen(x)


class test_slurm_mcmc(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.work_dir = os.path.dirname(__file__) + '/test_work_dir'
        self.verbosity = 1

    def tearDown(self):
        if os.path.isdir(self.work_dir):
            shutil.rmtree(self.work_dir)
        pass

    def test_slurm_mcmc(self):
        num_params = 2
        num_walkers = 10
        num_iters = 3
        minima = np.array([1, 1])
        p0 = np.array([minima for _ in range(num_walkers)]) + 0.5 * np.random.randn(num_walkers, num_params)
        sampler, slurm_pool = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=p0, num_iters=num_iters,
                                         verbosity=self.verbosity, slurm_vebosity=self.verbosity,
                                         cluster='local-map', progress=False)
        samples = sampler.get_chain(flat=True)
        samples = np.vstack([p0, samples])  # p0 is not inherently included
        num_calculated_points = num_walkers * (num_iters + 1)
        np.testing.assert_equal(samples.shape, (num_calculated_points, num_params))
        np.testing.assert_equal(slurm_pool.points_history.shape, (num_calculated_points, num_params))
        self.assertEqual(slurm_pool.num_calls, 7)

    def test_slurm_mcmc_with_budget(self):
        num_params = 2
        num_walkers = 10
        num_iters = 3
        minima = np.array([1, 1])
        p0 = np.array([minima for _ in range(num_walkers)]) + 0.5 * np.random.randn(num_walkers, num_params)
        sampler, slurm_pool = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=p0, num_iters=num_iters,
                                         verbosity=self.verbosity, slurm_vebosity=self.verbosity,
                                         cluster='local-map', progress=False, slurm_dict={'budget': 5})
        samples = sampler.get_chain(flat=True)
        samples = np.vstack([p0, samples])  # p0 is not inherently included
        num_calculated_points = num_walkers * (num_iters + 1)
        np.testing.assert_equal(samples.shape, (num_calculated_points, num_params))
        np.testing.assert_equal(slurm_pool.points_history.shape, (num_calculated_points, num_params))
        self.assertEqual(slurm_pool.num_calls, 8)


    def test_slurm_mcmc_with_log_file(self):
        num_params = 2
        num_walkers = 10
        num_iters = 3
        minima = np.array([1, 1])
        p0 = np.array([minima for _ in range(num_walkers)]) + 0.5 * np.random.randn(num_walkers, num_params)

        sampler, slurm_pool = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=p0, num_iters=num_iters,
                                         verbosity=self.verbosity, slurm_vebosity=self.verbosity,
                                         cluster='local-map', progress=False,
                                         work_dir=self.work_dir, log_file='log_file.txt')

        self.assertTrue(os.path.isfile(self.work_dir + '/log_file.txt'), 'log_file was not created.')


if __name__ == '__main__':
    unittest.main()
