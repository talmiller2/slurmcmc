import os
import unittest

import numpy as np
from scipy.optimize import rosen

from slurmcmc.general_utils import delete_directory_and_wait
from slurmcmc.general_utils import load_restart_file
from slurmcmc.mcmc import slurm_mcmc


def log_prob_fun(x):
    return -rosen(x)


class test_mcmc(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.work_dir = os.path.dirname(__file__) + '/test_work_dir'
        self.verbosity = 1

    def tearDown(self):
        self.assertTrue(delete_directory_and_wait(self.work_dir))

    def test_slurm_mcmc(self):
        num_params = 2
        num_walkers = 10
        num_iters = 3
        minima = np.array([1, 1])
        p0 = np.array([minima for _ in range(num_walkers)]) + 0.5 * np.random.randn(num_walkers, num_params)
        sampler = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=p0, num_iters=num_iters,
                             verbosity=self.verbosity, slurm_vebosity=self.verbosity,
                             cluster='local-map')

        samples = sampler.get_chain(flat=True)
        samples = np.vstack([p0, samples])  # p0 is not inherently included
        num_calculated_points = num_walkers * (num_iters + 1)
        np.testing.assert_equal(samples.shape, (num_calculated_points, num_params))
        np.testing.assert_equal(sampler.pool.points_history.shape, (num_calculated_points, num_params))
        self.assertEqual(sampler.pool.num_calls, 7)

    def test_slurm_mcmc_with_budget(self):
        num_params = 2
        num_walkers = 10
        num_iters = 3
        minima = np.array([1, 1])
        p0 = np.array([minima for _ in range(num_walkers)]) + 0.5 * np.random.randn(num_walkers, num_params)
        sampler = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=p0, num_iters=num_iters,
                             verbosity=self.verbosity, slurm_vebosity=self.verbosity,
                             cluster='local-map', slurm_dict={'budget': 5})
        samples = sampler.get_chain(flat=True)
        samples = np.vstack([p0, samples])  # p0 is not inherently included
        num_calculated_points = num_walkers * (num_iters + 1)
        np.testing.assert_equal(sampler.pool.points_history.shape, (num_calculated_points, num_params))
        self.assertEqual(sampler.pool.num_calls, 8)

    def test_slurm_mcmc_with_log_file(self):
        num_params = 2
        num_walkers = 10
        num_iters = 3
        minima = np.array([1, 1])
        p0 = np.array([minima for _ in range(num_walkers)]) + 0.5 * np.random.randn(num_walkers, num_params)

        sampler = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=p0, num_iters=num_iters,
                             verbosity=self.verbosity, slurm_vebosity=self.verbosity,
                             cluster='local-map',
                             work_dir=self.work_dir, log_file='log_file.txt')

        self.assertTrue(os.path.isfile(self.work_dir + '/log_file.txt'), 'log_file was not created.')

    def test_slurm_mcmc_with_restart(self):
        num_params = 2
        num_walkers = 10
        num_iters = 3

        num_slurm_call_init = 1
        num_slurm_call_mcmc = 2 * num_iters
        num_points_calc_init = num_walkers
        num_points_calc_mcmc = num_walkers * num_iters

        minima = np.array([1, 1])
        p0 = np.array([minima for _ in range(num_walkers)]) + 0.5 * np.random.randn(num_walkers, num_params)

        sampler_1 = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=p0, num_iters=num_iters,
                               verbosity=self.verbosity, slurm_vebosity=self.verbosity,
                               cluster='local-map',
                               work_dir=self.work_dir, save_restart=True, load_restart=False)

        total_num_slurm_call = num_slurm_call_init + num_slurm_call_mcmc
        total_num_points_calc = num_points_calc_init + num_points_calc_mcmc
        self.assertEqual(sampler_1.pool.num_calls, total_num_slurm_call)
        self.assertEqual(len(sampler_1.pool.points_history), total_num_points_calc)
        restart_1 = load_restart_file(self.work_dir, restart_file='mcmc_restart.pkl')
        self.assertEqual(restart_1['ini_iter'], num_iters)

        sampler_2 = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=p0, num_iters=num_iters,
                               verbosity=self.verbosity, slurm_vebosity=self.verbosity,
                               cluster='local-map',
                               work_dir=self.work_dir, save_restart=True, load_restart=True)

        total_num_slurm_call = num_slurm_call_init + 2 * num_slurm_call_mcmc
        total_num_points_calc = num_points_calc_init + 2 * num_points_calc_mcmc
        self.assertEqual(sampler_2.pool.num_calls, total_num_slurm_call)
        self.assertEqual(len(sampler_2.pool.points_history), total_num_points_calc)
        restart_2 = load_restart_file(self.work_dir, restart_file='mcmc_restart.pkl')
        self.assertEqual(restart_2['ini_iter'], 2 * num_iters)


if __name__ == '__main__':
    unittest.main()
