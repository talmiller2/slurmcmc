import unittest

import numpy as np
from scipy.optimize import rosen

from slurmcmc.mcmc import slurm_mcmc


def log_prob_fun(x):
    return -rosen(x)


class test_slurm_mcmc(unittest.TestCase):

    def test_slurm_mcmc(self):
        num_params = 2
        num_walkers = 4
        num_iters = 100
        minima = np.array([1, 1])
        p0 = np.array([minima for _ in range(num_walkers)]) + 0.5 * np.random.randn(num_walkers, num_params)
        sampler, slurm_pool = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=p0, num_iters=num_iters, cluster='local-map', progress=False)
        samples = sampler.get_chain(flat=True)
        samples = np.vstack([p0, samples]) # p0 is not inherently included
        num_calculated_points = num_walkers * (num_iters + 1)
        np.testing.assert_equal(samples.shape, (num_calculated_points, num_params))
        np.testing.assert_equal(slurm_pool.points_history.shape, (num_calculated_points, num_params))


if __name__ == '__main__':
    unittest.main()
