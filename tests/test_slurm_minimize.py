import unittest

import numpy as np
from scipy.optimize import rosen

from slurmcmc.optimization import slurm_minimize


def loss_fun(x):
    return rosen(x)


def constraint_fun(x, r_constraint=3, x0_constraint=-1, y0_constraint=-1):
    # return > 0 for violation
    if (x[0] - x0_constraint) ** 2 + (x[1] - y0_constraint) ** 2 > r_constraint ** 2:
        return 1
    else:
        return -1


class test_slurm_minimize(unittest.TestCase):
    def setUp(self):
        self.verbosity = 0

    def test_slurm_minimize_2params(self):
        np.random.seed(0)
        self.num_params = 2
        self.param_bounds = [[-5, 5] for _ in range(self.num_params)]
        self.expected_minima_point = np.ones(self.num_params)
        self.num_workers = 5
        self.num_iters = 20

        result = slurm_minimize(loss_fun=loss_fun,
                                param_bounds=self.param_bounds, num_workers=self.num_workers, num_iters=self.num_iters,
                                cluster='local-map', verbosity=self.verbosity,
                                )

        self.assertLessEqual(np.linalg.norm(result['x_min'] - self.expected_minima_point), 0.3)
        self.assertLessEqual(result['loss_min'], 0.05)

    def test_slurm_minimize_2params_with_constraint(self):
        np.random.seed(0)
        self.num_params = 2
        self.param_bounds = [[-5, 5] for _ in range(self.num_params)]
        self.expected_minima_point = np.ones(self.num_params)
        self.num_workers = 10
        self.num_iters = 30

        result = slurm_minimize(loss_fun=loss_fun, constraint_fun=constraint_fun,
                                param_bounds=self.param_bounds, num_workers=self.num_workers, num_iters=self.num_iters,
                                cluster='local-map', verbosity=self.verbosity,
                                )

        self.assertLessEqual(np.linalg.norm(result['x_min'] - self.expected_minima_point), 0.5)
        self.assertLessEqual(result['loss_min'], 0.05)

    def test_slurm_minimize_3params(self):
        np.random.seed(0)
        self.num_params = 3
        self.param_bounds = [[-5, 5] for _ in range(self.num_params)]
        self.expected_minima_point = np.ones(self.num_params)
        self.num_workers = 10
        self.num_iters = 100

        result = slurm_minimize(loss_fun=loss_fun,
                                param_bounds=self.param_bounds, num_workers=self.num_workers, num_iters=self.num_iters,
                                cluster='local-map', verbosity=self.verbosity,
                                )

        self.assertLessEqual(np.linalg.norm(result['x_min'] - self.expected_minima_point), 0.7)
        self.assertLessEqual(result['loss_min'], 0.1)


if __name__ == '__main__':
    unittest.main()
