import unittest
import os, shutil
import numpy as np
from scipy.optimize import rosen

from slurmcmc.optimization import slurm_minimize


def loss_fun_1d(x):
    return (x - 1) ** 2


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
        self.work_dir = os.path.dirname(__file__) + '/test_work_dir'
        self.verbosity = 1

    def tearDown(self):
        if os.path.isdir(self.work_dir):
            shutil.rmtree(self.work_dir)
        pass

    def test_slurm_minimize_1param(self):
        np.random.seed(0)
        self.num_params = 1
        self.param_bounds = [[-5, 5] for _ in range(self.num_params)]
        self.expected_minima_point = np.ones(self.num_params)
        self.num_workers = 5
        self.num_iters = 10

        result = slurm_minimize(loss_fun=loss_fun_1d,
                                param_bounds=self.param_bounds, num_workers=self.num_workers, num_iters=self.num_iters,
                                cluster='local-map', verbosity=self.verbosity)

        self.assertLessEqual(np.linalg.norm(result['x_min'] - self.expected_minima_point), 0.02)
        self.assertLessEqual(result['loss_min'], 1e-3)

    def test_slurm_minimize_2params(self):
        np.random.seed(0)
        self.num_params = 2
        self.param_bounds = [[-5, 5] for _ in range(self.num_params)]
        self.expected_minima_point = np.ones(self.num_params)
        self.num_workers = 5
        self.num_iters = 20

        result = slurm_minimize(loss_fun=loss_fun,
                                param_bounds=self.param_bounds, num_workers=self.num_workers, num_iters=self.num_iters,
                                cluster='local-map', verbosity=self.verbosity)

        self.assertLessEqual(np.linalg.norm(result['x_min'] - self.expected_minima_point), 0.3)
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


    def test_slurm_minimize_2params_with_constraint(self):
        np.random.seed(0)
        self.num_params = 2
        self.param_bounds = [[-5, 5] for _ in range(self.num_params)]
        self.expected_minima_point = np.ones(self.num_params)
        self.num_workers = 10
        self.num_iters = 30

        result = slurm_minimize(loss_fun=loss_fun, constraint_fun=constraint_fun,
                                param_bounds=self.param_bounds, num_workers=self.num_workers, num_iters=self.num_iters,
                                cluster='local-map', verbosity=self.verbosity)

        self.assertLessEqual(np.linalg.norm(result['x_min'] - self.expected_minima_point), 0.5)
        self.assertLessEqual(result['loss_min'], 0.05)

    def test_slurm_minimize_2params_with_constraint_from_init_points(self):
        np.random.seed(0)
        self.num_params = 2
        self.param_bounds = [[-5, 5] for _ in range(self.num_params)]
        self.expected_minima_point = np.ones(self.num_params)
        self.num_workers = 10
        self.num_iters = 30

        init_points = [np.array([-1, -1]) + np.random.rand(2) for _ in range(self.num_workers)]
        result = slurm_minimize(loss_fun=loss_fun, constraint_fun=constraint_fun, init_points=init_points,
                                param_bounds=self.param_bounds, num_workers=self.num_workers, num_iters=self.num_iters,
                                cluster='local-map', verbosity=self.verbosity)

        self.assertLessEqual(np.linalg.norm(result['x_min'] - self.expected_minima_point), 0.5)
        self.assertLessEqual(result['loss_min'], 0.05)

    def test_slurm_minimize_2params_with_constraint_from_illegal_init_points(self):
        np.random.seed(0)
        self.num_params = 2
        self.param_bounds = [[-5, 5] for _ in range(self.num_params)]
        self.expected_minima_point = np.ones(self.num_params)
        self.num_workers = 10
        self.num_iters = 30

        with self.assertRaises(ValueError):
            init_points = [np.array([-4, -4]) + 0.1 * np.random.rand(2) for _ in range(self.num_workers)]
            slurm_minimize(loss_fun=loss_fun, constraint_fun=constraint_fun, init_points=init_points,
                           param_bounds=self.param_bounds, num_workers=self.num_workers, num_iters=self.num_iters,
                           cluster='local-map', verbosity=self.verbosity)

    def test_slurm_minimize_2params_with_checkpoint(self):
        np.random.seed(0)
        self.num_params = 2
        self.param_bounds = [[-5, 5] for _ in range(self.num_params)]
        self.expected_minima_point = np.ones(self.num_params)
        self.num_workers = 5
        self.num_iters = 20

        # run and save checkpoint
        res_1 = slurm_minimize(loss_fun=loss_fun,
                                 param_bounds=self.param_bounds, num_workers=self.num_workers, num_iters=self.num_iters,
                                 cluster='local-map', verbosity=self.verbosity,
                                 save_checkpoint=True, load_checkpoint=False,
                                 )
        self.assertEqual(res_1['slurm_pool'].num_calls, self.num_iters)
        self.assertEqual(len(res_1['slurm_pool'].points_history), self.num_iters * self.num_workers)

        # run again from previous checkpoint
        res_2 = slurm_minimize(loss_fun=loss_fun,
                                 param_bounds=self.param_bounds, num_workers=self.num_workers, num_iters=self.num_iters,
                                 cluster='local-map', verbosity=self.verbosity,
                                 save_checkpoint=True, load_checkpoint=True,
                                 )
        self.assertEqual(res_2['slurm_pool'].num_calls, 2 * self.num_iters)
        self.assertEqual(len(res_2['slurm_pool'].points_history), 2 * self.num_iters * self.num_workers)


if __name__ == '__main__':
    unittest.main()
