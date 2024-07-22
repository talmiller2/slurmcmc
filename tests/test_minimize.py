import os
import shutil
import unittest

import numpy as np
import torch
from scipy.optimize import rosen
from slurmcmc.optimization import slurm_minimize


def loss_fun_1d(x):
    return (x - 1) ** 2


def loss_fun(x):
    return rosen(x)


def loss_fun_with_extra_arg(x, extra_arg):
    if extra_arg == 'sunny':
        return rosen(x)
    else:
        return None


r_constraint = 3
x0_constraint = -1
y0_constraint = -1


def constraint_fun(x):
    # return > 0 for violation
    if (x[0] - x0_constraint) ** 2 + (x[1] - y0_constraint) ** 2 > r_constraint ** 2:
        return 1
    else:
        return -1


def constraint_fun_with_extra_arg(x, extra_arg):
    # return > 0 for violation
    if extra_arg != 'sunny':
        return None
    else:
        if (x[0] - x0_constraint) ** 2 + (x[1] - y0_constraint) ** 2 > r_constraint ** 2:
            return 1
        else:
            return -1


class test_minimize(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        self.work_dir = os.path.dirname(__file__) + '/test_work_dir'
        self.verbosity = 1

    def tearDown(self):
        if os.path.isdir(self.work_dir):
            shutil.rmtree(self.work_dir)
        pass

    def test_slurm_minimize_1param(self):
        num_params = 1
        param_bounds = [[-5, 5] for _ in range(num_params)]
        expected_minima_point = np.ones(num_params)
        num_workers = 5
        num_iters = 10

        result = slurm_minimize(loss_fun=loss_fun_1d,
                                param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                                cluster='local-map', verbosity=self.verbosity)

        self.assertLessEqual(np.linalg.norm(result['x_min'] - expected_minima_point), 0.02)
        self.assertLessEqual(result['loss_min'], 1e-3)

    def test_slurm_minimize_2params(self):
        num_params = 2
        param_bounds = [[-5, 5] for _ in range(num_params)]
        expected_minima_point = np.ones(num_params)
        num_workers = 5
        num_iters = 20

        result = slurm_minimize(loss_fun=loss_fun,
                                param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                                cluster='local-map', verbosity=self.verbosity)

        self.assertLessEqual(np.linalg.norm(result['x_min'] - expected_minima_point), 0.3)
        self.assertLessEqual(result['loss_min'], 0.05)

    def test_slurm_minimize_3params(self):
        num_params = 3
        param_bounds = [[-5, 5] for _ in range(num_params)]
        expected_minima_point = np.ones(num_params)
        num_workers = 10
        num_iters = 100

        result = slurm_minimize(loss_fun=loss_fun,
                                param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                                cluster='local-map', verbosity=self.verbosity,
                                )

        self.assertLessEqual(np.linalg.norm(result['x_min'] - expected_minima_point), 0.7)
        self.assertLessEqual(result['loss_min'], 0.1)

    def test_slurm_minimize_2params_with_constraint(self):
        num_params = 2
        param_bounds = [[-5, 5] for _ in range(num_params)]
        expected_minima_point = np.ones(num_params)
        num_workers = 10
        num_iters = 30

        result = slurm_minimize(loss_fun=loss_fun, constraint_fun=constraint_fun,
                                param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                                cluster='local-map', verbosity=self.verbosity)

        self.assertLessEqual(np.linalg.norm(result['x_min'] - expected_minima_point), 0.5)
        self.assertLessEqual(result['loss_min'], 0.05)

    def test_slurm_minimize_2params_with_constraint_from_init_points(self):
        num_params = 2
        param_bounds = [[-5, 5] for _ in range(num_params)]
        expected_minima_point = np.ones(num_params)
        num_workers = 10
        num_iters = 40
        np.random.seed(0)

        init_points = [np.array([-1, -1]) + np.random.rand(2) for _ in range(num_workers)]
        result = slurm_minimize(loss_fun=loss_fun, constraint_fun=constraint_fun, init_points=init_points,
                                param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                                cluster='local-map', verbosity=self.verbosity)

        self.assertLessEqual(np.linalg.norm(result['x_min'] - expected_minima_point), 0.5)
        self.assertLessEqual(result['loss_min'], 0.05)

    def test_slurm_minimize_2params_with_constraint_from_illegal_init_points(self):
        num_params = 2
        param_bounds = [[-5, 5] for _ in range(num_params)]
        num_workers = 10
        num_iters = 30

        with self.assertRaises(ValueError):
            init_points = [np.array([-4, -4]) + 0.1 * np.random.rand(2) for _ in range(num_workers)]
            slurm_minimize(loss_fun=loss_fun, constraint_fun=constraint_fun, init_points=init_points,
                           param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                           cluster='local-map', verbosity=self.verbosity)

    def test_slurm_minimize_2params_with_restart(self):
        num_params = 2
        param_bounds = [[-5, 5] for _ in range(num_params)]
        num_workers = 5
        num_iters = 20

        # run and save restart
        res_1 = slurm_minimize(loss_fun=loss_fun,
                               param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                               cluster='local-map', verbosity=self.verbosity,
                               work_dir=self.work_dir, save_restart=True, load_restart=False,
                               )
        self.assertEqual(res_1['slurm_pool'].num_calls, num_iters)
        self.assertEqual(len(res_1['slurm_pool'].points_history), num_iters * num_workers)
        self.assertEqual(res_1['ini_iter'], num_iters)

        # run again from previous restart
        res_2 = slurm_minimize(loss_fun=loss_fun,
                               param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                               cluster='local-map', verbosity=self.verbosity,
                               work_dir=self.work_dir, save_restart=True, load_restart=True,
                               )
        self.assertEqual(res_2['slurm_pool'].num_calls, 2 * num_iters)
        self.assertEqual(len(res_2['slurm_pool'].points_history), 2 * num_iters * num_workers)
        self.assertEqual(res_2['ini_iter'], 2 * num_iters)

    def test_slurm_minimize_2params_with_log_file(self):
        num_params = 2
        param_bounds = [[-5, 5] for _ in range(num_params)]
        num_workers = 4
        num_iters = 3

        result = slurm_minimize(loss_fun=loss_fun,
                                param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                                cluster='local-map',
                                verbosity=self.verbosity, slurm_vebosity=self.verbosity,
                                work_dir=self.work_dir, log_file='log_file.txt')

        self.assertTrue(os.path.isfile(self.work_dir + '/log_file.txt'), 'log_file was not created.')

    def test_slurm_minimize_2params_with_constraint_and_with_extra_arg_fail(self):
        num_params = 2
        param_bounds = [[-5, 5] for _ in range(num_params)]
        expected_minima_point = np.ones(num_params)
        num_workers = 10
        num_iters = 30

        with self.assertRaises(TypeError):
            result = slurm_minimize(loss_fun=loss_fun_with_extra_arg, constraint_fun=constraint_fun_with_extra_arg,
                                    extra_arg=None,  # extra_arg not supplied and therefore should fail
                                    param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                                    cluster='local-map',
                                    verbosity=self.verbosity, slurm_vebosity=self.verbosity,
                                    work_dir=self.work_dir)

    def test_slurm_minimize_2params_with_constraint_and_with_extra_arg(self):
        num_params = 2
        param_bounds = [[-5, 5] for _ in range(num_params)]
        expected_minima_point = np.ones(num_params)
        num_workers = 10
        num_iters = 30

        result = slurm_minimize(loss_fun=loss_fun_with_extra_arg, constraint_fun=constraint_fun_with_extra_arg,
                                extra_arg='sunny',
                                param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                                cluster='local-map',
                                verbosity=self.verbosity, slurm_vebosity=self.verbosity,
                                work_dir=self.work_dir)
        self.assertLessEqual(np.linalg.norm(result['x_min'] - expected_minima_point), 0.5)
        self.assertLessEqual(result['loss_min'], 0.05)

    def test_slurm_minimize_1param_botorch(self):
        num_params = 1
        param_bounds = [[-5, 5] for _ in range(num_params)]
        expected_minima_point = np.ones(num_params)
        num_workers = 5
        num_iters = 10

        result = slurm_minimize(loss_fun=loss_fun_1d,
                                param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                                optimizer_package='botorch',
                                cluster='local-map', verbosity=self.verbosity)

        self.assertLessEqual(np.linalg.norm(result['x_min'] - expected_minima_point), 0.02)
        self.assertLessEqual(result['loss_min'], 1e-3)


if __name__ == '__main__':
    unittest.main()
