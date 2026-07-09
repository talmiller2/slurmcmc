import os

import numpy as np
import pytest
import torch
from scipy.optimize import rosen

from slurmcmc.optimization import slurm_minimize
from slurmcmc.slurm_utils import is_slurm_cluster
from tests.submitit_defaults import submitit_kwargs


@pytest.fixture()
def loss_fun_1d():
    def _loss_fun_1d(x):
        # nevergrad passes a shape-(1,) array for 1-param problems; squeeze to scalar
        x = float(np.squeeze(x))
        return float((x - 1) ** 2)

    return _loss_fun_1d


@pytest.fixture()
def loss_fun_1d_partially_nan():
    def _loss_fun_1d_partially_nan(x):
        x = float(np.squeeze(x))
        if x > 0.5:
            return float((x - 1) ** 2)
        else:
            return np.nan

    return _loss_fun_1d_partially_nan


@pytest.fixture()
def loss_fun():
    def _loss_fun(x):
        return rosen(x)

    return _loss_fun


@pytest.fixture()
def loss_fun_with_extra_arg():
    def _loss_fun_with_extra_arg(x, extra_arg):
        if extra_arg == 'sunny':
            return rosen(x)
        else:
            return None

    return _loss_fun_with_extra_arg


r_constraint = 3
x0_constraint = -1
y0_constraint = -1


@pytest.fixture()
def constraint_fun():
    def _constraint_fun(x):
        # return > 0 for violation
        if (x[0] - x0_constraint) ** 2 + (x[1] - y0_constraint) ** 2 > r_constraint ** 2:
            return 1
        else:
            return -1

    return _constraint_fun


@pytest.fixture()
def constraint_fun_with_extra_arg():
    def _constraint_fun_with_extra_arg(x, extra_arg):
        # return > 0 for violation
        if extra_arg != 'sunny':
            return None
        else:
            if (x[0] - x0_constraint) ** 2 + (x[1] - y0_constraint) ** 2 > r_constraint ** 2:
                return 1
            else:
                return -1

    return _constraint_fun_with_extra_arg


@pytest.fixture(scope="module")
def imported_constraint_fun_dict():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return {
        'module_dir': os.path.join(base_dir, 'example_module_dir'),
        'module_name': 'example_module',
        'function_name': 'example_constraint_fun'
    }


def test_slurm_minimize_1param(verbosity, seed, loss_fun_1d):
    num_params = 1
    param_bounds = [[-5, 5] for _ in range(num_params)]
    expected_minima_point = np.ones(num_params)
    num_workers = 5
    num_iters = 10

    result = slurm_minimize(loss_fun=loss_fun_1d,
                            param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                            cluster='local-map', verbosity=verbosity)

    assert np.linalg.norm(result['x_min'] - expected_minima_point) <= 0.02
    assert result['loss_min'] <= 1e-3
    assert len(result['candidates_ask_time_per_iter']) == num_iters


def test_slurm_minimize_1param_partially_nan_function(verbosity, seed, loss_fun_1d_partially_nan):
    num_params = 1
    param_bounds = [[-5, 5] for _ in range(num_params)]
    expected_minima_point = np.ones(num_params)
    num_workers = 5
    num_iters = 15

    result = slurm_minimize(loss_fun=loss_fun_1d_partially_nan,
                            param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                            cluster='local-map', verbosity=verbosity)

    assert np.linalg.norm(result['x_min'] - expected_minima_point) <= 0.02
    assert result['loss_min'] <= 1e-3
    assert len(result['slurm_pool'].inds_failed_points) > 0  # check some points indeed count as failing


def test_slurm_minimize_1param_local(work_dir, verbosity, seed, loss_fun_1d):
    num_params = 1
    param_bounds = [[-5, 5] for _ in range(num_params)]
    num_workers = 5
    num_iters = 2

    result = slurm_minimize(loss_fun=loss_fun_1d,
                            param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                            work_dir=work_dir, cluster='local', verbosity=verbosity)


def test_slurm_minimize_2params_local_with_imported_constraint_fun(work_dir, verbosity, seed, loss_fun,
                                                                   imported_constraint_fun_dict):
    num_params = 2
    param_bounds = [[-5, 5] for _ in range(num_params)]
    expected_minima_point = np.ones(num_params)
    num_workers = 3
    num_iters = 3

    result = slurm_minimize(loss_fun=loss_fun, constraint_fun=imported_constraint_fun_dict,
                            param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                            work_dir=work_dir, cluster='local', verbosity=verbosity)

    assert np.linalg.norm(result['x_min'] - expected_minima_point) <= 0.5
    assert result['loss_min'] <= 0.1


def test_slurm_minimize_2params(verbosity, seed, loss_fun):
    num_params = 2
    param_bounds = [[-5, 5] for _ in range(num_params)]
    expected_minima_point = np.ones(num_params)
    num_workers = 5
    num_iters = 20

    result = slurm_minimize(loss_fun=loss_fun,
                            param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                            cluster='local-map', verbosity=verbosity)

    assert np.linalg.norm(result['x_min'] - expected_minima_point) <= 0.3
    assert result['loss_min'] <= 0.05


def test_slurm_minimize_3params(verbosity, seed, loss_fun):
    num_params = 3
    param_bounds = [[-5, 5] for _ in range(num_params)]
    expected_minima_point = np.ones(num_params)
    num_workers = 10
    num_iters = 100

    result = slurm_minimize(loss_fun=loss_fun,
                            param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                            cluster='local-map', verbosity=verbosity)

    assert np.linalg.norm(result['x_min'] - expected_minima_point) <= 0.7
    assert result['loss_min'] <= 0.1


def test_slurm_minimize_2params_with_constraint(verbosity, seed, loss_fun, constraint_fun):
    num_params = 2
    param_bounds = [[-5, 5] for _ in range(num_params)]
    expected_minima_point = np.ones(num_params)
    num_workers = 10
    num_iters = 30

    result = slurm_minimize(loss_fun=loss_fun, constraint_fun=constraint_fun,
                            param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                            cluster='local-map', verbosity=verbosity)

    assert np.linalg.norm(result['x_min'] - expected_minima_point) <= 0.5
    assert result['loss_min'] <= 0.05


def test_slurm_minimize_2params_with_constraint_from_init_points(verbosity, seed, loss_fun, constraint_fun):
    num_params = 2
    param_bounds = [[-5, 5] for _ in range(num_params)]
    expected_minima_point = np.ones(num_params)
    num_workers = 10
    num_iters = 40

    init_points = [np.array([-1, -1]) + np.random.rand(2) for _ in range(num_workers)]
    result = slurm_minimize(loss_fun=loss_fun, constraint_fun=constraint_fun, init_points=init_points,
                            param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                            cluster='local-map', verbosity=verbosity)

    assert np.linalg.norm(result['x_min'] - expected_minima_point) <= 0.5
    assert result['loss_min'] <= 0.05


def test_slurm_minimize_2params_with_constraint_from_illegal_init_points(verbosity, seed, loss_fun, constraint_fun):
    num_params = 2
    param_bounds = [[-5, 5] for _ in range(num_params)]
    num_workers = 10
    num_iters = 30

    with pytest.raises(ValueError):
        init_points = [np.array([-4, -4]) + 0.1 * np.random.rand(2) for _ in range(num_workers)]
        slurm_minimize(loss_fun=loss_fun, constraint_fun=constraint_fun, init_points=init_points,
                       param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                       cluster='local-map', verbosity=verbosity)


def test_slurm_minimize_2params_with_restart(work_dir, verbosity, seed, loss_fun):
    num_params = 2
    param_bounds = [[-5, 5] for _ in range(num_params)]
    num_workers = 5
    num_iters = 20

    # run and save restart
    res_1 = slurm_minimize(loss_fun=loss_fun,
                           param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                           cluster='local-map', verbosity=verbosity,
                           work_dir=work_dir, save_restart=True, load_restart=False)
    assert res_1['slurm_pool'].num_calls == num_iters
    assert len(res_1['slurm_pool'].points_history) == num_iters * num_workers
    assert res_1['ini_iter'] == num_iters
    # the version stamp is added to the pickled copy only, not the returned status dict
    assert '_slurmcmc_version' not in res_1
    # the atomic-write temp file is cleaned up
    assert not os.path.exists(work_dir + '/opt_restart.pkl.tmp')

    # run again from previous restart
    res_2 = slurm_minimize(loss_fun=loss_fun,
                           param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                           verbosity=verbosity,
                           work_dir=work_dir, save_restart=True, load_restart=True)
    assert res_2['slurm_pool'].num_calls == 2 * num_iters
    assert len(res_2['slurm_pool'].points_history) == 2 * num_iters * num_workers
    assert res_2['ini_iter'] == 2 * num_iters


def test_slurm_minimize_2params_with_log_file(work_dir, verbosity, seed, loss_fun):
    num_params = 2
    param_bounds = [[-5, 5] for _ in range(num_params)]
    num_workers = 4
    num_iters = 3

    slurm_minimize(loss_fun=loss_fun,
                   param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                   cluster='local-map',
                   verbosity=verbosity,
                   work_dir=work_dir, log_file='log_file.txt')

    assert os.path.isfile(work_dir + '/log_file.txt'), 'log_file was not created.'


def test_slurm_minimize_2params_with_constraint_and_with_extra_arg_fail(verbosity, seed, loss_fun_with_extra_arg,
                                                                        constraint_fun_with_extra_arg):
    num_params = 2
    param_bounds = [[-5, 5] for _ in range(num_params)]
    num_workers = 10
    num_iters = 30

    with pytest.raises(TypeError):
        slurm_minimize(loss_fun=loss_fun_with_extra_arg, constraint_fun=constraint_fun_with_extra_arg,
                       extra_arg=None,  # extra_arg not supplied and therefore should fail
                       param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                       cluster='local-map',
                       verbosity=verbosity)


def test_slurm_minimize_2params_with_constraint_and_with_extra_arg(verbosity, seed, loss_fun_with_extra_arg,
                                                                   constraint_fun_with_extra_arg):
    num_params = 2
    param_bounds = [[-5, 5] for _ in range(num_params)]
    expected_minima_point = np.ones(num_params)
    num_workers = 10
    num_iters = 30

    result = slurm_minimize(loss_fun=loss_fun_with_extra_arg, constraint_fun=constraint_fun_with_extra_arg,
                            extra_arg='sunny',
                            param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                            cluster='local-map', verbosity=verbosity)
    assert np.linalg.norm(result['x_min'] - expected_minima_point) <= 0.5
    assert result['loss_min'] <= 0.05


def test_slurm_minimize_1param_botorch(verbosity, seed, loss_fun_1d):
    num_params = 1
    param_bounds = [[-5, 5] for _ in range(num_params)]
    expected_minima_point = np.ones(num_params)
    num_workers = 5
    num_iters = 8

    result = slurm_minimize(loss_fun=loss_fun_1d,
                            param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                            optimizer_package='botorch', cluster='local-map',
                            verbosity=verbosity)

    assert np.linalg.norm(result['x_min'] - expected_minima_point) <= 0.02
    assert result['loss_min'] <= 1e-3


def test_slurm_minimize_1param_botorch_with_restart(work_dir, verbosity, seed, loss_fun_1d):
    num_params = 1
    param_bounds = [[-5, 5] for _ in range(num_params)]
    num_workers = 5
    num_iters = 4

    # run and save restart
    res_1 = slurm_minimize(loss_fun=loss_fun_1d,
                           param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                           optimizer_package='botorch', verbosity=verbosity, cluster='local-map',
                           work_dir=work_dir, save_restart=True, load_restart=False)
    assert res_1['slurm_pool'].num_calls == num_iters
    assert len(res_1['slurm_pool'].points_history) == num_iters * num_workers
    assert res_1['ini_iter'] == num_iters

    # run again from previous restart
    res_2 = slurm_minimize(loss_fun=loss_fun_1d,
                           param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                           verbosity=verbosity,
                           work_dir=work_dir, save_restart=True, load_restart=True)
    assert res_2['slurm_pool'].num_calls == 2 * num_iters
    assert len(res_2['slurm_pool'].points_history) == 2 * num_iters * num_workers
    assert res_2['ini_iter'] == 2 * num_iters
    assert res_2['loss_min'] <= 1e-3


def test_local_remote_slurm_minimize_1param(work_dir, verbosity, seed, loss_fun_1d):
    num_params = 1
    param_bounds = [[-5, 5] for _ in range(num_params)]
    expected_minima_point = np.ones(num_params)
    num_workers = 5
    num_iters = 10

    job = slurm_minimize(loss_fun=loss_fun_1d,
                         param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                         work_dir=work_dir, cluster='local-map', verbosity=verbosity,
                         remote=True, remote_cluster='local')

    result = job.result()
    assert np.linalg.norm(result['x_min'] - expected_minima_point) <= 0.03
    assert result['loss_min'] <= 1e-3


@pytest.mark.skipif(not is_slurm_cluster(), reason="This test only runs on a Slurm cluster")
def test_slurm_remote_slurm_minimize_1param(work_dir, verbosity, seed, loss_fun_1d):
    num_params = 1
    param_bounds = [[-5, 5] for _ in range(num_params)]
    expected_minima_point = np.ones(num_params)
    num_workers = 5
    num_iters = 10

    job = slurm_minimize(loss_fun=loss_fun_1d,
                         param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                         work_dir=work_dir, cluster='local-map', verbosity=verbosity,
                         remote=True, remote_cluster='slurm', remote_submitit_kwargs=submitit_kwargs)

    result = job.result()
    assert np.linalg.norm(result['x_min'] - expected_minima_point) <= 0.02
    assert result['loss_min'] <= 1e-3


def test_slurm_minimize_init_points_informs_optimizer(verbosity, seed, loss_fun):
    """init_points must be told to the nevergrad optimizer so iter 0 history is used."""
    num_params = 2
    param_bounds = [[-5, 5]] * num_params
    num_workers = 5
    # Provide init_points very close to the optimum
    init_points = [np.array([1.0, 1.0]) + 0.01 * np.random.randn(2) for _ in range(num_workers)]

    result = slurm_minimize(loss_fun=loss_fun, init_points=init_points,
                            param_bounds=param_bounds, num_workers=num_workers, num_iters=5,
                            cluster='local-map', verbosity=verbosity)
    # With init_points near the optimum and the optimizer knowing about them,
    # loss_min after just a few iters should be very small
    assert result['loss_min'] <= 0.5
    # per-iteration bookkeeping covers all iterations, including iter 0 with init_points
    assert len(result['candidates_ask_time_per_iter']) == 5
    assert result['num_workers_per_iter'] == [num_workers] * 5


def test_slurm_minimize_restart_state_preserved(work_dir, verbosity, seed, loss_fun_1d):
    """Restarting from a checkpoint must preserve the full evaluation history."""
    num_params = 1
    param_bounds = [[-5, 5]]
    num_workers = 5
    num_iters = 4

    res_1 = slurm_minimize(loss_fun=loss_fun_1d,
                           param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                           cluster='local-map', verbosity=verbosity,
                           work_dir=work_dir, save_restart=True)
    num_pts_after_first = len(res_1['slurm_pool'].points_history)

    res_2 = slurm_minimize(loss_fun=loss_fun_1d,
                           param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                           verbosity=verbosity,
                           work_dir=work_dir, load_restart=True)
    assert len(res_2['slurm_pool'].points_history) == num_pts_after_first + num_iters * num_workers
    assert res_2['loss_min'] <= res_1['loss_min'] + 1e-9  # monotonically non-increasing


def test_slurm_minimize_botorch_with_init_points(verbosity, seed, loss_fun_1d):
    """Regression test: optimizer_package='botorch' with init_points used to raise
    UnboundLocalError on 'instrum' (a nevergrad-only object built unconditionally at iter 0)."""
    num_iters = 3
    init_points = [np.array([0.5]), np.array([1.5]), np.array([2.0])]

    result = slurm_minimize(loss_fun=loss_fun_1d, init_points=init_points,
                            param_bounds=[[-5, 5]], num_workers=3, num_iters=num_iters,
                            optimizer_package='botorch',
                            botorch_kwargs={'num_restarts': 3, 'raw_samples': 20},
                            cluster='local-map', verbosity=verbosity)

    assert result['loss_min'] <= 0.3
    assert len(result['candidates_ask_time_per_iter']) == num_iters


def test_slurm_minimize_all_failed_iteration_raises(verbosity, seed):
    """If every evaluation in an iteration fails, a clear RuntimeError is raised
    (instead of numpy's opaque 'All-NaN slice encountered')."""
    def always_nan(x):
        return np.nan

    with pytest.raises(RuntimeError, match='evaluations failed'):
        slurm_minimize(loss_fun=always_nan, param_bounds=[[-5, 5]], num_workers=3, num_iters=1,
                       cluster='local-map', verbosity=verbosity)


def test_botorch_optimizer_num_best_points(verbosity, seed, loss_fun_1d):
    """num_best_points trims the GP training set to the N best evaluations when exceeded."""
    num_params = 1
    param_bounds = [[-5, 5]]
    num_workers = 3
    num_iters = 6
    num_best_points = 5  # fewer than total evaluations (6 iters × 3 workers = 18 pts)

    result = slurm_minimize(loss_fun=loss_fun_1d,
                            param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                            optimizer_package='botorch',
                            botorch_kwargs={'num_best_points': num_best_points,
                                           'num_restarts': 3, 'raw_samples': 20},
                            cluster='local-map', verbosity=verbosity)

    # Trimming should not break convergence — result still meaningful
    assert result['loss_min'] < 1.0
    total_pts = num_iters * num_workers
    assert len(result['slurm_pool'].points_history) == total_pts


def test_botorch_optimizer_num_best_points_no_trim_when_below_cap(verbosity, seed, loss_fun_1d):
    """When total evaluations <= num_best_points, no trimming occurs and all points are used."""
    num_params = 1
    param_bounds = [[-5, 5]]
    num_workers = 3
    num_iters = 2
    num_best_points = 100  # much larger than total evaluations (2 × 3 = 6 pts)

    result = slurm_minimize(loss_fun=loss_fun_1d,
                            param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
                            optimizer_package='botorch',
                            botorch_kwargs={'num_best_points': num_best_points,
                                           'num_restarts': 3, 'raw_samples': 20},
                            cluster='local-map', verbosity=verbosity)

    total_pts = num_iters * num_workers
    assert len(result['slurm_pool'].points_history) == total_pts  # all points kept

