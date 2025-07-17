import os

import numpy as np
import pytest
from scipy.optimize import rosen

from slurmcmc.general_utils import delete_directory, load_restart_file
from slurmcmc.mcmc import slurm_mcmc, calculate_unique_points_weights
from slurmcmc.slurm_utils import is_slurm_cluster
from tests.submitit_defaults import submitit_kwargs


@pytest.fixture()
def work_dir(request):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.join(base_dir, f'test_work_dir_{request.node.name}')
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    yield work_dir
    os.chdir(base_dir)
    delete_directory(work_dir)


@pytest.fixture
def seed():
    np.random.seed(0)


@pytest.fixture()
def verbosity():
    return 1


def log_prob_fun(x):
    return -rosen(x)


def log_prob_fun_with_extra_arg(x, extra_arg):
    if extra_arg == 'sunny':
        return -rosen(x)
    else:
        return None


def test_slurm_mcmc(verbosity, seed):
    num_params = 2
    num_walkers = 10
    num_iters = 3
    minima = np.array([1, 1])
    init_points = np.array([minima for _ in range(num_walkers)]) + 0.5 * np.random.randn(num_walkers, num_params)
    status = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=init_points, num_iters=num_iters,
                        verbosity=verbosity, slurm_vebosity=verbosity,
                        cluster='local-map')
    samples = status['sampler'].get_chain(flat=True)
    samples = np.vstack([init_points, samples])  # init_points are not inherently included in the mcmc sampler samples
    num_calculated_points = num_walkers * (num_iters + 1)
    np.testing.assert_equal(samples.shape, (num_calculated_points, num_params))
    np.testing.assert_equal(status['slurm_pool'].points_history.shape, (num_calculated_points, num_params))
    assert status['slurm_pool'].num_calls == 7


def test_slurm_mcmc_local(work_dir, verbosity, seed):
    num_params = 2
    num_walkers = 10
    num_iters = 3
    minima = np.array([1, 1])
    init_points = np.array([minima for _ in range(num_walkers)]) + 0.5 * np.random.randn(num_walkers, num_params)
    status = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=init_points, num_iters=num_iters,
                        verbosity=verbosity, slurm_vebosity=verbosity,
                        work_dir=work_dir, cluster='local')


def test_slurm_mcmc_with_budget(verbosity, seed):
    num_params = 2
    num_walkers = 10
    num_iters = 3
    minima = np.array([1, 1])
    init_points = np.array([minima for _ in range(num_walkers)]) + 0.5 * np.random.randn(num_walkers, num_params)
    status = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=init_points, num_iters=num_iters,
                        verbosity=verbosity, slurm_vebosity=verbosity,
                        cluster='local-map', budget=5)
    num_calculated_points = num_walkers * (num_iters + 1)
    np.testing.assert_equal(status['slurm_pool'].points_history.shape, (num_calculated_points, num_params))
    assert status['slurm_pool'].num_calls == 8


def test_slurm_mcmc_with_log_file(work_dir, verbosity, seed):
    num_params = 2
    num_walkers = 10
    num_iters = 3
    minima = np.array([1, 1])
    init_points = np.array([minima for _ in range(num_walkers)]) + 0.5 * np.random.randn(num_walkers, num_params)
    status = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=init_points, num_iters=num_iters,
                        verbosity=verbosity, slurm_vebosity=verbosity,
                        cluster='local-map',
                        work_dir=work_dir, log_file='log_file.txt')

    assert os.path.isfile(os.path.join(work_dir, 'log_file.txt')), 'log_file was not created.'


def log_prob_fun_global(x):
    # defined globally for test_slurm_mcmc_with_restart
    # because the local version that is defined as a fixture fails to be pickled
    return -rosen(x)


def test_slurm_mcmc_with_restart_file(work_dir, verbosity, seed):
    num_params = 2
    num_walkers = 10
    num_iters = 3

    num_slurm_call_init = 1
    num_slurm_call_mcmc = 2 * num_iters
    num_points_calc_init = num_walkers
    num_points_calc_mcmc = num_walkers * num_iters

    minima = np.array([1, 1])
    init_points = np.array([minima for _ in range(num_walkers)]) + 0.5 * np.random.randn(num_walkers, num_params)

    status_1 = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=init_points, num_iters=num_iters,
                          verbosity=verbosity, slurm_vebosity=verbosity,
                          cluster='local-map',
                          work_dir=work_dir, save_restart=True, load_restart=False)

    total_num_slurm_call = num_slurm_call_init + num_slurm_call_mcmc
    total_num_points_calc = num_points_calc_init + num_points_calc_mcmc
    assert status_1['slurm_pool'].num_calls == total_num_slurm_call
    assert len(status_1['slurm_pool'].points_history) == total_num_points_calc
    restart_1 = load_restart_file(work_dir, restart_file='mcmc_restart.pkl')
    assert restart_1['ini_iter'] == num_iters

    status_2 = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=init_points, num_iters=num_iters,
                          verbosity=verbosity, slurm_vebosity=verbosity,
                          cluster='local-map',
                          work_dir=work_dir, save_restart=True, load_restart=True)

    total_num_slurm_call = num_slurm_call_init + 2 * num_slurm_call_mcmc
    total_num_points_calc = num_points_calc_init + 2 * num_points_calc_mcmc
    assert status_2['slurm_pool'].num_calls == total_num_slurm_call
    assert len(status_2['slurm_pool'].points_history) == total_num_points_calc
    restart_2 = load_restart_file(work_dir, restart_file='mcmc_restart.pkl')
    assert restart_2['ini_iter'] == 2 * num_iters


def test_slurm_mcmc_with_status_restart(work_dir, verbosity, seed):
    num_params = 2
    num_walkers = 10
    num_iters = 3

    num_slurm_call_init = 1
    num_slurm_call_mcmc = 2 * num_iters
    num_points_calc_init = num_walkers
    num_points_calc_mcmc = num_walkers * num_iters

    minima = np.array([1, 1])
    init_points = np.array([minima for _ in range(num_walkers)]) + 0.5 * np.random.randn(num_walkers, num_params)

    status_1 = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=init_points, num_iters=num_iters,
                          verbosity=verbosity, slurm_vebosity=verbosity,
                          cluster='local-map',
                          work_dir=work_dir)

    total_num_slurm_call = num_slurm_call_init + num_slurm_call_mcmc
    total_num_points_calc = num_points_calc_init + num_points_calc_mcmc
    assert status_1['slurm_pool'].num_calls == total_num_slurm_call
    assert len(status_1['slurm_pool'].points_history) == total_num_points_calc

    status_2 = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=init_points, num_iters=num_iters,
                          verbosity=verbosity, slurm_vebosity=verbosity,
                          cluster='local-map',
                          work_dir=work_dir, status_restart=status_1)

    total_num_slurm_call = num_slurm_call_init + 2 * num_slurm_call_mcmc
    total_num_points_calc = num_points_calc_init + 2 * num_points_calc_mcmc
    assert status_2['slurm_pool'].num_calls == total_num_slurm_call
    assert len(status_2['slurm_pool'].points_history) == total_num_points_calc


def test_slurm_mcmc_init_log_prob_fun_values(verbosity, seed):
    num_params = 2
    num_walkers = 10
    num_iters = 3
    minima = np.array([1, 1])
    init_points = np.array([minima for _ in range(num_walkers)]) + 0.5 * np.random.randn(num_walkers, num_params)
    init_log_prob_fun_values = [log_prob_fun(point) for point in init_points]
    status = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=init_points, num_iters=num_iters,
                        init_log_prob_fun_values=init_log_prob_fun_values,
                        verbosity=verbosity, slurm_vebosity=verbosity,
                        cluster='local-map')
    samples = status['sampler'].get_chain(flat=True)
    num_calculated_points = num_walkers * num_iters  # the initial iteration was not calculated
    np.testing.assert_equal(samples.shape, (num_calculated_points, num_params))
    np.testing.assert_equal(status['slurm_pool'].points_history.shape, (num_calculated_points, num_params))
    assert status['slurm_pool'].num_calls == 6


def test_slurm_mcmc_with_extra_arg_localmap(verbosity, seed):
    # print('locals=', locals())

    num_params = 2
    num_walkers = 5
    num_iters = 2
    minima = np.array([1, 1])
    init_points = np.array([minima for _ in range(num_walkers)]) + 0.5 * np.random.randn(num_walkers, num_params)
    status = slurm_mcmc(log_prob_fun=log_prob_fun_with_extra_arg, init_points=init_points, num_iters=num_iters,
                        extra_arg='sunny',
                        verbosity=verbosity, slurm_vebosity=verbosity,
                        cluster='local-map')


def test_slurm_mcmc_with_extra_arg_local(work_dir, verbosity, seed):
    num_params = 2
    num_walkers = 5
    num_iters = 2
    minima = np.array([1, 1])
    init_points = np.array([minima for _ in range(num_walkers)]) + 0.5 * np.random.randn(num_walkers, num_params)
    status = slurm_mcmc(log_prob_fun=log_prob_fun_with_extra_arg, init_points=init_points, num_iters=num_iters,
                        extra_arg='sunny',
                        verbosity=verbosity, slurm_vebosity=verbosity,
                        work_dir=work_dir, cluster='local')
    assert os.path.isfile(os.path.join(work_dir, 'extra_arg.txt')), 'extra_arg.txt does not appear.'


def test_slurm_mcmc_init_log_prob_fun_values_with_extra_arg(verbosity, seed):
    num_params = 2
    num_walkers = 10
    num_iters = 3
    minima = np.array([1, 1])
    init_points = np.array([minima for _ in range(num_walkers)]) + 0.5 * np.random.randn(num_walkers, num_params)
    extra_arg = 'sunny'

    init_log_prob_fun_values = [log_prob_fun_with_extra_arg(point, extra_arg) for point in init_points]
    status = slurm_mcmc(log_prob_fun=log_prob_fun_with_extra_arg, init_points=init_points, num_iters=num_iters,
                        extra_arg=extra_arg,
                        init_log_prob_fun_values=init_log_prob_fun_values,
                        verbosity=verbosity, slurm_vebosity=verbosity,
                        cluster='local-map')


def test_local_remote_slurm_mcmc(work_dir, verbosity, seed):
    num_params = 2
    num_walkers = 10
    num_iters = 3
    minima = np.array([1, 1])
    init_points = np.array([minima for _ in range(num_walkers)]) + 0.5 * np.random.randn(num_walkers, num_params)

    job = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=init_points, num_iters=num_iters,
                     verbosity=verbosity, slurm_vebosity=verbosity,
                     work_dir=work_dir, cluster='local-map',
                     remote=True, remote_cluster='local')

    status = job.result()
    samples = status['sampler'].get_chain(flat=True)
    samples = np.vstack([init_points, samples])  # init_points are not inherently included in the mcmc sampler samples
    num_calculated_points = num_walkers * (num_iters + 1)
    np.testing.assert_equal(samples.shape, (num_calculated_points, num_params))
    np.testing.assert_equal(status['slurm_pool'].points_history.shape, (num_calculated_points, num_params))
    assert status['slurm_pool'].num_calls == 7


@pytest.mark.skipif(not is_slurm_cluster(), reason="This test only runs on a Slurm cluster")
def test_slurm_remote_slurm_mcmc(work_dir, verbosity, seed):
    num_params = 2
    num_walkers = 10
    num_iters = 3
    minima = np.array([1, 1])
    init_points = np.array([minima for _ in range(num_walkers)]) + 0.5 * np.random.randn(num_walkers, num_params)

    job = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=init_points, num_iters=num_iters,
                     verbosity=verbosity, slurm_vebosity=verbosity,
                     work_dir=work_dir, cluster='local-map',
                     remote=True, remote_cluster='slurm', remote_submitit_kwargs=submitit_kwargs)

    status = job.result()
    samples = status['sampler'].get_chain(flat=True)
    samples = np.vstack([init_points, samples])  # init_points are not inherently included in the mcmc sampler samples
    num_calculated_points = num_walkers * (num_iters + 1)
    np.testing.assert_equal(samples.shape, (num_calculated_points, num_params))
    np.testing.assert_equal(status['slurm_pool'].points_history.shape, (num_calculated_points, num_params))
    assert status['slurm_pool'].num_calls == 7


def test_calculate_unique_points_weights(verbosity):
    points = [[0, 0], [0, 1], [2, 0], [1, 3], [0, 1], [0, 0], [0, 0]]
    unique_points_set, points_weights_dict = calculate_unique_points_weights(points)
    unique_points_set_expected = {(0, 0), (0, 1), (1, 3), (2, 0)}
    points_weights_dict_expected = {(0, 0): 3, (0, 1): 2, (2, 0): 1, (1, 3): 1}
    assert unique_points_set == unique_points_set_expected, "incorrect unique_points_set"
    assert points_weights_dict == points_weights_dict_expected, "incorrect points_weights_dict"
