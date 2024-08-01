import os

import numpy as np
import pytest

from slurmcmc.general_utils import delete_directory
from slurmcmc.slurm_utils import SlurmPool


@pytest.fixture()
def work_dir(request):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.join(base_dir, f'test_work_dir_{request.node.name}')
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    yield work_dir
    os.chdir(base_dir)
    delete_directory(work_dir)


@pytest.fixture()
def verbosity():
    return 1


def fun_with_extra_arg(x, weather):
    if weather == 'sunny':
        return x ** 2
    else:
        return x + 30


def fun_that_writes_file(x):
    with open('test_file.txt', 'a') as log_file:
        print(x, file=log_file)
    return x[0] ** 2 + x[1] ** 2


def test_slurmpool_localmap(verbosity):
    slurm_pool = SlurmPool(cluster='local-map', verbosity=verbosity)
    fun = lambda x: x ** 2
    points = [2, 3, 4]
    res_expected = [fun(point) for point in points]
    res = slurm_pool.map(fun, points)
    assert res == res_expected
    assert slurm_pool.num_calls == 1


def test_slurmpool_localmap_history(verbosity):
    slurm_pool = SlurmPool(cluster='local-map', verbosity=verbosity)
    fun = lambda x: x ** 2
    points_1 = [2, 3, 4]
    res_1 = slurm_pool.map(fun, points_1)
    points_2 = [5, 6, 7]
    res_2 = slurm_pool.map(fun, points_2)
    points_history_expected = np.append(points_1, points_2).reshape(-1, 1)
    values_history_expected = np.append(res_1, res_2).reshape(-1, 1)
    np.testing.assert_array_equal(slurm_pool.points_history, points_history_expected)
    np.testing.assert_array_equal(slurm_pool.values_history, values_history_expected)


def test_slurmpool_localmap_history_with_failed_points(verbosity):
    slurm_pool = SlurmPool(cluster='local-map', verbosity=verbosity)
    fun = lambda x: x ** 2
    fun_that_fails = lambda x: None
    points_1 = [2, 3, 4]
    res_1 = slurm_pool.map(fun, points_1)
    points_2 = [5, 6, 7]
    res_2 = slurm_pool.map(fun_that_fails, points_2)

    np.testing.assert_array_equal(slurm_pool.points_history, np.array(points_1).reshape(-1, 1))
    np.testing.assert_array_equal(slurm_pool.values_history, np.array(res_1).reshape(-1, 1))
    assert all(r is None for r in res_2)  # Check that all elements in res_2 are None
    np.testing.assert_array_equal(slurm_pool.failed_points_history, np.array(points_2).reshape(-1, 1))


def test_slurmpool_localmap_2params(verbosity):
    slurm_pool = SlurmPool(cluster='local-map', verbosity=verbosity)
    fun = lambda x: x[0] ** 2 + x[1] ** 2
    points = [[2, 3], [3, 4], [4, 5]]
    res_expected = [fun(point) for point in points]
    res = slurm_pool.map(fun, points)
    assert res == res_expected


def test_slurmpool_localmap_history_2params(verbosity):
    slurm_pool = SlurmPool(cluster='local-map', verbosity=verbosity)
    fun = lambda x: x[0] ** 2 + x[1] ** 2
    points_1 = [[2, 3], [3, 4], [4, 5]]
    res_1 = slurm_pool.map(fun, points_1)
    points_2 = [[5, 6], [7, 8], [8, 9]]
    res_2 = slurm_pool.map(fun, points_2)
    points_history_expected = np.vstack([np.array(points_1), np.array(points_2)])
    values_history_expected = np.append(res_1, res_2).reshape(-1, 1)
    np.testing.assert_array_equal(slurm_pool.points_history, points_history_expected)
    np.testing.assert_array_equal(slurm_pool.values_history, values_history_expected)


def test_slurmpool_localmap_with_budget(verbosity):
    slurm_pool = SlurmPool(cluster='local-map', verbosity=verbosity, budget=2)
    fun = lambda x: x ** 2
    points = [2, 3, 4, 5, 6]
    res_expected = [fun(point) for point in points]
    res = slurm_pool.map(fun, points)
    assert res == res_expected
    assert slurm_pool.num_calls == 3


def test_slurmpool_local(work_dir, verbosity):
    slurm_pool = SlurmPool(work_dir, cluster='local', verbosity=verbosity)
    fun = lambda x: x ** 2
    points = [2, 3, 4]
    res_expected = [fun(point) for point in points]
    res = slurm_pool.map(fun, points)
    assert res == res_expected


def test_slurmpool_local_2params(work_dir, verbosity):
    slurm_pool = SlurmPool(work_dir, cluster='local', verbosity=verbosity)
    fun = lambda x: x[0] ** 2 + x[1] ** 2
    points = [[2, 3], [3, 4], [4, 5]]
    res_expected = [fun(point) for point in points]
    res = slurm_pool.map(fun, points)
    assert res == res_expected


def test_slurmpool_local_2params_with_log_file(work_dir, verbosity):
    slurm_pool = SlurmPool(work_dir, cluster='local', verbosity=verbosity, log_file='log_file.txt')
    fun = lambda x: x[0] ** 2 + x[1] ** 2
    points = [[2, 3], [3, 4], [4, 5]]
    slurm_pool.map(fun, points)


def test_slurmpool_local_2params_check_query_dir(work_dir, verbosity):
    slurm_pool = SlurmPool(work_dir, cluster='local', verbosity=verbosity)
    points = [[2, 3], [3, 4], [4, 5]]
    slurm_pool.map(fun_that_writes_file, points)

    for i in range(len(points)):
        assert os.path.isfile(os.path.join(work_dir, '0', str(i), 'test_file.txt')), \
            'test_file was not created in the query dir of each point.'


def test_slurmpool_fail_on_existing_work_dir(work_dir, verbosity):
    os.makedirs(os.path.join(work_dir, '0'), exist_ok=True)
    with pytest.raises(ValueError):
        SlurmPool(work_dir, cluster='local', verbosity=verbosity)


def test_slurmpool_localmap_2params_3outputs(verbosity):
    slurm_pool = SlurmPool(cluster='local-map', verbosity=verbosity)
    fun = lambda x: [x[0] ** 2 + x[1] ** 2, 10 * x[0], 10 * x[1]]
    points = [[2, 3], [3, 4], [4, 5]]
    res_expected = [fun(point) for point in points]
    res = slurm_pool.map(fun, points)
    assert res == res_expected


def test_slurmpool_localmap_2params_3outputs_history_with_failed_points(verbosity):
    slurm_pool = SlurmPool(cluster='local-map', verbosity=verbosity)
    fun = lambda x: [2 * x[0], 3 * x[0], 4 * x[1]]
    fun_that_partially_fails_1 = lambda x: [2 * x[0], None, 4 * x[1]]
    fun_that_partially_fails_2 = lambda x: [2 * x[0], 3 * x[0], np.nan]
    points = [[2, 3], [3, 4], [4, 5]]
    res = slurm_pool.map(fun, points)
    _ = slurm_pool.map(fun_that_partially_fails_1, points)
    _ = slurm_pool.map(fun_that_partially_fails_2, points)

    np.testing.assert_array_equal(slurm_pool.points_history, np.array(points))
    np.testing.assert_array_equal(slurm_pool.values_history, np.array(res).reshape(-1, 1))
    np.testing.assert_array_equal(slurm_pool.failed_points_history, np.vstack([points, points]))


def test_slurmpool_localmap_with_extra_arg(verbosity):
    for weather in ['sunny', 'rainy', None]:
        slurm_pool = SlurmPool(cluster='local-map', verbosity=verbosity, extra_arg=weather)
        points = [1, 2, 3]
        if weather is None:  # should fail
            with pytest.raises(TypeError):
                slurm_pool.map(fun_with_extra_arg, points)
        else:
            res_expected = [fun_with_extra_arg(point, weather) for point in points]
            res = slurm_pool.map(fun_with_extra_arg, points)
            assert res == res_expected


def test_slurmpool_local_with_extra_arg(work_dir, verbosity):
    weather = 'sunny'
    slurm_pool = SlurmPool(work_dir, cluster='local', verbosity=verbosity, extra_arg=weather)
    points = [1, 2, 3]
    res_expected = [fun_with_extra_arg(point, weather) for point in points]
    res = slurm_pool.map(fun_with_extra_arg, points)
    assert res == res_expected
    assert os.path.isfile(os.path.join(work_dir, '0/extra_arg.txt')), 'extra_arg.txt does not appear.'
