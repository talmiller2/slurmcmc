import os
import time

import numpy as np
import pytest

from slurmcmc.general_utils import point_to_tuple
from slurmcmc.slurm_utils import SlurmPool


@pytest.fixture()
def fun_with_extra_arg():
    def _fun_with_extra_arg(x, weather):
        if weather == 'sunny':
            return x ** 2
        else:
            return x + 30

    return _fun_with_extra_arg


@pytest.fixture()
def fun_that_writes_file():
    def _fun_that_writes_file(x):
        with open('test_file.txt', 'a') as log_file:
            print(x, file=log_file)
        return x[0] ** 2 + x[1] ** 2

    return _fun_that_writes_file


@pytest.fixture()
def fun_that_sleeps():
    def _fun_that_sleeps(x):
        time.sleep(5)
        return x ** 2

    return _fun_that_sleeps


def test_slurmpool_localmap(verbosity):
    slurm_pool = SlurmPool(dim_input=1, dim_output=1, cluster='local-map', verbosity=verbosity)
    fun = lambda x: x ** 2
    points = [2, 3, 4]
    res_expected = [fun(point) for point in points]
    res = slurm_pool.map(fun, points)
    assert res == res_expected
    assert slurm_pool.num_calls == 1

    # check inconsistent dimension of input/output cause error
    fun_inconsistent_dim = lambda x: [x ** 2, x ** 3]
    with pytest.raises(ValueError):
        _ = slurm_pool.map(fun_inconsistent_dim, points)
    points_inconsistent_dim = [[2, 2], [3, 3]]
    with pytest.raises(ValueError):
        _ = slurm_pool.map(fun, points_inconsistent_dim)


def test_slurmpool_localmap_history(verbosity):
    slurm_pool = SlurmPool(dim_input=1, dim_output=1, cluster='local-map', verbosity=verbosity)
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
    slurm_pool = SlurmPool(dim_input=1, dim_output=1, cluster='local-map', verbosity=verbosity)
    fun = lambda x: x ** 2
    fun_that_fails = lambda x: None
    points_1 = [2, 3, 4]
    res_1 = slurm_pool.map(fun, points_1)
    points_2 = [5, 6, 7]
    res_2 = slurm_pool.map(fun_that_fails, points_2)
    all_points = points_1 + points_2
    all_res = res_1 + res_2
    assert slurm_pool.num_evaluated_points == len(all_points)
    np.testing.assert_array_equal(slurm_pool.points_history, np.array(all_points).reshape(-1, 1))
    np.testing.assert_array_equal(slurm_pool.values_history, np.array(all_res).reshape(-1, 1))
    np.testing.assert_array_equal(slurm_pool.inds_success_points, [0, 1, 2])
    np.testing.assert_array_equal(slurm_pool.inds_failed_points, [3, 4, 5])


def test_slurmpool_localmap_2params(verbosity):
    slurm_pool = SlurmPool(dim_input=2, dim_output=1, cluster='local-map', verbosity=verbosity)
    fun = lambda x: x[0] ** 2 + x[1] ** 2
    points = [[2, 3], [3, 4], [4, 5]]
    res_expected = [fun(point) for point in points]
    res = slurm_pool.map(fun, points)
    assert res == res_expected


def test_slurmpool_localmap_history_2params(verbosity):
    slurm_pool = SlurmPool(dim_input=2, dim_output=1, cluster='local-map', verbosity=verbosity)
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
    slurm_pool = SlurmPool(dim_input=1, dim_output=1, cluster='local-map', verbosity=verbosity, budget=2)
    fun = lambda x: x ** 2
    points = [2, 3, 4, 5, 6]
    res_expected = [fun(point) for point in points]
    res = slurm_pool.map(fun, points)
    assert res == res_expected
    assert slurm_pool.num_calls == 3


def test_slurmpool_local(work_dir, verbosity):
    slurm_pool = SlurmPool(dim_input=1, dim_output=1, work_dir=work_dir, cluster='local', verbosity=verbosity)
    fun = lambda x: x ** 2
    points_1 = [2, 3, 4]
    res_expected_1 = [fun(point) for point in points_1]
    res_1 = slurm_pool.map(fun, points_1)
    assert res_1 == res_expected_1

    points_2 = [5, 6, 7]
    res_expected_2 = [fun(point) for point in points_2]
    res_2 = slurm_pool.map(fun, points_2)
    assert res_2 == res_expected_2

    # test evaluated_points_set and point_loc_dict features of slurm_pool
    assert len(slurm_pool.evaluated_points_set) == len(points_1) + len(points_2)
    assert len(slurm_pool.point_loc_dict) == len(points_1) + len(points_2)
    point_first = slurm_pool.points_history[0]
    assert slurm_pool.point_loc_dict[point_to_tuple(point_first)] == (0, 0)
    point_last = slurm_pool.points_history[-1]
    assert slurm_pool.point_loc_dict[point_to_tuple(point_last)] == (1, len(points_2) - 1)


def test_slurmpool_local_2params(work_dir, verbosity):
    slurm_pool = SlurmPool(work_dir=work_dir, dim_input=2, dim_output=1, cluster='local', verbosity=verbosity)
    fun = lambda x: x[0] ** 2 + x[1] ** 2
    points = [[2, 3], [3, 4], [4, 5]]
    res_expected = [fun(point) for point in points]
    res = slurm_pool.map(fun, points)
    assert res == res_expected


def test_slurmpool_local_2params_with_log_file(work_dir, verbosity):
    slurm_pool = SlurmPool(work_dir=work_dir, dim_input=2, dim_output=1, cluster='local', verbosity=verbosity,
                           log_file='log_file.txt')
    fun = lambda x: x[0] ** 2 + x[1] ** 2
    points = [[2, 3], [3, 4], [4, 5]]
    slurm_pool.map(fun, points)


def test_slurmpool_local_2params_check_query_dir(work_dir, verbosity, fun_that_writes_file):
    slurm_pool = SlurmPool(work_dir=work_dir, dim_input=2, dim_output=1, cluster='local', verbosity=verbosity)
    points = [[2, 3], [3, 4], [4, 5]]
    slurm_pool.map(fun_that_writes_file, points)

    for i in range(len(points)):
        assert os.path.isfile(os.path.join(work_dir, '0', str(i), 'test_file.txt')), \
            'test_file was not created in the query dir of each point.'


def test_slurmpool_fail_on_existing_work_dir(work_dir, verbosity):
    os.makedirs(os.path.join(work_dir, '1'), exist_ok=True)
    with pytest.raises(ValueError):
        SlurmPool(dim_input=1, dim_output=1, work_dir=work_dir, cluster='local', verbosity=verbosity)


def test_slurmpool_localmap_2params_3outputs(verbosity):
    slurm_pool = SlurmPool(dim_input=2, dim_output=3, cluster='local-map', verbosity=verbosity)
    fun = lambda x: [x[0] ** 2 + x[1] ** 2, 10 * x[0], 10 * x[1]]
    points = [[2, 3], [3, 4], [4, 5]]
    res_expected = [fun(point) for point in points]
    res = slurm_pool.map(fun, points)
    assert res == res_expected


def test_slurmpool_localmap_2params_3outputs_history_with_failed_points(verbosity):
    slurm_pool = SlurmPool(dim_input=2, dim_output=3, cluster='local-map', verbosity=verbosity)
    fun = lambda x: [2 * x[0], 3 * x[0], 4 * x[1]]
    fun_that_partially_fails_1 = lambda x: [2 * x[0], None, 4 * x[1]]
    fun_that_partially_fails_2 = lambda x: [2 * x[0], 3 * x[0], np.nan]
    points = [[2, 3], [3, 4], [4, 5]]
    res_1 = slurm_pool.map(fun, points)
    res_2 = slurm_pool.map(fun_that_partially_fails_1, points)
    res_3 = slurm_pool.map(fun_that_partially_fails_2, points)
    all_points = points + points + points
    all_res = res_1 + res_2 + res_3
    assert slurm_pool.num_evaluated_points == len(all_points)
    np.testing.assert_array_equal(slurm_pool.points_history, np.array(all_points))
    np.testing.assert_array_equal(slurm_pool.inds_success_points, [0, 1, 2])
    np.testing.assert_array_equal(slurm_pool.inds_failed_points, [3, 4, 5, 6, 7, 8])
    assert slurm_pool.values_history.shape == (len(all_points), 3)


def test_slurmpool_localmap_with_extra_arg(verbosity, fun_with_extra_arg):
    for weather in ['sunny', 'rainy', None]:
        slurm_pool = SlurmPool(dim_input=1, dim_output=1, cluster='local-map', verbosity=verbosity, extra_arg=weather)
        points = [1, 2, 3]
        if weather is None:  # should fail
            with pytest.raises(TypeError):
                slurm_pool.map(fun_with_extra_arg, points)
        else:
            res_expected = [fun_with_extra_arg(point, weather) for point in points]
            res = slurm_pool.map(fun_with_extra_arg, points)
            assert res == res_expected


def test_slurmpool_local_with_extra_arg(work_dir, verbosity, fun_with_extra_arg):
    weather = 'sunny'
    slurm_pool = SlurmPool(work_dir=work_dir, dim_input=1, dim_output=1, cluster='local', verbosity=verbosity,
                           extra_arg=weather)
    points = [1, 2, 3]
    res_expected = [fun_with_extra_arg(point, weather) for point in points]
    res = slurm_pool.map(fun_with_extra_arg, points)
    assert res == res_expected
    assert os.path.isfile(os.path.join(work_dir, '0/extra_arg.pkl')), 'extra_arg.pkl does not appear.'


def test_slurmpool_local_fail_on_slurmpool_timeout(work_dir, verbosity, fun_that_sleeps):
    job_fail_value = np.nan
    slurm_pool = SlurmPool(work_dir=work_dir, dim_input=1, dim_output=1, cluster='local', verbosity=verbosity,
                           check_output_timeout_minutes=2 / 60.0, check_output_interval_seconds=0.1,
                           job_fail_value=job_fail_value)
    points = [2, 4]
    res = slurm_pool.map(fun_that_sleeps, points)
    res_expected_fail = [job_fail_value for _ in points]
    assert res == res_expected_fail


def test_slurmpool_localmap_record_history_false(verbosity):
    slurm_pool = SlurmPool(dim_input=1, dim_output=1, cluster='local-map', verbosity=verbosity,
                           record_history=False)
    fun = lambda x: x ** 2
    points = [2, 3, 4]
    res = slurm_pool.map(fun, points)
    for attr in ('points_history', 'values_history', 'evaluated_points_set',
                 'inds_success_points', 'inds_failed_points', 'point_loc_dict'):
        assert not hasattr(slurm_pool, attr), f"slurm_pool should not have '{attr}' when record_history=False"


def test_slurmpool_split_points_edge_cases(verbosity):
    """split_points handles exact multiples, single-chunk, and budget=1 correctly."""
    pool = SlurmPool(dim_input=1, dim_output=1, cluster='local-map', verbosity=verbosity)
    points = [1, 2, 3, 4]
    assert pool.split_points(points, budget=4) == [[1, 2, 3, 4]]   # exact fit → 1 chunk
    assert pool.split_points(points, budget=2) == [[1, 2], [3, 4]]  # even split
    assert pool.split_points(points, budget=3) == [[1, 2, 3], [4]]  # remainder chunk
    assert pool.split_points(points, budget=1) == [[1], [2], [3], [4]]  # budget=1


def test_slurmpool_local_relative_work_dir(work_dir, verbosity):
    """
    Regression test: a relative work_dir used to break cluster='local'/'slurm' with
    multiple points, because send_and_receive_jobs chdirs into per-point directories
    and relative paths then resolved against the wrong base (nested directory garbage
    followed by FileNotFoundError). SlurmPool now stores work_dir as an absolute path.
    """
    # cwd is a fresh tmp dir (work_dir fixture chdirs there); pass a *relative* work_dir
    slurm_pool = SlurmPool(work_dir='relative_work_dir', dim_input=1, dim_output=1,
                           cluster='local', verbosity=verbosity)
    fun = lambda x: x ** 2
    res = slurm_pool.map(fun, [2, 3, 4])
    assert res == [4, 9, 16]
    # per-point dirs are laid out flat, with no recursively nested work_dir inside them
    assert os.path.isdir(os.path.join('relative_work_dir', '0', '2'))
    assert not os.path.isdir(os.path.join('relative_work_dir', '0', '0', 'relative_work_dir'))


def test_slurmpool_localmap_empty_points(verbosity):
    """map() with an empty point list returns an empty list without errors."""
    pool = SlurmPool(dim_input=1, dim_output=1, cluster='local-map', verbosity=verbosity)
    res = pool.map(lambda x: x ** 2, [])
    assert res == []
    assert pool.num_calls == 0
    assert pool.num_evaluated_points == 0
