import os

import numpy as np
import pandas as pd
import pytest

from slurmcmc.general_utils import delete_directory, point_to_tuple
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


# def normalize_array(arr):
#     arr_float = np.array(arr)
#     for i, vi in enumerate(arr_float):
#         for j, vj in enumerate(vi):
#             if isinstance(vj, (int, float)) and not np.isnan(vj):
#                 arr_float[i, j] = float(vj)
#             elif np.isnan(vj):
#                 arr_float[i, j] = None
#     return arr_float

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
    # check values_history contains the right values, where ints and floats are considered the same
    df1 = pd.DataFrame(slurm_pool.values_history)
    df2 = pd.DataFrame(np.array(all_res))
    pd.testing.assert_frame_equal(df1, df2, check_dtype=False, check_exact=False)


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
    assert os.path.isfile(os.path.join(work_dir, '0/extra_arg.txt')), 'extra_arg.txt does not appear.'
