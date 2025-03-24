import os

import pytest

from slurmcmc.general_utils import delete_directory
from slurmcmc.import_utils import import_function_from_module
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


@pytest.fixture(scope="module")
def fun_dict():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return {
        'module_dir': os.path.join(base_dir, 'example_module_dir'),
        'module_name': 'example_module',
        'function_name': 'example_fun'
    }


@pytest.fixture(scope="module")
def fun(fun_dict):
    return import_function_from_module(fun_dict['module_dir'],
                                       fun_dict['module_name'],
                                       fun_dict['function_name'])


@pytest.fixture(scope="module")
def fun_with_extra_arg_dict():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return {
        'module_dir': os.path.join(base_dir, 'example_module_dir'),
        'module_name': 'example_module',
        'function_name': 'example_fun_with_extra_arg'
    }


@pytest.fixture(scope="module")
def fun_with_extra_arg(fun_with_extra_arg_dict):
    return import_function_from_module(fun_with_extra_arg_dict['module_dir'],
                                       fun_with_extra_arg_dict['module_name'],
                                       fun_with_extra_arg_dict['function_name'])


@pytest.fixture(scope="module")
def verbosity():
    return 1


def test_slurmpool_local_imported_fun(work_dir, fun_dict, fun, verbosity):
    """
    An imported fun might not pass the submitit pickling pipeline, so instead pass fun_dict to defer the import until
    evaluations are needed.
    """
    points = [2, 3, 4]
    res_expected = [fun(point) for point in points]
    slurm_pool = SlurmPool(work_dir, cluster='local', verbosity=verbosity)
    res = slurm_pool.map(fun_dict, points)
    assert res == res_expected


def test_slurmpool_local_imported_fun_with_extra_arg(work_dir, fun_with_extra_arg_dict, fun_with_extra_arg, verbosity):
    """
    Same but with a function that has extra_arg.
    """
    setup_dict = {'weather': 'sunny'}
    points = [2, 3, 4]
    res_expected = [fun_with_extra_arg(point, setup_dict) for point in points]
    slurm_pool = SlurmPool(work_dir, cluster='local', verbosity=verbosity, extra_arg=setup_dict)
    res = slurm_pool.map(fun_with_extra_arg_dict, points)
    assert res == res_expected


def test_slurmpool_local_imported_fun_fail(work_dir, fun_with_extra_arg, verbosity):
    """
    Using a function that is imported from a different directory, should fail when running with
    cluster='local' or 'slurm' because does not pickle properly.
    """
    setup_dict = {'weather': 'sunny'}
    points = [2, 3, 4]
    slurm_pool = SlurmPool(work_dir, cluster='local', verbosity=verbosity, extra_arg=setup_dict)
    with pytest.raises(Exception):
        slurm_pool.map(fun_with_extra_arg, points)
