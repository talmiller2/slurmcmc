import os

import pytest

from slurmcmc.general_utils import delete_directory
from slurmcmc.import_utils import import_function_from_module, imported_fun
from slurmcmc.slurm_utils import SlurmPool


@pytest.fixture()
def work_dir(request):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.join(base_dir, f'test_work_dir_{request.node.name}')
    os.makedirs(work_dir, exist_ok=True)
    original_dir = os.getcwd()
    os.chdir(work_dir)
    yield work_dir
    os.chdir(original_dir)
    delete_directory(work_dir)


@pytest.fixture(scope="module")
def imported_fun_dict():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return {
        'module_dir': os.path.join(base_dir, 'test_module_dir'),
        'module_name': 'test_module',
        'function_name': 'fun_with_extra_arg'
    }


@pytest.fixture(scope="module")
def fun_with_extra_arg(imported_fun_dict):
    return import_function_from_module(imported_fun_dict['module_dir'],
                                       imported_fun_dict['module_name'],
                                       imported_fun_dict['function_name'])


@pytest.fixture(scope="module")
def verbosity():
    return 1


def test_slurmpool_local_imported_fun_method1(work_dir, imported_fun_dict, fun_with_extra_arg, verbosity):
    """
    use imported_fun to allow the function to pass the submitit pipeline,
    with the imported_fun_dict supplied as an extra_arg.
    """
    setup_dict = {'weather': 'sunny'}
    setup_dict.update(imported_fun_dict)
    points = [2, 3, 4]
    res_expected = [fun_with_extra_arg(point, setup_dict) for point in points]
    slurm_pool = SlurmPool(work_dir, cluster='local', verbosity=verbosity, extra_arg=setup_dict)
    res = slurm_pool.map(imported_fun, points)
    assert res == res_expected


def test_slurmpool_local_imported_fun_method2(work_dir, imported_fun_dict, fun_with_extra_arg, verbosity):
    """
    use imported_fun to allow the function to pass the submitit pipeline,
    with the imported_fun_dict supplied as the 'imported_fun_dict' element of extra_arg.
    """
    setup_dict = {'weather': 'sunny'}
    setup_dict['imported_fun_dict'] = imported_fun_dict
    points = [2, 3, 4]
    res_expected = [fun_with_extra_arg(point, setup_dict) for point in points]
    slurm_pool = SlurmPool(work_dir, cluster='local', verbosity=verbosity, extra_arg=setup_dict)
    res = slurm_pool.map(imported_fun, points)
    assert res == res_expected


def test_slurmpool_local_imported_fun_fail(work_dir, fun_with_extra_arg, verbosity):
    """
    using a function that is imported from a different directory, should fail when running with
    cluster='local' or 'slurm' because does not (cloud)pickle properly.
    """
    setup_dict = {'weather': 'sunny'}
    points = [2, 3, 4]
    slurm_pool = SlurmPool(work_dir, cluster='local', verbosity=verbosity, extra_arg=setup_dict)
    with pytest.raises(Exception):
        slurm_pool.map(fun_with_extra_arg, points)
