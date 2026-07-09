import os

import numpy as np
import pytest

from slurmcmc.import_utils import import_function_from_module
from slurmcmc.slurm_utils import SlurmPool


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


def test_slurmpool_local_imported_fun(work_dir, fun_dict, fun, verbosity):
    """
    An imported fun might not pass the submitit pickling pipeline, so instead pass fun_dict to defer the import until
    evaluations are needed.
    """
    points = [2, 3, 4]
    res_expected = [fun(point) for point in points]
    slurm_pool = SlurmPool(work_dir=work_dir, dim_input=1, dim_output=1, cluster='local', verbosity=verbosity)
    res = slurm_pool.map(fun_dict, points)
    assert res == res_expected


def test_slurmpool_local_imported_fun_with_extra_arg(work_dir, fun_with_extra_arg_dict, fun_with_extra_arg, verbosity):
    """
    Same but with a function that has extra_arg.
    """
    setup_dict = {'weather': 'sunny'}
    points = [2, 3, 4]
    res_expected = [fun_with_extra_arg(point, setup_dict) for point in points]
    slurm_pool = SlurmPool(work_dir=work_dir, dim_input=1, dim_output=1, cluster='local', verbosity=verbosity,
                           extra_arg=setup_dict)
    res = slurm_pool.map(fun_with_extra_arg_dict, points)
    assert res == res_expected


def test_deferred_import_function_caches_import(tmp_path, monkeypatch):
    """DeferredImportFunction imports the module once per instance and caches it;
    a pickled copy starts with an empty cache so remote jobs still get a fresh import.
    Uses a throwaway module (not example_module) because the deferred import reloads
    the module, which would invalidate function identity for the other tests here."""
    import pickle

    from slurmcmc import import_utils
    from slurmcmc.import_utils import deferred_import_function_wrapper

    (tmp_path / 'tiny_module_for_cache_test.py').write_text('def tiny_fun(x):\n    return x ** 2\n')
    fun_dict = {'module_dir': str(tmp_path),
                'module_name': 'tiny_module_for_cache_test',
                'function_name': 'tiny_fun'}

    calls = {'n': 0}
    real_import = import_utils.import_function_from_module

    def counting_import(*args, **kwargs):
        calls['n'] += 1
        return real_import(*args, **kwargs)

    monkeypatch.setattr(import_utils, 'import_function_from_module', counting_import)

    deferred = deferred_import_function_wrapper(fun_dict)
    assert deferred(2) == 4
    assert deferred(3) == 9
    assert calls['n'] == 1  # second call reused the cached function

    clone = pickle.loads(pickle.dumps(deferred))
    assert clone._cached_fun is None  # cache is not carried through pickling
    assert clone(2) == 4


def test_slurmpool_local_imported_fun_fail(work_dir, fun_with_extra_arg, verbosity):
    """
    Using a function that is imported from a different directory, should fail when running with
    cluster='local' or 'slurm' because does not pickle properly.
    """
    setup_dict = {'weather': 'sunny'}
    points = [2, 3, 4]
    job_fail_value = np.nan
    slurm_pool = SlurmPool(work_dir=work_dir, dim_input=1, dim_output=1, cluster='local', verbosity=verbosity,
                           extra_arg=setup_dict, job_fail_value=job_fail_value)
    res = slurm_pool.map(fun_with_extra_arg, points)
    res_expected = [job_fail_value for _ in points]
    np.testing.assert_array_equal(res, res_expected)
