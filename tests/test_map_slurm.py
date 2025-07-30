import copy
import os
import time

import numpy as np
import pytest

from slurmcmc.general_utils import delete_directory
from slurmcmc.slurm_utils import SlurmPool, is_slurm_cluster
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


def fun(x):
    time.sleep(120)
    return x ** 2


@pytest.mark.skipif(not is_slurm_cluster(), reason="This test only runs on a Slurm cluster")
def test_slurmpool_slurm(work_dir):
    slurm_pool = SlurmPool(work_dir=work_dir, dim_input=1, dim_output=1, job_name='test_slurmpool', cluster='slurm',
                           log_file='log.txt', submitit_kwargs=submitit_kwargs)
    fun = lambda x: x ** 2
    points = [2, 3, 4]
    res_expected = [fun(point) for point in points]
    res = slurm_pool.map(fun, points)
    assert res == res_expected


@pytest.mark.skipif(not is_slurm_cluster(), reason="This test only runs on a Slurm cluster")
def test_slurmpool_slurm_fail_on_slurm_timeout(work_dir):
    # revise submitit_kwargs to have a short timeout (given in seconds, timeout_min<1 not allowed)
    submitit_kwargs_short_timeout = copy.deepcopy(submitit_kwargs)
    submitit_kwargs_short_timeout.pop('timeout_min')
    submitit_kwargs_short_timeout['slurm_additional_parameters']['time'] = '00:00:10'  # 10 seconds
    submitit_kwargs_short_timeout['slurm_signal_delay_s'] = 5

    job_fail_value = np.nan
    slurm_pool = SlurmPool(work_dir=work_dir, dim_input=1, dim_output=1, job_name='test_slurmpool', cluster='slurm',
                           log_file='log.txt', job_fail_value=job_fail_value,
                           submitit_kwargs=submitit_kwargs_short_timeout, )
    points = [5, 6]
    res_expected_fail = [job_fail_value for _ in points]  # all jobs should fail due to timeout
    res = slurm_pool.map(fun, points)
    assert res == res_expected_fail


@pytest.mark.skipif(not is_slurm_cluster(), reason="This test only runs on a Slurm cluster")
def test_slurmpool_slurm_fail_on_slurmpool_timeout(work_dir):
    job_fail_value = np.nan
    slurm_pool = SlurmPool(work_dir=work_dir, dim_input=1, dim_output=1, job_name='test_slurmpool', cluster='slurm',
                           log_file='log.txt', job_fail_value=job_fail_value,
                           check_output_interval_seconds=1, check_output_timeout_minutes=0.1)
    points = [5, 6]
    res_expected_fail = [job_fail_value for _ in points]  # all jobs should fail due to timeout
    res = slurm_pool.map(fun, points)
    assert res == res_expected_fail
