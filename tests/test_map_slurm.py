import os

import pytest

from slurmcmc.general_utils import delete_directory
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


def test_slurmpool_slurm(work_dir):
    slurm_pool = SlurmPool(work_dir, job_name='test_slurmpool',
                           cluster='slurm', slurm_partition='socket',
                           # cluster='slurm', slurm_constraint='serial',
                           )
    fun = lambda x: x ** 2
    points = [2, 3, 4]
    res_expected = [fun(point) for point in points]
    res = slurm_pool.map(fun, points)
    assert res == res_expected
