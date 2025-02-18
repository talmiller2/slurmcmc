import os

import pytest

from slurmcmc.general_utils import delete_directory
from slurmcmc.slurm_utils import SlurmPool, is_slurm_cluster

submitit_kwargs = {'cluster': 'slurm', 'slurm_partition': 'core', 'timeout_min': 10}
# submitit_kwargs = {'cluster': 'slurm', 'slurm_constraint': 'serial'}

@pytest.fixture()
def work_dir(request):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.join(base_dir, f'test_work_dir_{request.node.name}')
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    yield work_dir
    os.chdir(base_dir)
    delete_directory(work_dir)


@pytest.mark.skipif(not is_slurm_cluster(), reason="This test only runs on a Slurm cluster")
def test_slurmpool_slurm(work_dir):
    slurm_pool = SlurmPool(work_dir, job_name='test_slurmpool', submitit_kwargs=submitit_kwargs)
    fun = lambda x: x ** 2
    points = [2, 3, 4]
    res_expected = [fun(point) for point in points]
    res = slurm_pool.map(fun, points)
    assert res == res_expected
