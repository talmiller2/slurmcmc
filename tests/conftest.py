import os
import shutil
import uuid

import numpy as np
import pytest
import torch

# Repo root must be importable inside submitit's spawned job processes, which
# unpickle functions referencing the `tests` package. The editable install only
# exposes the `slurmcmc` package, so export the root via PYTHONPATH (inherited
# by subprocesses).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Work dirs live under tests/, not the system temp dir. On a Slurm cluster /tmp is
# typically local to each node and not visible to submitted jobs, so a work_dir under
# pytest's tmp_path (which resolves under /tmp) silently breaks every test that
# actually submits a Slurm job — the compute node can't see the directory the
# submission node created. The repo checkout is normally on a shared filesystem
# visible to both, so that's where these need to live.
_TEST_WORK_DIRS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_work_dirs')


@pytest.fixture(autouse=True)
def _repo_root_on_subprocess_path(monkeypatch):
    existing = os.environ.get('PYTHONPATH', '')
    pythonpath = _REPO_ROOT + (os.pathsep + existing if existing else '')
    monkeypatch.setenv('PYTHONPATH', pythonpath)


def pytest_runtest_makereport(item, call):
    # stash the outcome of each test phase on the item, so the work_dir fixture's
    # teardown can tell whether the test failed (see below)
    if call.when == 'call':
        item.stash_test_failed = call.excinfo is not None


@pytest.fixture()
def work_dir(request, monkeypatch):
    """
    Fresh, uniquely-named work directory under tests/test_work_dirs/, cwd set to it
    (restored on teardown). Deleted after the test unless it failed or
    SLURMCMC_KEEP_TEST_DIRS is set — set that env var to inspect a test's Slurm
    submission layout (input.txt/output.txt per point, logs, etc.) after the run.
    """
    dir_name = f'{request.node.name}_{uuid.uuid4().hex[:8]}'  # uuid avoids collisions with stale/kept dirs
    work_dir_path = os.path.join(_TEST_WORK_DIRS_ROOT, dir_name)
    os.makedirs(work_dir_path, exist_ok=True)
    monkeypatch.chdir(work_dir_path)

    yield work_dir_path

    keep = os.environ.get('SLURMCMC_KEEP_TEST_DIRS', '0').lower() not in ('', '0', 'false')
    failed = getattr(request.node, 'stash_test_failed', False)
    if not keep and not failed:
        shutil.rmtree(work_dir_path, ignore_errors=True)


@pytest.fixture()
def verbosity():
    return 1


@pytest.fixture()
def seed():
    np.random.seed(0)
    torch.manual_seed(0)
