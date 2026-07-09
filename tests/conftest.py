import os

import numpy as np
import pytest
import torch

# Repo root must be importable inside submitit's spawned job processes, which
# unpickle functions referencing the `tests` package. The editable install only
# exposes the `slurmcmc` package, so export the root via PYTHONPATH (inherited
# by subprocesses).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _repo_root_on_subprocess_path(monkeypatch):
    existing = os.environ.get('PYTHONPATH', '')
    pythonpath = _REPO_ROOT + (os.pathsep + existing if existing else '')
    monkeypatch.setenv('PYTHONPATH', pythonpath)


@pytest.fixture()
def work_dir(tmp_path, monkeypatch):
    """Fresh isolated work directory for each test; cwd is set to it and auto-restored."""
    monkeypatch.chdir(tmp_path)
    return str(tmp_path)


@pytest.fixture()
def verbosity():
    return 1


@pytest.fixture()
def seed():
    np.random.seed(0)
    torch.manual_seed(0)
