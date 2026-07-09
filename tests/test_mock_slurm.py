"""
Tests for Slurm-specific code paths using MockSlurmJob.
These run without a real Slurm cluster by mocking submitit.AutoExecutor and subprocess.run.
"""

import subprocess

import numpy as np
import pytest
import submitit

from slurmcmc.slurm_utils import SlurmPool, check_slurm_job_state
from tests.mock_slurm import MockSlurmJob, make_mock_executor_class


@pytest.fixture(autouse=True)
def reset_registry():
    """Ensure a clean MockSlurmJob registry for every test."""
    MockSlurmJob.reset()
    yield
    MockSlurmJob.reset()


# ---------------------------------------------------------------------------
# check_slurm_job_state unit tests (mocked subprocess.run)
# ---------------------------------------------------------------------------

def test_check_slurm_job_state_running(monkeypatch):
    """RUNNING squeue output → RUNNING."""
    monkeypatch.setattr(subprocess, 'run', lambda cmd, **kw: subprocess.CompletedProcess(
        args=cmd, returncode=0, stdout='RUNNING', stderr=''))
    assert check_slurm_job_state(1) == 'RUNNING'


def test_check_slurm_job_state_pending(monkeypatch):
    """PENDING squeue output → PENDING."""
    monkeypatch.setattr(subprocess, 'run', lambda cmd, **kw: subprocess.CompletedProcess(
        args=cmd, returncode=0, stdout='PENDING', stderr=''))
    assert check_slurm_job_state(1) == 'PENDING'


def test_check_slurm_job_state_not_found_empty_output(monkeypatch):
    """Empty squeue output (job left queue) → NOT_FOUND."""
    monkeypatch.setattr(subprocess, 'run', lambda cmd, **kw: subprocess.CompletedProcess(
        args=cmd, returncode=0, stdout='', stderr=''))
    assert check_slurm_job_state(1) == 'NOT_FOUND'


def test_check_slurm_job_state_other(monkeypatch):
    """Non-RUNNING/PENDING states (e.g. COMPLETING) → OTHER."""
    monkeypatch.setattr(subprocess, 'run', lambda cmd, **kw: subprocess.CompletedProcess(
        args=cmd, returncode=0, stdout='COMPLETING', stderr=''))
    assert check_slurm_job_state(1) == 'OTHER'


def test_check_slurm_job_state_invalid_job_id(monkeypatch):
    """Invalid job ID error from squeue → NOT_FOUND (treated as completed/gone)."""
    def raise_invalid(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd, stderr='Invalid job id specified')
    monkeypatch.setattr(subprocess, 'run', raise_invalid)
    assert check_slurm_job_state(99999) == 'NOT_FOUND'


# ---------------------------------------------------------------------------
# SlurmPool integration tests with MockSlurmExecutor
# ---------------------------------------------------------------------------

def test_slurmpool_mock_slurm_basic(work_dir, verbosity, monkeypatch):
    """Full Slurm path: submit → poll PENDING/RUNNING → collect result."""
    MockExecutor = make_mock_executor_class(
        outcome='success',
        state_sequence=['PENDING', 'RUNNING', ''],
    )
    monkeypatch.setattr(submitit, 'AutoExecutor', MockExecutor)
    monkeypatch.setattr(subprocess, 'run', MockSlurmJob.make_squeue_subprocess_run())

    pool = SlurmPool(work_dir=work_dir, dim_input=1, dim_output=1,
                     cluster='slurm', verbosity=verbosity,
                     check_output_interval_seconds=0.01)
    res = pool.map(lambda x: x ** 2, [2, 3, 4])
    assert res == [4, 9, 16]
    assert pool.num_evaluated_points == 3
    assert len(pool.inds_success_points) == 3
    assert len(pool.inds_failed_points) == 0


def test_slurmpool_mock_slurm_job_failure(work_dir, verbosity, monkeypatch):
    """Jobs returning error outcome are recorded as failures and return job_fail_value."""
    MockExecutor = make_mock_executor_class(
        outcome='error',
        state_sequence=['RUNNING', ''],
    )
    monkeypatch.setattr(submitit, 'AutoExecutor', MockExecutor)
    monkeypatch.setattr(subprocess, 'run', MockSlurmJob.make_squeue_subprocess_run())

    job_fail_value = np.nan
    pool = SlurmPool(work_dir=work_dir, dim_input=1, dim_output=1,
                     cluster='slurm', verbosity=verbosity,
                     job_fail_value=job_fail_value,
                     check_output_interval_seconds=0.01)
    res = pool.map(lambda x: x ** 2, [2, 3])
    assert all(np.isnan(r) for r in res)
    assert len(pool.inds_failed_points) == 2
    assert len(pool.inds_success_points) == 0


def test_slurmpool_mock_slurm_timeout(work_dir, verbosity, monkeypatch):
    """Jobs stuck in RUNNING past check_output_timeout_minutes are cancelled and fail."""
    MockExecutor = make_mock_executor_class(
        outcome='success',
        state_sequence=['RUNNING'] * 200,  # never completes
    )
    monkeypatch.setattr(submitit, 'AutoExecutor', MockExecutor)
    monkeypatch.setattr(subprocess, 'run', MockSlurmJob.make_squeue_subprocess_run())

    job_fail_value = np.nan
    pool = SlurmPool(work_dir=work_dir, dim_input=1, dim_output=1,
                     cluster='slurm', verbosity=verbosity,
                     job_fail_value=job_fail_value,
                     check_output_interval_seconds=0.01,
                     check_output_timeout_minutes=0.005)  # ~0.3 s
    res = pool.map(lambda x: x ** 2, [2])
    assert np.isnan(res[0])
    assert len(pool.inds_failed_points) == 1


def test_slurmpool_mock_slurm_history(work_dir, verbosity, monkeypatch):
    """points_history and values_history are correctly populated for the slurm path."""
    MockExecutor = make_mock_executor_class(state_sequence=['RUNNING', ''])
    monkeypatch.setattr(submitit, 'AutoExecutor', MockExecutor)
    monkeypatch.setattr(subprocess, 'run', MockSlurmJob.make_squeue_subprocess_run())

    pool = SlurmPool(work_dir=work_dir, dim_input=1, dim_output=1,
                     cluster='slurm', verbosity=verbosity,
                     check_output_interval_seconds=0.01)
    points = [2, 3, 4]
    pool.map(lambda x: x ** 2, points)
    np.testing.assert_array_equal(pool.points_history, np.array(points).reshape(-1, 1))
    np.testing.assert_array_equal(pool.values_history, np.array([4, 9, 16]).reshape(-1, 1))


def test_slurmpool_mock_slurm_submitit_kwargs_not_mutated(work_dir, verbosity, monkeypatch):
    """submitit_kwargs dict must not be mutated by SlurmPool (each job gets its own copy)."""
    MockExecutor = make_mock_executor_class(state_sequence=['RUNNING', ''])
    monkeypatch.setattr(submitit, 'AutoExecutor', MockExecutor)
    monkeypatch.setattr(subprocess, 'run', MockSlurmJob.make_squeue_subprocess_run())

    original_kwargs = {'slurm_partition': 'gpu', 'timeout_min': 60}
    import copy
    kwargs_before = copy.deepcopy(original_kwargs)

    pool = SlurmPool(work_dir=work_dir, dim_input=1, dim_output=1,
                     cluster='slurm', verbosity=verbosity,
                     submitit_kwargs=original_kwargs,
                     check_output_interval_seconds=0.01)
    pool.map(lambda x: x ** 2, [1, 2])
    # slurm_job_name is auto-added by __init__, but partition and timeout should be unchanged
    assert original_kwargs['slurm_partition'] == kwargs_before['slurm_partition']
    assert original_kwargs['timeout_min'] == kwargs_before['timeout_min']


def test_submit_with_retry_succeeds_after_transient_failures(work_dir, verbosity, monkeypatch):
    """submit_with_retry retries on submit() exceptions and succeeds once stable."""
    call_count = {'n': 0}

    class FlakyExecutor:
        def __init__(self, folder, cluster):
            self.folder = folder

        def update_parameters(self, **kwargs):
            pass

        def submit(self, fun, *args):
            call_count['n'] += 1
            if call_count['n'] < 3:
                raise RuntimeError("Simulated transient submit failure")
            return MockSlurmJob(fun=fun, args=args, state_sequence=['RUNNING', ''])

    monkeypatch.setattr(submitit, 'AutoExecutor', FlakyExecutor)
    monkeypatch.setattr(subprocess, 'run', MockSlurmJob.make_squeue_subprocess_run())

    pool = SlurmPool(work_dir=work_dir, dim_input=1, dim_output=1,
                     cluster='slurm', verbosity=verbosity,
                     submit_retry_max_attempts=5,
                     submit_retry_wait_seconds=0,
                     check_output_interval_seconds=0.01)
    res = pool.map(lambda x: x ** 2, [3])
    assert res == [9]
    assert call_count['n'] == 3  # failed twice, succeeded on third


def test_submit_with_retry_exhausts_raises(work_dir, verbosity, monkeypatch):
    """submit_with_retry raises after exhausting all attempts."""
    class AlwaysFailExecutor:
        def __init__(self, folder, cluster):
            pass
        def update_parameters(self, **kwargs):
            pass
        def submit(self, fun, *args):
            raise RuntimeError("Always fails")

    monkeypatch.setattr(submitit, 'AutoExecutor', AlwaysFailExecutor)

    pool = SlurmPool(work_dir=work_dir, dim_input=1, dim_output=1,
                     cluster='slurm', verbosity=verbosity,
                     submit_retry_max_attempts=2,
                     submit_retry_wait_seconds=0)
    with pytest.raises(RuntimeError, match="max submit retry attempts reached"):
        pool.map(lambda x: x ** 2, [2])
