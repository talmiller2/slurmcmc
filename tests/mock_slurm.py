"""
MockSlurmJob and MockSlurmExecutor — test helpers for exercising Slurm-specific code
paths (state polling, timeouts, retries) without a real cluster.

Usage pattern in tests:
    MockExecutor = make_mock_executor_class(outcome='success', state_sequence=['PENDING', 'RUNNING', ''])
    monkeypatch.setattr(submitit, 'AutoExecutor', MockExecutor)
    monkeypatch.setattr(subprocess, 'run', MockSlurmJob.make_squeue_subprocess_run())
"""

import itertools
import subprocess


class MockSlurmJob:
    """
    Simulates a submitit Job on a Slurm cluster.

    Each call to get_squeue_state() advances through state_sequence, mimicking
    the lifecycle seen via `squeue`. An empty string ('') simulates a job that
    has left the queue (completed or cancelled).
    """

    _registry: dict = {}
    _id_counter = itertools.count(1)

    def __init__(self, fun, args, outcome='success', state_sequence=None):
        self.job_id = str(next(MockSlurmJob._id_counter))
        self._fun = fun
        self._args = args
        self._outcome = outcome
        self._cancelled = False
        # Default: PENDING → RUNNING → done (empty squeue output)
        self._state_sequence = state_sequence if state_sequence is not None else ['PENDING', 'RUNNING', '']
        self._state_idx = 0
        MockSlurmJob._registry[self.job_id] = self

    def get_squeue_state(self) -> str:
        """Return the squeue stdout string for the current simulated state."""
        if self._cancelled:
            return ''
        state = self._state_sequence[min(self._state_idx, len(self._state_sequence) - 1)]
        self._state_idx += 1
        return state

    def _get_outcome_and_result(self):
        if self._cancelled or self._outcome == 'error':
            return 'error', None
        result = self._fun(*self._args)
        return 'success', result

    def exception(self):
        return RuntimeError(f"Simulated Slurm job failure (job_id={self.job_id})")

    def cancel(self):
        self._cancelled = True

    @classmethod
    def make_squeue_subprocess_run(cls):
        """
        Return a mock for subprocess.run that answers squeue queries using the job registry.
        Pass to monkeypatch.setattr(subprocess, 'run', MockSlurmJob.make_squeue_subprocess_run()).
        """
        def mock_run(cmd, *args, **kwargs):
            if isinstance(cmd, list) and len(cmd) > 1 and cmd[0] == 'squeue':
                job_id = cmd[2]
                job = cls._registry.get(str(job_id))
                stdout = job.get_squeue_state() if job is not None else ''
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr='')
            raise RuntimeError(f"Unexpected subprocess call in mock: {cmd}")
        return mock_run

    @classmethod
    def reset(cls):
        """Clear the job registry and reset the ID counter. Call between tests."""
        cls._registry.clear()
        cls._id_counter = itertools.count(1)


class MockSlurmExecutor:
    """
    Simulates submitit.AutoExecutor, creating MockSlurmJobs on submit().
    Not used directly — use make_mock_executor_class() instead.
    """

    def __init__(self, folder, cluster='slurm', outcome='success', state_sequence=None):
        self.folder = folder
        self.cluster = cluster
        self._outcome = outcome
        self._state_sequence = state_sequence
        self._parameters = {}

    def update_parameters(self, **kwargs):
        self._parameters.update(kwargs)

    def submit(self, fun, *args):
        return MockSlurmJob(fun=fun, args=args, outcome=self._outcome,
                            state_sequence=self._state_sequence)


def make_mock_executor_class(outcome='success', state_sequence=None):
    """
    Factory returning a MockSlurmExecutor *class* (not instance).
    Suitable for monkeypatching submitit.AutoExecutor, which is called as
    submitit.AutoExecutor(folder=..., cluster=...).

    Args:
        outcome: 'success' or 'error' — controls job._get_outcome_and_result().
        state_sequence: list of squeue stdout strings the job cycles through.
            Use '' (empty string) to signal completion/NOT_FOUND.
            Default: ['PENDING', 'RUNNING', '']
    """
    _outcome = outcome
    _state_sequence = state_sequence

    class _MockExecutor(MockSlurmExecutor):
        def __init__(self, folder, cluster='slurm'):
            super().__init__(folder, cluster, outcome=_outcome, state_sequence=_state_sequence)

    return _MockExecutor
