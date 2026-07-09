from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import submitit

from slurmcmc.general_utils import (combine_args, set_logging, save_extra_arg_to_file, point_to_tuple,
                                    list_directories, calc_dimension)
from slurmcmc.history import History
from slurmcmc.import_utils import deferred_import_function_wrapper

# Cluster mode type alias — used in type hints throughout the package.
Cluster = Literal["slurm", "local", "local-map"]


class SlurmPool:
    """
    A drop-in replacement for ``multiprocessing.Pool`` whose ``.map()`` method
    dispatches evaluations to a Slurm cluster (or local processes) via
    `submitit <https://github.com/facebookincubator/submitit>`_.

    Designed to be compatible with emcee's ``EnsembleSampler(pool=...)``
    interface so that MCMC walkers are evaluated in parallel on a cluster.

    Parameters
    ----------
    work_dir : str
        Root directory for per-iteration and per-point output files.
        Must be empty (no numeric sub-directories) when starting a fresh run.
        Relative paths are converted to absolute at construction time.
    job_name : str
        Base name for submitted Slurm jobs.
    cluster : Cluster
        ``'slurm'`` — submit to a real Slurm cluster via submitit.
        ``'local'`` — run locally using submitit's local executor (same
        directory layout as ``'slurm'``; useful for debugging).
        ``'local-map'`` — evaluate sequentially in-process (fastest for
        analytic functions and CI tests).
    verbosity : int
        0 = silent, 1 = iteration-level info, 2 = timing, 3 = full debug.
    log_file : str or None
        If given, write log messages to ``work_dir/log_file`` in addition
        to stdout.
    extra_arg : any
        A constant extra argument forwarded to every function call as the
        second positional argument: ``fun(point, extra_arg)``.
    submitit_kwargs : dict or None
        Keyword arguments forwarded to ``submitit.AutoExecutor.update_parameters``.
    dim_input : int
        Number of input dimensions (must be a positive integer).
    dim_output : int
        Number of output dimensions (must be a positive integer).
    budget : int
        Maximum number of points per ``map()`` call.  Larger batches are
        split into chunks of this size.
    job_fail_value : float
        Sentinel value returned (and stored in history) when a job fails.
        Defaults to ``np.nan``.
    submit_retry_max_attempts : int
        Number of times to retry a failed job *submission* before raising.
    submit_retry_wait_seconds : float
        Seconds to wait between submission retries.
    submit_delay_seconds : float
        Optional delay between successive job submissions (rate-limiting).
    check_output_interval_seconds : float
        How often to poll job state while waiting for results.
    check_output_timeout_minutes : float
        Maximum time a job may spend in RUNNING state before being cancelled
        and counted as failed.
    record_history : bool
        If True (default), maintain ``points_history``, ``values_history``,
        and related tracking structures.  Set to False to save memory for
        very long runs where history is not needed.
    """

    def __init__(
        self,
        work_dir: str = 'slurmpool',
        job_name: str = 'slurmpool',
        cluster: Cluster = 'slurm',
        verbosity: int = 1,
        log_file: Optional[str] = None,
        extra_arg: Any = None,
        submitit_kwargs: Optional[Dict[str, Any]] = None,
        dim_input: Optional[int] = None,
        dim_output: Optional[int] = None,
        budget: int = int(1e6),
        job_fail_value: float = np.nan,
        submit_retry_max_attempts: int = 5,
        submit_retry_wait_seconds: float = 10,
        submit_delay_seconds: float = 0,
        check_output_interval_seconds: float = 1,
        check_output_timeout_minutes: float = int(1e5),
        record_history: bool = True,
    ) -> None:
        if not (isinstance(dim_input, int) and dim_input > 0):
            err_msg = f'dim_input must be a positive integer. dim_input={dim_input}'
            logging.error(err_msg)
            raise ValueError(err_msg)
        self.dim_input = dim_input

        if not (isinstance(dim_output, int) and dim_output > 0):
            err_msg = f'dim_output must be a positive integer. dim_output={dim_output}'
            logging.error(err_msg)
            raise ValueError(err_msg)
        self.dim_output = dim_output

        self.num_calls: int = 0
        self.run_time_minutes_per_call: List[float] = []

        self.record_history = record_history
        # History is stored in a dedicated object; SlurmPool exposes its
        # attributes as read-only properties for full backward compatibility.
        self._history: Optional[History] = History(dim_input, dim_output) if record_history else None

        # store as an absolute path: send_and_receive_jobs chdirs into per-point
        # directories, so relative paths would resolve against the wrong base
        self.work_dir = os.path.abspath(work_dir)
        self.job_name = job_name
        self.cluster = cluster
        self.verbosity = verbosity
        self.log_file = log_file

        if submitit_kwargs is None:
            submitit_kwargs = {}
        if 'slurm_job_name' not in submitit_kwargs:
            submitit_kwargs['slurm_job_name'] = job_name
        if 'timeout_min' not in submitit_kwargs:
            submitit_kwargs['timeout_min'] = int(60 * 24 * 30)  # 1 month
        self.submitit_kwargs = submitit_kwargs

        self.budget = budget
        self.job_fail_value = job_fail_value
        self.submit_retry_max_attempts = submit_retry_max_attempts
        self.submit_retry_wait_seconds = submit_retry_wait_seconds
        self.submit_delay_seconds = submit_delay_seconds
        self.check_output_interval_seconds = check_output_interval_seconds
        self.check_output_timeout_minutes = check_output_timeout_minutes

        set_logging(self.work_dir, self.log_file)

        if cluster in ['local', 'slurm']:
            os.makedirs(self.work_dir, exist_ok=True)
            # Only reject directories that contain numeric sub-dirs (iteration outputs).
            # Non-numeric entries (e.g. a restart .pkl or a log file) are fine.
            numeric_dirs = [d for d in list_directories(self.work_dir) if d.isdigit()]
            if numeric_dirs:
                err_msg = ('work_dir already contains iteration output directories — '
                           'move or delete them before starting a fresh run, or use load_restart=True.\n'
                           f'work_dir: {self.work_dir}')
                raise ValueError(err_msg)
                # note: when continuing from a restart, SlurmPool is loaded from the
                # pickle (not re-initialised), so this check is never triggered.

        self.extra_arg = extra_arg

    # ------------------------------------------------------------------
    # Backward-compatible property delegates onto self._history
    # ------------------------------------------------------------------
    # Raising AttributeError (not returning None) keeps hasattr() correct:
    # hasattr(pool, 'points_history') returns False when record_history=False.

    def _require_history(self, attr: str) -> History:
        if self._history is None:
            raise AttributeError(
                f"SlurmPool.{attr} is not available when record_history=False"
            )
        return self._history

    @property
    def points_history(self) -> np.ndarray:
        return self._require_history('points_history').points_history

    @property
    def values_history(self) -> np.ndarray:
        return self._require_history('values_history').values_history

    @property
    def num_evaluated_points(self) -> int:
        return self._require_history('num_evaluated_points').num_evaluated_points

    @property
    def inds_success_points(self) -> List[int]:
        return self._require_history('inds_success_points').inds_success_points

    @property
    def inds_failed_points(self) -> List[int]:
        return self._require_history('inds_failed_points').inds_failed_points

    @property
    def evaluated_points_set(self) -> set:
        return self._require_history('evaluated_points_set').evaluated_points_set

    @property
    def point_loc_dict(self) -> dict:
        return self._require_history('point_loc_dict').point_loc_dict

    # ------------------------------------------------------------------
    # Core map interface
    # ------------------------------------------------------------------

    def map(self, fun: Callable, points: List) -> List:
        fun = deferred_import_function_wrapper(fun)

        # split points into chunks if the batch exceeds budget
        chunks = self.split_points(points, self.budget)
        if self.verbosity >= 1 and len(chunks) > 1:
            chunk_sizes = [len(c) for c in chunks]
            logging.info(f'split points into {len(chunks)} chunks of sizes {chunk_sizes}.')

        res: List = []
        for chunk in chunks:
            res += self.map_chunk(fun, chunk)
        return res

    def split_points(self, points: List, budget: int) -> List[List]:
        """Partition *points* into sub-lists of at most *budget* entries."""
        num_chunks = len(points) // budget + (1 if len(points) % budget else 0)
        return [points[i * budget:(i + 1) * budget] for i in range(num_chunks)]

    def _combine_args(self, point: Any) -> List:
        return combine_args(point, self.extra_arg)

    def submit_with_retry(self, fun: Callable, point: Any):
        """Submit a single job, retrying up to submit_retry_max_attempts times on failure."""
        attempts = 0
        while attempts < self.submit_retry_max_attempts:
            try:
                job = self.executor.submit(fun, *self._combine_args(point))
                return job
            except Exception as e:
                attempts += 1
                logging.info(f"Submission failed: {e}. Retrying {attempts}/{self.submit_retry_max_attempts}")
                time.sleep(self.submit_retry_wait_seconds)
            if attempts == self.submit_retry_max_attempts:
                err_msg = "max submit retry attempts reached."
                logging.error(err_msg)
                raise RuntimeError(err_msg)

    def map_chunk(self, fun: Callable, points: List) -> List:
        map_start_time = time.time()

        if self.verbosity >= 1:
            logging.info(f'slurm_pool.map_chunk called with {len(points)} points.')

        # validate input dimensions
        for point in points:
            dim_curr_input = calc_dimension(point)
            if dim_curr_input != self.dim_input:
                err_msg = (f'inconsistent dimensions. expecting dim_input={self.dim_input} '
                           f'but dim_curr_input={dim_curr_input}')
                logging.error(err_msg)
                raise ValueError(err_msg)

        # evaluate fun on the points
        if self.cluster == 'local-map':
            res = [fun(*self._combine_args(point)) for point in points]
        else:
            res = self.send_and_receive_jobs(fun, points)

        # validate output dimensions
        for output in res:
            dim_curr_output = calc_dimension(output)
            if dim_curr_output != self.dim_output:
                err_msg = (f'inconsistent dimensions. expecting dim_output={self.dim_output} '
                           f'but dim_curr_output={dim_curr_output}')
                logging.error(err_msg)
                raise ValueError(err_msg)

        if self.record_history:
            self._history.record(points, res, self.check_failed)

        self.num_calls += 1
        self.run_time_minutes_per_call.append((time.time() - map_start_time) / 60.0)

        return res

    def check_failed(self, r: Any) -> bool:
        """Return True if *r* indicates a failed evaluation (None, NaN, or job_fail_value)."""
        if isinstance(r, (list, np.ndarray)) and np.ndim(r) > 0:
            elements = list(np.ravel(r))
        else:
            elements = [r]
        for e in elements:
            if e is None:
                return True
            try:
                if np.isnan(e):
                    return True
            except (TypeError, ValueError):
                pass
            if e == self.job_fail_value:
                return True
        return False

    def send_and_receive_jobs(self, fun: Callable, points: List) -> List:
        ini_dir = os.getcwd()

        # prepare per-iteration and per-point directories
        iteration_dir = self.work_dir + '/' + str(self.num_calls)
        os.makedirs(iteration_dir, exist_ok=True)
        point_dirs = []
        for ind_point, point in enumerate(points):
            point_dir = iteration_dir + '/' + str(ind_point)
            point_dirs.append(point_dir)
            os.makedirs(point_dir, exist_ok=True)
            np.savetxt(point_dir + '/input.txt', [point])

        np.savetxt(iteration_dir + '/inputs.txt', np.array(points))
        save_extra_arg_to_file(iteration_dir, self.extra_arg)

        # Phase 1: register point locations in history before jobs run
        if self.record_history:
            self._history.record_point_locations(points, self.num_calls)

        # submit all jobs
        jobs = []
        for ind_point, (point, point_dir) in enumerate(zip(points, point_dirs)):
            job_name = f'{self.job_name}_{self.num_calls}_{ind_point}'
            submitit_kwargs_point = dict(self.submitit_kwargs)  # copy — don't mutate caller's dict
            submitit_kwargs_point['slurm_job_name'] = job_name
            self.executor = submitit.AutoExecutor(folder=point_dir, cluster=self.cluster)
            self.executor.update_parameters(**submitit_kwargs_point)
            os.chdir(point_dir)  # worker process starts in its own directory
            job = self.submit_with_retry(fun, point)
            jobs.append(job)
            if self.submit_delay_seconds > 0:
                time.sleep(self.submit_delay_seconds)

        # collect results
        outputs = []
        for ind_point, job in enumerate(jobs):
            try:
                check_output_timeout_seconds = self.check_output_timeout_minutes * 60
                running_started = False
                job_running_start_time = None
                job_failed = False

                state = check_job_state(job, self.cluster)
                while state in ['RUNNING', 'PENDING']:
                    if state == 'RUNNING':
                        if not running_started:
                            running_started = True
                            job_running_start_time = time.time()
                        elif time.time() - job_running_start_time > check_output_timeout_seconds:
                            if self.verbosity >= 1:
                                logging.info(
                                    f"ind_point {ind_point} job {job.job_id} exceeded "
                                    f"{self.check_output_timeout_minutes:.2f} min. Cancelling."
                                )
                            job.cancel()
                            job_failed = True
                            break

                    time.sleep(self.check_output_interval_seconds)
                    state = check_job_state(job, self.cluster)

                if not job_failed:
                    outcome, output = job._get_outcome_and_result()
                    if outcome == "error":
                        if self.verbosity >= 1:
                            logging.info('job._get_outcome_and_result() failed. Exception:\n'
                                         + str(job.exception()))
                        job_failed = True

            except Exception as e:
                if self.verbosity >= 1:
                    logging.info('Failed obtaining job result. Exception:\n' + str(e))
                job_failed = True

            if job_failed:
                output = self.job_fail_value if self.dim_output == 1 else [self.job_fail_value] * self.dim_output

            point_dir = iteration_dir + '/' + str(ind_point)
            np.savetxt(point_dir + '/output.txt', [output])
            outputs.append(output)

        np.savetxt(iteration_dir + '/outputs.txt', np.array(outputs))
        os.chdir(ini_dir)
        return outputs


# ---------------------------------------------------------------------------
# Cluster helpers
# ---------------------------------------------------------------------------

def is_slurm_cluster() -> bool:
    """Return True if running on a machine connected to a Slurm cluster."""
    return shutil.which('srun') is not None


def check_job_state(job, cluster: Cluster) -> str:
    """Return the current state string for *job* on the given *cluster* type."""
    if cluster == 'local':
        return job.state
    elif cluster == 'slurm':
        return check_slurm_job_state(job.job_id)
    else:
        err_msg = f"invalid cluster type: {cluster}."
        logging.error(err_msg)
        raise ValueError(err_msg)


def check_slurm_job_state(job_id: Union[int, str]) -> str:
    """
    Query ``squeue`` for the state of the given Slurm job.

    Returns one of: ``'RUNNING'``, ``'PENDING'``, ``'NOT_FOUND'``, ``'OTHER'``.
    """
    try:
        result = subprocess.run(
            ['squeue', '-j', str(job_id), '-h', '-o', '%T'],
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout.strip()
        if not output:
            return "NOT_FOUND"
        if output in ("RUNNING", "PENDING"):
            return output
        return "OTHER"  # COMPLETING, FAILED, etc.

    except subprocess.CalledProcessError as e:
        if "Invalid job id" in e.stderr:
            return "NOT_FOUND"
        err_msg = f"Error checking job status: {e}"
        logging.error(err_msg)
        raise ValueError(err_msg)
