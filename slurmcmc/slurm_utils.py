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


def _run_in_dir(run_dir: str, fun: Callable, args: List):
    """
    Worker-side shim: chdir into the point's own directory, evaluate, chdir back.

    Runs *inside* the submitit worker process, so the orchestrating process never
    changes its own cwd (a historical source of subtle path bugs and a
    thread-safety hazard). Module-level so it pickles by reference.
    """
    ini_dir = os.getcwd()
    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)
    try:
        return fun(*args)
    finally:
        try:
            os.chdir(ini_dir)
        except OSError:
            pass  # original dir vanished; worker is exiting anyway


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
        Base name for submitted Slurm jobs. Each iteration is submitted as one
        job array named ``{job_name}_{num_calls}`` (one scheduler transaction
        per iteration; array task indices correspond to point indices).
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
        Optional delay after each job-array submission (rate-limiting; with
        arrays there is only one submission per iteration).
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

    def submit_array_with_retry(self, executor, fun: Callable, point_dirs: List[str],
                                args_per_point: List[List]) -> List:
        """
        Submit the whole batch as a single job array (one scheduler transaction per
        iteration instead of one per point), retrying up to submit_retry_max_attempts
        times on failure. Each task runs through _run_in_dir so the worker starts in
        its own point directory.
        """
        run_dirs = [os.path.abspath(d) for d in point_dirs]
        funs = [fun] * len(run_dirs)
        attempts = 0
        while True:
            try:
                return executor.map_array(_run_in_dir, run_dirs, funs, args_per_point)
            except Exception as e:
                attempts += 1
                logging.info(f"Submission failed: {e}. Retrying {attempts}/{self.submit_retry_max_attempts}")
                if attempts >= self.submit_retry_max_attempts:
                    err_msg = "max submit retry attempts reached."
                    logging.error(err_msg)
                    raise RuntimeError(err_msg)
                time.sleep(self.submit_retry_wait_seconds)

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

        # submit the whole batch as one job array (workers chdir into their own
        # point directory via the _run_in_dir shim — the orchestrator's cwd is
        # never touched)
        executor = submitit.AutoExecutor(folder=iteration_dir, cluster=self.cluster)
        submitit_kwargs_iter = dict(self.submitit_kwargs)  # copy — don't mutate caller's dict
        submitit_kwargs_iter['slurm_job_name'] = f'{self.job_name}_{self.num_calls}'
        executor.update_parameters(**submitit_kwargs_iter)

        args_per_point = [self._combine_args(point) for point in points]
        jobs = self.submit_array_with_retry(executor, fun, point_dirs, args_per_point)
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

            except Exception:
                if self.verbosity >= 1:
                    # full traceback so transient infrastructure errors (squeue hiccups,
                    # slow filesystems) are diagnosable, not silently folded into job failure
                    logging.exception(f'Failed obtaining result of ind_point {ind_point}.')
                job_failed = True

            if job_failed:
                output = self.job_fail_value if self.dim_output == 1 else [self.job_fail_value] * self.dim_output

            point_dir = iteration_dir + '/' + str(ind_point)
            np.savetxt(point_dir + '/output.txt', [output])
            outputs.append(output)

        np.savetxt(iteration_dir + '/outputs.txt', np.array(outputs))
        return outputs


# ---------------------------------------------------------------------------
# Cluster helpers
# ---------------------------------------------------------------------------

def submit_remote_run(run_fun: Callable, config, work_dir: str, job_name: str,
                      remote_cluster: str, remote_submitit_kwargs: Optional[Dict]) -> submitit.Job:
    """
    Submit an orchestrator run (`run_fun(config)`) as its own Slurm/local job, so the
    long-running optimization/MCMC loop lives on the cluster rather than the login node.

    The entire run is described by a single picklable config dataclass — this replaces
    the old pattern of re-submitting the calling function with `locals()`.
    """
    remote_kwargs = dict(remote_submitit_kwargs or {})
    remote_kwargs.setdefault('slurm_job_name', 'main_' + job_name)
    remote_kwargs.setdefault('timeout_min', int(60 * 24 * 30))  # 1 month
    executor = submitit.AutoExecutor(folder=work_dir, cluster=remote_cluster)
    executor.update_parameters(**remote_kwargs)
    return executor.submit(run_fun, config)


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


def check_slurm_job_state(job_id: Union[int, str], num_attempts: int = 3,
                          retry_wait_seconds: float = 2) -> str:
    """
    Query ``squeue`` for the state of the given Slurm job.

    Returns one of: ``'RUNNING'``, ``'PENDING'``, ``'NOT_FOUND'``, ``'OTHER'``.

    Transient squeue failures (e.g. a busy Slurm controller) are retried
    num_attempts times before raising, so a momentary hiccup does not cause a
    healthy job to be counted as failed.
    """
    last_error = None
    for attempt in range(num_attempts):
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
            if e.stderr and "Invalid job id" in e.stderr:
                return "NOT_FOUND"
            last_error = e
            logging.warning(f"squeue query failed (attempt {attempt + 1}/{num_attempts}): {e}")
            if attempt < num_attempts - 1:
                time.sleep(retry_wait_seconds)

    err_msg = f"Error checking job status after {num_attempts} attempts: {last_error}"
    logging.error(err_msg)
    raise ValueError(err_msg)
