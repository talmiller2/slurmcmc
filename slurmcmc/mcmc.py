from __future__ import annotations

import dataclasses
import logging
import signal
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import emcee
import numpy as np
import submitit

from slurmcmc.general_utils import (set_logging, save_restart_file, load_restart_file, save_extra_arg_to_file,
                                    point_to_tuple, signal_handler)
from slurmcmc.import_utils import deferred_import_function_wrapper
from slurmcmc.slurm_utils import Cluster, SlurmPool, submit_remote_run


@dataclass
class MCMCConfig:
    """
    Complete, picklable description of a slurm_mcmc run.

    This is the single source of truth for a run's parameters: `MCMCRunner`
    consumes it, and remote mode pickles it into the driver Slurm job (instead
    of the old, fragile `locals()` re-submission trick).
    """
    log_prob_fun: Union[Callable, Dict]
    init_points: np.ndarray
    num_iters: int
    init_log_prob_fun_values: Optional[List[float]] = None
    progress: bool = False
    skip_initial_state_check: bool = True
    verbosity: int = 1
    slurm_verbosity: int = 0
    print_iter_interval: int = 1
    log_file: Optional[str] = None
    extra_arg: Any = None
    save_restart: bool = False
    load_restart: bool = False
    restart_file: str = 'mcmc_restart.pkl'
    status_restart: Optional[Dict] = None
    work_dir: str = 'mcmc'
    job_name: str = 'mcmc'
    cluster: Cluster = 'slurm'
    submitit_kwargs: Optional[Dict] = None
    emcee_kwargs: Optional[Dict] = None
    budget: int = int(1e6)
    job_fail_value: float = -1e10
    submit_retry_max_attempts: int = 5
    submit_retry_wait_seconds: float = 10
    submit_delay_seconds: float = 0
    check_output_interval_seconds: float = 1
    check_output_timeout_minutes: float = int(1e5)
    restart_save_interval: int = 1
    record_history: bool = True
    install_signal_handler: bool = True
    # remote run params:
    remote: bool = False
    remote_cluster: Literal['slurm', 'local'] = 'slurm'
    remote_submitit_kwargs: Optional[Dict] = None


class MCMCRunner:
    """
    Runner for ensemble MCMC on a Slurm cluster (submitit + emcee).

    The walkers' log-probability evaluations of each MCMC iteration are
    dispatched in parallel through SlurmPool (which emcee sees as a
    multiprocessing.Pool).

    Usage:
        status = MCMCRunner(MCMCConfig(log_prob_fun=..., init_points=..., ...)).run()

    Most callers use the `slurm_mcmc(...)` convenience wrapper instead.
    """

    def __init__(self, config: MCMCConfig) -> None:
        self.cfg = config
        self.sampler: Optional[emcee.EnsembleSampler] = None
        self.slurm_pool: Optional[SlurmPool] = None
        self.ini_iter: int = 0
        self.time_per_iter: List[float] = []

    # ------------------------------------------------------------------
    # Setup / state
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        cfg = self.cfg
        if cfg.status_restart is not None:
            if cfg.verbosity >= 1:
                logging.info('restarting from status_restart argument.')
            status = cfg.status_restart
        else:
            if cfg.verbosity >= 1:
                logging.info('loading restart file: ' + cfg.work_dir + '/' + cfg.restart_file)
            status = load_restart_file(cfg.work_dir, cfg.restart_file)
        self.sampler = status['sampler']
        self.slurm_pool = status['slurm_pool']
        self.sampler.pool = self.slurm_pool
        self.ini_iter = status['ini_iter']
        self.time_per_iter = status['time_per_iter']

    def _init_fresh_state(self, log_prob_fun: Callable) -> None:
        cfg = self.cfg
        emcee_kwargs = dict(cfg.emcee_kwargs or {})

        # using extra_arg=None because emcee deals with extra_arg internally by wrapping the function
        self.slurm_pool = SlurmPool(cfg.work_dir, cfg.job_name, cfg.cluster, verbosity=cfg.slurm_verbosity,
                                    extra_arg=cfg.extra_arg, submitit_kwargs=dict(cfg.submitit_kwargs or {}),
                                    dim_input=cfg.init_points.shape[1], dim_output=1,
                                    budget=cfg.budget, job_fail_value=cfg.job_fail_value,
                                    submit_retry_max_attempts=cfg.submit_retry_max_attempts,
                                    submit_retry_wait_seconds=cfg.submit_retry_wait_seconds,
                                    submit_delay_seconds=cfg.submit_delay_seconds,
                                    check_output_interval_seconds=cfg.check_output_interval_seconds,
                                    check_output_timeout_minutes=cfg.check_output_timeout_minutes,
                                    record_history=cfg.record_history,
                                    )

        # save the extra_arg in the work folder to document the full input used
        if cfg.cluster != 'local-map':
            save_extra_arg_to_file(cfg.work_dir, cfg.extra_arg)

        # supply args=[extra_arg] to emcee for it to wrap it internally
        if (cfg.extra_arg is not None) and ('args' not in emcee_kwargs):
            emcee_kwargs['args'] = [cfg.extra_arg]

        nwalkers, ndim = np.array(cfg.init_points).shape
        self.sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=log_prob_fun,
                                             pool=self.slurm_pool, **emcee_kwargs)

        init_log_prob_fun_values = cfg.init_log_prob_fun_values
        if init_log_prob_fun_values is None:
            # calculate the log probabilities of the init_points
            if cfg.verbosity >= 1:
                logging.info('### explicitly calculating the log probabilities of the init_points.')
            init_log_prob_fun_values = self.slurm_pool.map(log_prob_fun, cfg.init_points)
        else:
            if cfg.verbosity >= 1:
                logging.info('### setting the input init_log_prob_fun_values for the init_points.')

        # manually set the initial state
        self.sampler.initial_state = emcee.State(cfg.init_points, log_prob=np.array(init_log_prob_fun_values))

        # initializations
        self.ini_iter = 0
        self.time_per_iter = []

        # from here on, emcee deals itself with extra arguments by internally wrapping the log_prob_fun,
        # so we remove it from slurm_pool to avoid erroneously double wrapping it
        self.slurm_pool.extra_arg = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        cfg = self.cfg
        set_logging(cfg.work_dir, cfg.log_file)
        if cfg.install_signal_handler:
            signal.signal(signal.SIGTERM, signal_handler)  # force termination on scancel

        log_prob_fun = deferred_import_function_wrapper(cfg.log_prob_fun)

        if cfg.load_restart or cfg.status_restart is not None:
            self._load_state()
        else:
            self._init_fresh_state(log_prob_fun)

        status = None
        for curr_iter in range(self.ini_iter, self.ini_iter + cfg.num_iters):
            if cfg.verbosity >= 1 and np.mod(curr_iter, cfg.print_iter_interval) == 0:
                logging.info('### curr mcmc iter: ' + str(curr_iter))
            t_start_iter = time.time()
            state = self.sampler.run_mcmc(initial_state=self.sampler.initial_state, nsteps=1,
                                          progress=cfg.progress,
                                          skip_initial_state_check=cfg.skip_initial_state_check)
            curr_iter_time = time.time() - t_start_iter
            if cfg.verbosity >= 2 and np.mod(curr_iter, cfg.print_iter_interval) == 0:
                logging.info(f'    current iter run time: {curr_iter_time:.3f}s.')
            self.sampler.initial_state = state

            # mcmc status
            self.time_per_iter += [curr_iter_time]
            status = {
                'sampler': self.sampler,
                'slurm_pool': self.slurm_pool,
                'ini_iter': curr_iter + 1,
                'time_per_iter': self.time_per_iter,
            }

            if cfg.save_restart and np.mod(curr_iter, cfg.restart_save_interval) == 0:
                if cfg.verbosity >= 3:
                    logging.info('    saving restart file: ' + cfg.work_dir + '/' + cfg.restart_file)
                save_restart_file(status, cfg.work_dir, cfg.restart_file)
                self.sampler.pool = self.slurm_pool  # need to redefine the pool because pickling removes sampler.pool

        return status


def run_mcmc(config: MCMCConfig) -> Dict:
    """Run an MCMCRunner from a config object. Module-level so remote mode can pickle it by reference."""
    return MCMCRunner(config).run()


def slurm_mcmc(
        log_prob_fun: Union[Callable, Dict],
        init_points: np.ndarray,
        num_iters: int,
        init_log_prob_fun_values: Optional[List[float]] = None,
        progress: bool = False,
        skip_initial_state_check: bool = True,
        verbosity: int = 1,
        slurm_verbosity: int = 0,
        print_iter_interval: int = 1,
        log_file: Optional[str] = None,
        extra_arg: Any = None,
        save_restart: bool = False,
        load_restart: bool = False,
        restart_file: str = 'mcmc_restart.pkl',
        status_restart: Optional[Dict] = None,
        work_dir: str = 'mcmc',
        job_name: str = 'mcmc',
        cluster: Cluster = 'slurm',
        submitit_kwargs: Optional[Dict] = None,
        emcee_kwargs: Optional[Dict] = None,
        budget: int = int(1e6),
        job_fail_value: float = -1e10,
        submit_retry_max_attempts: int = 5,
        submit_retry_wait_seconds: float = 10,
        submit_delay_seconds: float = 0,
        check_output_interval_seconds: float = 1,
        check_output_timeout_minutes: float = int(1e5),
        restart_save_interval: int = 1,
        record_history: bool = True,
        install_signal_handler: bool = True,
        # remote run params:
        remote: bool = False,
        remote_cluster: Literal['slurm', 'local'] = 'slurm',
        remote_submitit_kwargs: Optional[Dict] = None,
) -> Union[Dict, submitit.Job]:
    """
    Combine submitit + emcee to allow ensemble MCMC on a Slurm cluster.

    The number of parallelisable walker evaluations per iteration is
    ``len(init_points) // 2`` (the default emcee stretch-move); the very
    first evaluation (on ``init_points``) uses all walkers.

    This is a thin convenience wrapper around MCMCConfig + MCMCRunner; use those
    directly for programmatic access to the run's configuration and state.

    Parameters
    ----------
    log_prob_fun : callable or dict
        Log-probability function ``log_prob_fun(x)`` (or ``log_prob_fun(x, extra_arg)``
        when *extra_arg* is given).  Can also be a dict with keys
        ``module_dir``, ``module_name``, ``function_name`` for deferred import.
    init_points : np.ndarray, shape (nwalkers, ndim)
        Starting positions of the MCMC walkers.
    num_iters : int
        Number of MCMC iterations to run.
    install_signal_handler : bool
        If True (default), install a SIGTERM handler that exits with code 1
        so Slurm marks the job as FAILED rather than COMPLETED on scancel.
        Set to False if you are embedding this function in a larger application
        that manages its own signal handling.
    remote : bool
        If True, submit the whole MCMC loop as its own job on remote_cluster and
        return the submitit Job handle immediately (job.result() gives the status
        dict). If False (default), run the loop in this process and return the
        status dict.
    """
    config = MCMCConfig(
        log_prob_fun=log_prob_fun, init_points=init_points, num_iters=num_iters,
        init_log_prob_fun_values=init_log_prob_fun_values, progress=progress,
        skip_initial_state_check=skip_initial_state_check,
        verbosity=verbosity, slurm_verbosity=slurm_verbosity, print_iter_interval=print_iter_interval,
        log_file=log_file, extra_arg=extra_arg,
        save_restart=save_restart, load_restart=load_restart, restart_file=restart_file,
        status_restart=status_restart,
        work_dir=work_dir, job_name=job_name, cluster=cluster,
        submitit_kwargs=submitit_kwargs, emcee_kwargs=emcee_kwargs,
        budget=budget, job_fail_value=job_fail_value,
        submit_retry_max_attempts=submit_retry_max_attempts,
        submit_retry_wait_seconds=submit_retry_wait_seconds,
        submit_delay_seconds=submit_delay_seconds,
        check_output_interval_seconds=check_output_interval_seconds,
        check_output_timeout_minutes=check_output_timeout_minutes,
        restart_save_interval=restart_save_interval, record_history=record_history,
        install_signal_handler=install_signal_handler,
        remote=remote, remote_cluster=remote_cluster, remote_submitit_kwargs=remote_submitit_kwargs,
    )

    if config.remote:
        set_logging(config.work_dir, config.log_file)
        if config.install_signal_handler:
            signal.signal(signal.SIGTERM, signal_handler)
        print('Running slurm_mcmc remotely.')
        return submit_remote_run(run_mcmc, dataclasses.replace(config, remote=False),
                                 config.work_dir, config.job_name,
                                 config.remote_cluster, config.remote_submitit_kwargs)

    return MCMCRunner(config).run()


def get_gelman_rubin_statistic(chains):
    """
    Calculate the Gelman-Rubin statistic for MCMC chains.
    Not really relevant for correlated chains as in the emcee algorithm, for more read:
    https://emcee.readthedocs.io/en/stable/tutorials/autocorr/

    Parameters:
    chains: np.ndarray of shape (nsteps, nwalkers, ndim)
        The MCMC samples from the emcee package.

    Returns:
    R_hat: np.ndarray of shape (ndim,)
        The Gelman-Rubin statistic for each parameter.
    """
    nsteps, nwalkers, ndim = chains.shape

    # Mean of each chain
    chain_means = np.mean(chains, axis=0)  # shape (nwalkers, ndim)

    # Variance of each chain
    chain_variances = np.var(chains, axis=0, ddof=1)  # shape (nwalkers, ndim)

    # Overall mean of chain variances
    W = np.mean(chain_variances, axis=0)  # shape (ndim,)

    # Between-chain variance (variance of the means of the chains, multiplied by an extra factor of nsteps)
    B = nsteps * np.var(chain_means, axis=0, ddof=1)  # shape (ndim,)

    # Estimate of the marginal posterior variance
    var_hat = W * (nsteps - 1) / nsteps + B / nsteps

    # calculate the potential scale reduction factor
    R_hat = np.sqrt(var_hat / W)

    return R_hat


def calculate_unique_points_weights(samples):
    """
    Extract the unique points and their weights (duplicates) from a points samples set.
    """
    unique_points_set = set()
    points_weights_dict = {}
    for point in samples:
        point_tuple = point_to_tuple(point)
        if point_tuple not in unique_points_set:
            # initialize new point
            unique_points_set.add(point_tuple)
            points_weights_dict[point_tuple] = 1
        else:
            points_weights_dict[point_tuple] += 1
    return unique_points_set, points_weights_dict
