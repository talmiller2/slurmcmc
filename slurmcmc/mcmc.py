from __future__ import annotations

import functools
import logging
import signal
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import emcee
import numpy as np
import submitit

from slurmcmc.general_utils import (set_logging, save_restart_file, load_restart_file, save_extra_arg_to_file,
                                    point_to_tuple, signal_handler)
from slurmcmc.import_utils import deferred_import_function_wrapper
from slurmcmc.slurm_utils import Cluster, SlurmPool


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
):
    """
    Combine submitit + emcee to allow ensemble MCMC on a Slurm cluster.

    The number of parallelisable walker evaluations per iteration is
    ``len(init_points) // 2`` (the default emcee stretch-move); the very
    first evaluation (on ``init_points``) uses all walkers.

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
    """
    set_logging(work_dir, log_file)
    if install_signal_handler:
        signal.signal(signal.SIGTERM, signal_handler)

    if remote == True:
        print('Running slurm_mcmc remotely.')
        if remote_submitit_kwargs is None:
            remote_submitit_kwargs = {}
        if 'slurm_job_name' not in remote_submitit_kwargs:
            remote_submitit_kwargs['slurm_job_name'] = 'main_' + job_name
        if 'timeout_min' not in remote_submitit_kwargs:
            remote_submitit_kwargs['timeout_min'] = int(60 * 24 * 30)  # 1 month
        kwargs = locals()
        kwargs['remote'] = False
        executor = submitit.AutoExecutor(folder=work_dir, cluster=remote_cluster)
        executor.update_parameters(**remote_submitit_kwargs)
        job = executor.submit(functools.partial(slurm_mcmc, **kwargs))
        return job

    else:
        if submitit_kwargs is None:
            submitit_kwargs = {}
        if emcee_kwargs is None:
            emcee_kwargs = {}

        log_prob_fun = deferred_import_function_wrapper(log_prob_fun)

        if load_restart == True or status_restart is not None:
            if status_restart is not None:
                if verbosity >= 1:
                    logging.info('restarting from status_restart argument.')
                status = status_restart
            else:
                if verbosity >= 1:
                    logging.info('loading restart file: ' + work_dir + '/' + restart_file)
                status = load_restart_file(work_dir, restart_file)
            sampler = status['sampler']
            slurm_pool = status['slurm_pool']
            sampler.pool = slurm_pool
            ini_iter = status['ini_iter']
            time_per_iter = status['time_per_iter']
        else:
            # using extra_arg=None because emcee deals with extra_arg internally by wrapping the function
            slurm_pool = SlurmPool(work_dir, job_name, cluster, verbosity=slurm_verbosity, extra_arg=extra_arg,
                                   submitit_kwargs=submitit_kwargs, dim_input=init_points.shape[1], dim_output=1,
                                   budget=budget, job_fail_value=job_fail_value,
                                   submit_retry_max_attempts=submit_retry_max_attempts,
                                   submit_retry_wait_seconds=submit_retry_wait_seconds,
                                   submit_delay_seconds=submit_delay_seconds,
                                   check_output_interval_seconds=check_output_interval_seconds,
                                   check_output_timeout_minutes=check_output_timeout_minutes,
                                   record_history=record_history,
                                   )

            # save the extra_arg in the work folder to document the full input used
            if cluster != 'local-map':
                save_extra_arg_to_file(work_dir, extra_arg)

            # supply args=[extra_arg] to emcee for it to wrap it internally
            if (extra_arg is not None) and ('args' not in emcee_kwargs):
                emcee_kwargs['args'] = [extra_arg]

            nwalkers, ndim = np.array(init_points).shape
            sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=log_prob_fun,
                                            pool=slurm_pool, **emcee_kwargs)

            if init_log_prob_fun_values is None:
                # calculate the log probabilities of the init_points
                if verbosity >= 1:
                    logging.info('### explicitly calculating the log probabilities of the init_points.')
                init_log_prob_fun_values = slurm_pool.map(log_prob_fun, init_points)
            else:
                if verbosity >= 1:
                    logging.info('### setting the input init_log_prob_fun_values for the init_points.')

            # manually set the initial state
            sampler.initial_state = emcee.State(init_points, log_prob=np.array(init_log_prob_fun_values))

            # initializations
            ini_iter = 0
            time_per_iter = []

            # from here on, emcee deals itself with extra arguments by internally wrapping the log_prob_fun,
            # so we remove it from slurm_pool to avoid erroneously double wrapping it
            slurm_pool.extra_arg = None

        for curr_iter in range(ini_iter, ini_iter + num_iters):
            if verbosity >= 1 and np.mod(curr_iter, print_iter_interval) == 0:
                logging.info('### curr mcmc iter: ' + str(curr_iter))
            t_start_iter = time.time()
            state = sampler.run_mcmc(initial_state=sampler.initial_state, nsteps=1,
                                     progress=progress, skip_initial_state_check=skip_initial_state_check)
            curr_iter_time = time.time() - t_start_iter
            if verbosity >= 2 and np.mod(curr_iter, print_iter_interval) == 0:
                logging.info(f'    current iter run time: {curr_iter_time:.3f}s.')
            sampler.initial_state = state

            # mcmc status
            status = {}
            status['sampler'] = sampler
            status['slurm_pool'] = slurm_pool
            status['ini_iter'] = curr_iter + 1
            time_per_iter += [curr_iter_time]
            status['time_per_iter'] = time_per_iter

            if save_restart and np.mod(curr_iter, restart_save_interval) == 0:
                if verbosity >= 3:
                    logging.info('    saving restart file: ' + work_dir + '/' + restart_file)
                save_restart_file(status, work_dir, restart_file)
                sampler.pool = slurm_pool  # need to redefine the pool because pickling removes sampler.pool

        return status


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
