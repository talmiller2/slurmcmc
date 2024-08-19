import logging

import emcee
import numpy as np

from slurmcmc.general_utils import (set_logging, save_restart_file, load_restart_file, save_extra_arg_to_file,
                                    point_to_tuple)
from slurmcmc.slurm_utils import SlurmPool


def slurm_mcmc(log_prob_fun, init_points, num_iters=10, init_log_prob_fun_values=None,
               progress=False, verbosity=1, slurm_vebosity=0, log_file=None, extra_arg=None,
               save_restart=False, load_restart=False, restart_file='mcmc_restart.pkl',
               work_dir='tmp', job_name='mcmc', cluster='slurm', slurm_dict={}, emcee_dict={}):
    """
    combine submitit + emcee to allow ensemble mcmc on slurm.
    the number of parallelizable evaluations in the default emcee "move" is len(init_points)/2,
    except the first one on the init_points which is len(init_points).
    """
    set_logging(work_dir, log_file)

    if load_restart:
        if verbosity >= 1:
            logging.info('loading restart file: ' + work_dir + '/' + restart_file)

            status = load_restart_file(work_dir, restart_file)
            sampler = status['sampler']
            slurm_pool = status['slurm_pool']
            sampler.pool = slurm_pool
            ini_iter = status['ini_iter']
    else:
        # using extra_arg=None because emcee deals with extra_arg internally by wrapping the function
        slurm_pool = SlurmPool(work_dir, job_name, cluster, verbosity=slurm_vebosity, extra_arg=extra_arg, **slurm_dict)

        # save the extra_arg in the work folder to document the full input used
        if cluster != 'local-map':
            save_extra_arg_to_file(work_dir, extra_arg)

        # supply args=[extra_arg] to emcee for it to wrap it internally
        if (extra_arg is not None) and ('args' not in emcee_dict):
            emcee_dict['args'] = [extra_arg]

        nwalkers, ndim = np.array(init_points).shape
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=log_prob_fun,
                                        pool=slurm_pool, **emcee_dict)

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
        sampler.mcmc_points_set = set()
        sampler.points_weights_dict = {}

        # from here on, emcee deals itself with extra arguments by internally wrapping the log_prob_fun,
        # so we remove it from slurm_pool to avoid erroneously double wrapping it
        slurm_pool.extra_arg = None

    for curr_iter in range(ini_iter, ini_iter + num_iters):
        if verbosity >= 1:
            logging.info('### curr mcmc iter: ' + str(curr_iter))
        state = sampler.run_mcmc(initial_state=sampler.initial_state, nsteps=1, progress=progress)
        sampler.initial_state = state

        # track the points in the mcmc set and their weights
        for point in state.coords:
            point_tuple = point_to_tuple(point)
            if point_tuple not in sampler.mcmc_points_set:
                sampler.mcmc_points_set.add(point_tuple)
                sampler.points_weights_dict[point_tuple] = 1  # initialize
            else:
                sampler.points_weights_dict[point_tuple] += 1

        if save_restart:
            if verbosity >= 3:
                logging.info('    saving restart file: ' + work_dir + '/' + restart_file)
            status = {'state': state, 'sampler': sampler, 'slurm_pool': slurm_pool, 'ini_iter': curr_iter + 1}
            save_restart_file(status, work_dir, restart_file)
            sampler.pool = slurm_pool  # need to redefine the pool because pickling removes sampler.pool

    return sampler


def get_gelman_rubin_statistic(chains):
    """
    Calculate the Gelman-Rubin statistic for MCMC chains.
    Not really relevant for correlated chains as in the emcee algorithm, for more read:
    https://emcee.readthedocs.io/en/stable/tutorials/autocorr/

    Parameters:
    chains: np.ndarray of shape (nwalkers, nsteps, ndim)
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
