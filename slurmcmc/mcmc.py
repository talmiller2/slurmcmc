import logging

import emcee
import numpy as np

from slurmcmc.general_utils import set_logging, save_restart_file, load_restart_file, save_extra_arg_to_file
from slurmcmc.slurm_utils import SlurmPool


def slurm_mcmc(log_prob_fun, init_points, num_iters=10, progress=False,
               verbosity=1, slurm_vebosity=0, log_file=None, extra_arg=None,
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
            initial_state = status['state']
            sampler = status['sampler']
            slurm_pool = status['slurm_pool']
            sampler.pool = slurm_pool
            ini_iter = status['ini_iter']

    else:
        # using extra_arg=None because emcee deals with extra_arg internally by wrapping the function
        slurm_pool = SlurmPool(work_dir, job_name, cluster, verbosity=slurm_vebosity, extra_arg=None, **slurm_dict)

        # save the extra_arg in the work folder to document the full input used
        if cluster != 'local-map':
            save_extra_arg_to_file(work_dir, extra_arg)

        # supply args=[extra_arg] to emcee for it to wrap it internally
        if (extra_arg is not None) and ('args' not in emcee_dict):
            emcee_dict['args'] = [extra_arg]

        nwalkers, ndim = np.array(init_points).shape
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=log_prob_fun,
                                        pool=slurm_pool, **emcee_dict)
        initial_state = init_points
        ini_iter = 0

    for curr_iter in range(ini_iter, ini_iter + num_iters):
        if verbosity >= 1:
            logging.info('### curr mcmc iter: ' + str(curr_iter))
        state = sampler.run_mcmc(initial_state=initial_state, nsteps=1, progress=progress)
        initial_state = state

        if save_restart:
            if verbosity >= 3:
                logging.info('    saving restart file: ' + work_dir + '/' + restart_file)
            status = {'state': state, 'sampler': sampler, 'slurm_pool': slurm_pool, 'ini_iter': curr_iter + 1}
            save_restart_file(status, work_dir, restart_file)
            sampler.pool = slurm_pool  # need to redefine the pool becuase pickling removes sampler.pool

    return sampler
