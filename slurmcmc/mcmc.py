import emcee
import numpy as np
from slurmcmc.general_utils import print_log, save_restart_file, load_restart_file
from slurmcmc.slurm_utils import SlurmPool


def slurm_mcmc(log_prob_fun, init_points, num_iters=10, progress=True,
               verbosity=1, slurm_vebosity=0, log_file=None,
               save_restart=False, load_restart=False, restart_file='mcmc_restart.pkl',
               work_dir='tmp', job_name='mcmc', cluster='slurm', slurm_dict={}, **emcee_kwargs):
    """
    combine submitit + emcee to allow ensemble mcmc on slurm.
    the number of parallelizable evaluations in the default emcee "move" is len(init_points)/2,
    except the first one on the init_points which is len(init_points).
    """
    if load_restart:
        if verbosity >= 1:
            print_log('loading restart file.', work_dir, log_file)
            status = load_restart_file(work_dir, restart_file)
            initial_state = status['state']
            sampler = status['sampler']
            slurm_pool = status['slurm_pool']
            sampler.pool = slurm_pool

    else:
        slurm_pool = SlurmPool(work_dir, job_name, cluster, verbosity=slurm_vebosity, **slurm_dict)
        nwalkers, ndim = np.array(init_points).shape
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=log_prob_fun,
                                        pool=slurm_pool, **emcee_kwargs)
        initial_state = init_points

    for curr_iter in range(num_iters):
        if verbosity >= 1:
            print_log('### curr mcmc iter: ' + str(curr_iter), work_dir, log_file)
        state = sampler.run_mcmc(initial_state=initial_state, nsteps=1, progress=progress)
        initial_state = state

        if save_restart:
            if verbosity >= 1:
                print_log('saving restart.', work_dir, log_file)
            status = {'state': state, 'sampler': sampler, 'slurm_pool': slurm_pool}
            save_restart_file(status, work_dir, restart_file)
            sampler.pool = slurm_pool  # need to redefine the pool becuase pickling removes sampler.pool

    return sampler
