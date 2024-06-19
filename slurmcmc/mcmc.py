import emcee
import numpy as np

from slurmcmc.slurm_utils import SlurmPool, print_log

def slurm_mcmc(log_prob_fun, init_points, num_iters=10, progress=True,
               verbosity=1, slurm_vebosity=0, log_file=None,
               work_dir='tmp', job_name='minimize', cluster='slurm', slurm_dict={}, **emcee_kwargs):
    """
    combine submitit + emcee to allow ensemble mcmc on slurm.
    the number of parallelizable evaluations in the default emcee "move" is len(init_points)/2,
    except the first one on the init_points which is len(init_points).

    # TODO: starting for restart is possible using backend, but need to also save slurmpool to keep track
       on the runs structure. or maybe it already just works, anyway needs testing.
    """

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

    return sampler, slurm_pool
