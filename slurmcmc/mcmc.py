import emcee
import numpy as np

from slurmcmc.slurm_utils import SlurmPool

def slurm_mcmc(log_prob_fun, init_points, num_iters=10, progress=True,
               work_dir='tmp', job_name='minimize', cluster='local', slurm_dict={}, **emcee_kwargs):
    """
    combine submitit + emcee to allow ensemble mcmc on slurm.
    """
    # TODO: allow to restart from file

    slurm_pool = SlurmPool(work_dir, job_name, cluster, **slurm_dict)
    nwalkers, ndim = np.array(init_points).shape
    sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=log_prob_fun,
                                    pool=slurm_pool, **emcee_kwargs)

    sampler.run_mcmc(initial_state=init_points, nsteps=num_iters, progress=progress)
    # initial_state = init_points
    # for num_iter in range(num_iters):
    #     state = sampler.run_mcmc(initial_state=initial_state, nsteps=1, progress=progress)
    #     initial_state = state

    return sampler, slurm_pool
