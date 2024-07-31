import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
from scipy.optimize import rosen

from slurmcmc.mcmc import slurm_mcmc

plt.close('all')

np.random.seed(0)

param_bounds = [[-5, 5], [-5, 5]]
minima = np.array([1, 1])


def log_prob(x):
    return -rosen(x)


r_constraint = 4
x0_constraint = -1
y0_constraint = -1


def constraint_fun(x):
    # return > 0 for violation
    if (x[0] - x0_constraint) ** 2 + (x[1] - y0_constraint) ** 2 > r_constraint ** 2:
        return 1
    else:
        return -1


def log_prob_with_constraint(x):
    if constraint_fun(x) > 0:
        return -1e7
    else:
        return log_prob(x)


num_params = 2
num_walkers = 20
num_iters = 200
print('num of slurmpool calls should be', num_walkers / 2 * (num_iters + 1))

# initial points chosen to satisfy constraint
init_points = (np.array([[x0_constraint, y0_constraint] for _ in range(num_walkers)])
               + (r_constraint * (np.random.rand(num_walkers, num_params) - 0.5)))

# log_prob_fun = log_prob
log_prob_fun = log_prob_with_constraint
sampler = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=init_points, num_iters=num_iters,
                     cluster='local-map',
                     )

# print('acceptance fractions:', sampler.acceptance_fraction)
print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

tau = sampler.get_autocorr_time(quiet=True)
print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(tau)))

# burnin = int(np.max(tau))
# thin = int(np.min(tau))
burnin = 0
thin = 1
samples = sampler.get_chain(discard=burnin, thin=thin)
samples_flat = sampler.get_chain(discard=burnin, thin=thin, flat=True)

# plot mcmc chains
fig, axes = plt.subplots(nrows=num_params, figsize=(10, 7), sharex=True, num=1)
labels = ["x", "y"]
color_list = cm.rainbow(np.linspace(0, 1, num_walkers))
for i in range(num_params):
    ax = axes[i]
    for j in range(num_walkers):
        ax.plot(samples[:, j, i], alpha=0.5, color=color_list[j])
    ax.set_ylabel(labels[i])
axes[-1].set_xlabel("step number")

# loss_fun 2d plot
plt.figure(2, figsize=(8, 7))
x = np.linspace(param_bounds[0][0], param_bounds[0][1], 100)
y = np.linspace(param_bounds[1][0], param_bounds[1][1], 100)
X, Y = np.meshgrid(x, y)
Z = np.nan * X
for ix, x_curr in enumerate(x):
    for iy, y_curr in enumerate(y):
        Z[iy, ix] = log_prob([x_curr, y_curr])
plt.pcolormesh(X, Y, np.log(np.abs(Z)))
plt.xlim(param_bounds[0])
plt.ylim(param_bounds[1])
plt.colorbar()

# plot points sampled during mcmc
plt.scatter(sampler.pool.points_history[:, 0], sampler.pool.points_history[:, 1], c='k', marker='o', s=10, alpha=0.5)
# plot points accepted during mcmc
plt.scatter(samples_flat[:, 0], samples_flat[:, 1], c='r', marker='o', s=10, alpha=0.5)
plt.tight_layout()

# plot analytic minima point for reference
for markeredgecolor, markeredgewidth in zip(['k', 'w'], [3, 1]):
    plt.plot(minima[0], minima[1], markersize=10, marker='*',
             markerfacecolor='none', markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth)

# plot constraint bound
theta = np.linspace(0, 2 * np.pi, 100)
plt.plot(x0_constraint + r_constraint * np.cos(theta), y0_constraint + r_constraint * np.sin(theta), color='w')
plt.tight_layout()

# emcee corner plot
import corner
fig = plt.figure(num=3, figsize=(7, 7))
corner.corner(samples_flat, labels=labels, color='k', truths=minima, truth_color='r', fig=fig,
              label_kwargs={"fontsize": 14}, labelpad=-0.1)
plt.tight_layout()
