import corner
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
from scipy.optimize import rosen

from slurmcmc.mcmc import slurm_mcmc, get_gelman_rubin_statistic

plt.close('all')
plt.interactive(True)
plt.rcParams.update({'font.size': 12})

save_plots = False
# save_plots = True

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
num_walkers = 10
num_iters = 10000

# initial points chosen to satisfy constraint
init_points = (np.array([[x0_constraint, y0_constraint] for _ in range(num_walkers)])
               + (r_constraint * (np.random.rand(num_walkers, num_params) - 0.5)))
# init_points = (np.array([minima for _ in range(num_walkers)])
#                + 0.5 * np.random.randn(num_walkers, num_params))

# log_prob_fun = log_prob
log_prob_fun = log_prob_with_constraint
sampler = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=init_points, num_iters=num_iters,
                     cluster='local-map', verbosity=0)

print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

tau = sampler.get_autocorr_time(quiet=True)
print("Autocorrelation time (tau) per paramter:", [np.round(t, 2) for t in tau])

thin = 1
# burnin = 0
burnin = int(2 * np.max(tau))

if burnin > 0:
    tau = sampler.get_autocorr_time(discard=burnin, thin=thin, quiet=True)
    print("Post burn-in, autocorrelation time (tau) per parameter:", [np.round(t, 2) for t in tau])

samples = sampler.get_chain(discard=burnin, thin=thin)
samples_flat = sampler.get_chain(discard=burnin, thin=thin, flat=True)

# Effective Sample Size (ESS)
num_steps = samples.shape[0]
ESS = num_steps / tau
print("Effective Sample Size (ESS) per parameter:", [np.round(e, 2) for e in ESS])

# Gelman-Rubin statistic
GR_statistic = get_gelman_rubin_statistic(samples)
print('Gelman-Rubin statistic per parameter:', [np.round(g, 3) for g in GR_statistic])

# plot mcmc chains evolution
fig, axes = plt.subplots(nrows=num_params, figsize=(10, 7), sharex=True)
labels = ["x", "y"]
color_list = cm.rainbow(np.linspace(0, 1, num_walkers))
for i in range(num_params):
    ax = axes[i]
    for j in range(num_walkers):
        if j == 0:  # add MCMC statistics information to the plots
            label = '$\\tau$={0:d}, ESS={1:.2f}, GR={2:.3f}'.format(int(tau[i]), ESS[i], GR_statistic[i])
        else:
            label = None
        ax.plot(samples[:, j, i], alpha=0.5, color=color_list[j], label=label)
    ax.set_ylabel(labels[i])
    ax.legend(loc='lower right')

axes[-1].set_xlabel('# iteration')
axes[0].set_title('evolution of the parameters in different mcmc chains')
plt.tight_layout()

if save_plots:
    plt.savefig('example_mcmc_chains_progress')

# loss_fun 2d plot
plt.figure(figsize=(8, 7))
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
plt.xlabel('x')
plt.ylabel('y')
plt.title('log probability function: rosenbrock')

# plot points sampled during mcmc
plt.scatter(sampler.pool.points_history[:, 0], sampler.pool.points_history[:, 1], c='k', marker='o', s=10, alpha=0.1)
# plot points accepted during mcmc
plt.scatter(samples_flat[:, 0], samples_flat[:, 1], c='r', marker='o', s=10, alpha=0.1)
plt.tight_layout()

# plot analytic minima point for reference
for markeredgecolor, markeredgewidth in zip(['k', 'w'], [3, 1]):
    plt.plot(minima[0], minima[1], markersize=10, marker='*',
             markerfacecolor='none', markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth)

# plot constraint bound
theta = np.linspace(0, 2 * np.pi, 100)
plt.plot(x0_constraint + r_constraint * np.cos(theta), y0_constraint + r_constraint * np.sin(theta), color='w')
plt.tight_layout()

if save_plots:
    plt.savefig('example_mcmc_2d_visualization')

# emcee corner plot
fig = plt.figure(figsize=(7, 7))
corner.corner(samples_flat, labels=labels, color='k', truths=minima, truth_color='r', fig=fig, labelpad=-0.1)
plt.suptitle('mcmc parameters distribution')
plt.tight_layout()

if save_plots:
    plt.savefig('example_mcmc_parameters_distribution')
