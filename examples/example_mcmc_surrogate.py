import time

import corner
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.optimize import rosen
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from slurmcmc.mcmc import slurm_mcmc

plt.close('all')
plt.interactive(True)
plt.rcParams.update({'font.size': 12})

save_plots = False
# save_plots = True

np.random.seed(0)

param_bounds = [[-5, 5], [-5, 5]]
minima = np.array([1, 1])

param_labels = ["x", "y"]

# use_constraints = False
use_constraints = True


def log_prob_rosen(x):
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


def log_prob_rosen_with_constraint(x):
    if constraint_fun(x) > 0:
        return -1e7
    else:
        return log_prob_rosen(x)


num_params = 2
num_walkers = 10
num_iters = 2000

# initial points chosen to satisfy constraint
init_points = (np.array([[x0_constraint, y0_constraint] for _ in range(num_walkers)])
               + (r_constraint * (np.random.rand(num_walkers, num_params) - 0.5)))
# init_points = (np.array([minima for _ in range(num_walkers)])
#                + 0.5 * np.random.randn(num_walkers, num_params))

if use_constraints:
    log_prob_fun = log_prob_rosen_with_constraint
else:
    log_prob_fun = log_prob_rosen

print('running mcmc with expensive function.')
status = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=init_points, num_iters=num_iters,
                    cluster='local-map', verbosity=0)
sampler = status['sampler']
slurm_pool = status['slurm_pool']

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

## fitting a surrogate model and rerunning mcmc
num_iters_for_training = 500
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
surrogate = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6)

# assemble training data
# combine mcmc points and some "random" points for regularization in far away regions
num_iters_for_training_mcmc = int(0.5 * num_iters_for_training)
X_train, y_train = [], []
for point, value in zip(slurm_pool.points_history, slurm_pool.values_history):
    if tuple(point) in sampler.mcmc_points_set:
        X_train += [point]
        y_train += [value]
    if len(y_train) == num_iters_for_training_mcmc:
        break

while True:
    point = np.array([param_bounds[i][0] + (param_bounds[i][1] - param_bounds[i][0]) * np.random.rand()
                      for i in range(num_params)])
    if constraint_fun(point) < 0:
        X_train += [point]
        y_train += [np.array([log_prob_rosen_with_constraint(point)])]
        if len(y_train) == num_iters_for_training:
            break

X_train, y_train = np.array(X_train), np.array(y_train)

# train the surrogate model
print('starting surrogate model fit.')
time_start = time.time()
surrogate.fit(X_train, y_train)
time_end = time.time()
print('surrogate model fit done, training time=' + '{:.2f}'.format(time_end - time_start) + 's.')

log_prob_fun_surrogate = lambda x: surrogate.predict(x.reshape(1, -1))[0]  # for GP

# test the accuracy of the surrogate model on a separate test set
num_points_test = 100
X_test, y_test = [], []
for point, value in zip(slurm_pool.points_history, slurm_pool.values_history):
    if tuple(point) in sampler.mcmc_points_set and point not in X_train:
        X_test += [point]
        y_test += [value]
    if len(y_test) == num_points_test:
        break
X_test, y_test = np.array(X_test), np.array(y_test)
y_test_surrogate = np.array([log_prob_fun_surrogate(point) for point in X_test])
dy_test = y_test_surrogate - y_test[:, 0]
print('surrogate error on test set mean='
      + '{0:.4f}'.format(np.mean(dy_test)) + ', std=' + '{0:.4f}'.format(np.std(dy_test)))


# running mcmc again with surrogate
def log_prob_surrogate_with_constraint(x):
    if constraint_fun(x) > 0:
        return -1e7
    else:
        return log_prob_fun_surrogate(x)


if use_constraints:
    log_prob_fun = log_prob_surrogate_with_constraint
else:
    log_prob_fun = log_prob_fun_surrogate

print('running mcmc with surrogate.')
status_2 = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=init_points, num_iters=num_iters,
                       cluster='local-map', verbosity=0)
sampler_2 = status_2['sampler']
print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler_2.acceptance_fraction)))

tau = sampler_2.get_autocorr_time(quiet=True)
print("Autocorrelation time (tau) per paramter:", [np.round(t, 2) for t in tau])

thin = 1
# burnin = 0
burnin = int(2 * np.max(tau))

if burnin > 0:
    tau = sampler_2.get_autocorr_time(discard=burnin, thin=thin, quiet=True)
    print("Post burn-in, autocorrelation time (tau) per parameter:", [np.round(t, 2) for t in tau])

samples_2 = sampler_2.get_chain(discard=burnin, thin=thin)
samples_2_flat = sampler_2.get_chain(discard=burnin, thin=thin, flat=True)

# plot log-probability function 2d plot for visualization
x = np.linspace(param_bounds[0][0], param_bounds[0][1], 100)
y = np.linspace(param_bounds[1][0], param_bounds[1][1], 100)
X, Y = np.meshgrid(x, y)
Z = np.nan * X
Z_2 = np.nan * X
for ix, x_curr in enumerate(x):
    for iy, y_curr in enumerate(y):
        Z[iy, ix] = log_prob_rosen([x_curr, y_curr])
        Z_2[iy, ix] = log_prob_fun_surrogate(np.array([x_curr, y_curr]))

plt.figure(figsize=(16, 7))
for ind_plot, Z_curr in enumerate([Z, Z_2]):
    plt.subplot(1, 2, ind_plot + 1)
    plt.pcolormesh(X, Y, np.log(np.abs(Z_curr)), vmin=-2, vmax=10)
    plt.xlim(param_bounds[0])
    plt.ylim(param_bounds[1])
    plt.colorbar()
    plt.xlabel(param_labels[0])
    plt.ylabel(param_labels[1])
    if ind_plot == 0:
        plt.title('expensive')
    else:
        plt.title('surrogate')

    # plot training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c='r', marker='o', s=10, alpha=0.3)

    # plot constraint bound
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(x0_constraint + r_constraint * np.cos(theta), y0_constraint + r_constraint * np.sin(theta), color='w')

plt.tight_layout()

if save_plots:
    plt.savefig('example_mcmc_surrogate_2d_visualization')

# corner plot of mcmc samples in both cases
fig = plt.figure(figsize=(7, 7))
corner_fig = corner.corner(samples_flat, labels=param_labels, color='k', truths=minima, fig=fig, labelpad=-0.1,
                           hist_kwargs={"density": True})
corner.corner(samples_2_flat, labels=param_labels, color='b', truths=minima, fig=fig, labelpad=-0.1,
              hist_kwargs={"density": True}, truth_color='r')
plt.suptitle('mcmc parameters distribution')

# add "fake legend" to indicate what the colors mean
plot_dim = int(np.sqrt(len(corner_fig.axes)))
ax_legend = corner_fig.add_subplot(plot_dim, plot_dim, plot_dim)
ax_legend.set_xlim(0, 1)
ax_legend.set_ylim(0, 1)
ax_legend.axis('off')  # Turn off axis for clean look
patch1_params = {'x': 0.2, 'y': 0.5, 'width': 0.15, 'height': 0.03, 'color': 'k', 'label': 'expensive'}
patch2_params = {'x': 0.2, 'y': 0.4, 'width': 0.15, 'height': 0.03, 'color': 'b', 'label': 'surrogate'}
for p in [patch1_params, patch2_params]:
    patch = Rectangle((p['x'], p['y']), p['width'], p['height'], color=p['color'])
    ax_legend.add_patch(patch)
    ax_legend.text(p['x'] + 1.5 * p['width'], p['y'], p['label'])
plt.tight_layout()

if save_plots:
    plt.savefig('example_mcmc_surrogate_parameters_distribution')

# plot the probability ratios (importance weights) between expensive and surrogate
num_validation_points = 1000
log_prob_expensive_list = []
log_prob_surrogate_list = []
importance_weights_list = []
for i in range(num_validation_points):
    point = samples_2_flat[i]
    log_prob_expensive_list += [log_prob_rosen(point)]
    log_prob_surrogate_list += [log_prob_fun_surrogate(point)]
    importance_weights_list += [np.exp(log_prob_expensive_list[-1] - log_prob_surrogate_list[-1])]
importance_mean = np.mean(importance_weights_list)
importance_std = np.std(importance_weights_list)
print('importance_weights mean', importance_mean, ', std=', importance_std)

plt.figure(figsize=(5, 8))
plt.subplot(2, 1, 1)
hist_range = (importance_mean - 3 * importance_std, importance_mean + 3 * importance_std)
plt.hist(importance_weights_list, color='b', bins=20, range=hist_range)
plt.title('mean=' + '{:.3f}'.format(importance_mean) + ', std=' + '{:.3f}'.format(importance_std))
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])

plt.subplot(2, 1, 2)
plt.scatter(importance_weights_list, log_prob_expensive_list, color='b', alpha=0.3)
plt.xlabel('importance weights')
plt.ylabel('log probability (expensive)')
plt.xlim([hist_range[0], hist_range[1]])

plt.tight_layout()

if save_plots:
    plt.savefig('example_mcmc_surrogate_importance_weights')
