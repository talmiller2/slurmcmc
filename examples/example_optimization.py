import matplotlib.pyplot as plt
import nevergrad as ng
import numpy as np
import torch
import time

from matplotlib.pyplot import cm
from scipy.optimize import rosen
from slurmcmc.optimization import slurm_minimize

np.random.seed(0)
torch.manual_seed(0)

plt.close('all')

num_params = 2
param_bounds = [[-5, 5], [-5, 5]]


def loss_fun(x):
    return rosen(x)


minima = [1, 1]  # rosen

r_constraint = 4
x0_constraint = -1
y0_constraint = -1


def constraint_fun(x):
    # return > 0 for violation
    if (x[0] - x0_constraint) ** 2 + (x[1] - y0_constraint) ** 2 > r_constraint ** 2:
        return 1
    else:
        return -1


def loss_fun_with_constraint(x):
    if constraint_fun(x) > 0:
        return 1e7
    else:
        return loss_fun(x)

num_workers = 10
num_iters = 20

## define the optimization algorithm

# in the nevergrad package, the parallelizable alagorithms are Differential-Evolution and Particle-Swarm-Optimization
# (see https://facebookresearch.github.io/nevergrad/machinelearning.html)
optimizer_package = 'nevergrad'
optimizer_class = ng.optimizers.DifferentialEvolution(crossover="twopoints", popsize=num_workers)
# optimizer_class = ng.optimizers.ConfPSO(popsize=num_workers)

# the nevergrad package also supports baysian-optimizaiton algorithms but that are parallelizable
# in the botorch package there are bayesian-optimization algorithms that are parallelizable
# (see https://botorch.org/tutorials/closed_loop_botorch_only)
# optimizer_package = 'botorch'
# optimizer_class = None
# botorch_kwargs = {}
# botorch_kwargs = {'num_restarts': 5, 'raw_samples': 5, 'num_best_points': None}

time_start = time.time()
result = slurm_minimize(
    loss_fun=loss_fun,
    # loss_fun=loss_fun_with_constraint,
    constraint_fun=constraint_fun,
    param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
    cluster='local-map',
    verbosity=3,
    optimizer_package=optimizer_package,
    optimizer_class=optimizer_class,
    # botorch_kwargs=botorch_kwargs,
)
time_end = time.time()

print('opt calculation time:', time_end - time_start, 'seconds.')
print('num_loss_fun_calls_total:', result['num_loss_fun_calls_total'])
print('num_constraint_fun_calls_total:', result['num_constraint_fun_calls_total'])
print('num_asks_total:', result['num_asks_total'])

# loss_fun 2d plot
plt.figure(1, figsize=(8, 7))
x = np.linspace(param_bounds[0][0], param_bounds[0][1], 100)
y = np.linspace(param_bounds[1][0], param_bounds[1][1], 100)
X, Y = np.meshgrid(x, y)
Z = np.nan * X
for ix, x_curr in enumerate(x):
    for iy, y_curr in enumerate(y):
        Z[iy, ix] = loss_fun([x_curr, y_curr])
plt.pcolormesh(X, Y, np.log(np.abs(Z)))
plt.xlim(param_bounds[0])
plt.ylim(param_bounds[1])
plt.colorbar()

# plot points sampled during optimization
history = result['slurm_pool'].points_history
color_list = cm.magma(np.linspace(0, 1, len(history)))
for ind_iter, pos in enumerate(history):
    plt.plot(pos[0], pos[1], marker='o', markersize=4, color=color_list[ind_iter], linewidth=0)

# plot analytic minima point for reference
for markeredgecolor, markeredgewidth in zip(['k', 'w'], [3, 1]):
    plt.plot(minima[0], minima[1], markersize=10, marker='*',
             markerfacecolor='none', markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth)

# plot constraint bound
theta = np.linspace(0, 2 * np.pi, 100)
plt.plot(x0_constraint + r_constraint * np.cos(theta), y0_constraint + r_constraint * np.sin(theta), color='w')
plt.tight_layout()

# plot the progression of the loss function with optimization iterations
loss_history = result['slurm_pool'].values_history[:, 0]
point_num_array = [i for i in range(len(loss_history))]
loss_min_iters = result['loss_min_iters']
loss_per_iters = result['loss_per_iters']
iter_num_array = [(i + 1) * num_workers for i in range(len(loss_min_iters))]
iter_num_array = [(i + 1) * num_workers for i in range(len(loss_min_iters))]

plt.figure(2, figsize=(8, 5))
plt.plot(point_num_array, loss_history, '.b', label='all samples')
plt.plot(iter_num_array, loss_per_iters, '-og', label='iteration min')
plt.plot(iter_num_array, loss_min_iters, '-or', label='best min')
plt.yscale('log')
plt.xlabel('# sample')
plt.ylabel('loss')
plt.grid(True)
plt.legend()
plt.tight_layout()