import time

import matplotlib.pyplot as plt
import nevergrad as ng
import numpy as np
import torch
from matplotlib.pyplot import cm
from scipy.optimize import rosen

from slurmcmc.optimization import slurm_minimize

torch.manual_seed(0)

plt.close('all')
plt.interactive(True)
plt.rcParams.update({'font.size': 12})

save_plots = False
# save_plots = True

num_params = 2
# num_params = 5
# num_params = 10
param_bounds = [[-10, 10] for _ in range(num_params)]


def loss_fun(x):
    return rosen(x)

num_workers = 50
num_iters = 50

## define the optimization algorithm
optimizer_packages, optimizer_classes, optimizer_labels, opt_run_times, loss_curves = [], [], [], [], []

# in the nevergrad package, the parallelizable alagorithms are Differential-Evolution and Particle-Swarm-Optimization
# (see https://facebookresearch.github.io/nevergrad/machinelearning.html)
optimizer_package = 'nevergrad'
optimizer_class = ng.optimizers.DifferentialEvolution(crossover="twopoints", popsize=num_workers)
optimizer_packages += [optimizer_package]
optimizer_classes += [optimizer_class]
optimizer_labels += ['nevergrad DE']

optimizer_package = 'nevergrad'
optimizer_class = ng.optimizers.ConfPSO(popsize=num_workers)
optimizer_packages += [optimizer_package]
optimizer_classes += [optimizer_class]
optimizer_labels += ['nevergrad PSO']

# # the nevergrad package also supports baysian-optimizaiton algorithms but that are parallelizable
# # in the botorch package there are bayesian-optimization algorithms that are parallelizable
# # (see https://botorch.org/tutorials/closed_loop_botorch_only)
optimizer_package = 'botorch'
optimizer_class = None
optimizer_packages += [optimizer_package]
optimizer_classes += [optimizer_class]
optimizer_labels += ['botorch BO']

colors = ['b', 'g', 'r']

# num_seeds = 1
num_seeds = 3
linestyles = ['-', '--', ':']

for ind_seed in range(num_seeds):
    print('$$$$$$$$$$ ind_seed=' + str(ind_seed))
    np.random.seed(ind_seed * 2000)

    # define init points same for all optimizers
    init_points = np.random.rand(num_workers, num_params)
    for i in range(num_params):
        init_points[:, i] = param_bounds[i][0] + (param_bounds[i][1] - param_bounds[i][0]) * init_points[:, i]

    # loop over different optimizers
    for ind_opt in range(len(optimizer_packages)):
        optimizer_package = optimizer_packages[ind_opt]
        optimizer_class = optimizer_classes[ind_opt]
        optimizer_label = optimizer_labels[ind_opt]
        print('@@@@@@ ' + optimizer_label)

        time_start = time.time()
        result = slurm_minimize(
            loss_fun=loss_fun,
            param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
            init_points=init_points,
            cluster='local-map',
            verbosity=3,
            optimizer_package=optimizer_package,
            optimizer_class=optimizer_class,
            botorch_kwargs=None,
            # botorch_kwargs={'num_restarts': 10, 'raw_samples': 32, 'options': {"maxiter": 100}, 'num_best_points': 1000}, # trying different BO options
        )
        time_end = time.time()
        opt_run_time = time_end - time_start
        opt_run_times += [opt_run_time]
        print('opt calculation time:', opt_run_time, 'seconds.')
        print('num_loss_fun_calls_total:', result['num_loss_fun_calls_total'])
        print('num_constraint_fun_calls_total:', result['num_constraint_fun_calls_total'])
        print('num_asks_total:', result['num_asks_total'])

        # plot the progression of the loss function with optimization iterations
        loss_history = result['slurm_pool'].values_history[:, 0]
        point_num_array = [i for i in range(len(loss_history))]
        loss_min_all_iter = result['loss_min_all_iter']
        loss_curves += [result['loss_min_all_iter']]
        loss_min_per_iter = result['loss_min_per_iter']
        num_workers_per_iter = result['num_workers_per_iter']
        iter_num_array = np.cumsum(result['num_workers_per_iter'])

# plot loss for all optimization sets
cnt = -1
for ind_seed in range(num_seeds):
    for ind_opt in range(len(optimizer_packages)):
        cnt += 1
        plt.figure(2, figsize=(8, 5))
        if ind_seed == 0:
            label = optimizer_labels[ind_opt] + f' [T={opt_run_times[cnt]:.2f}s]'
        else:
            label = None
        plt.scatter(1, loss_curves[cnt][0],
                    facecolors='none',  # no fill
                    edgecolors='black',
                    linewidth=2,
                 )
        plt.plot(np.arange(1, len(loss_curves[cnt]) + 1), loss_curves[cnt],
                 color=colors[ind_opt], linestyle=linestyles[ind_seed],
                 label=label)

plt.yscale('log')
plt.xlabel('# iter')
plt.ylabel('loss')
plt.title('evolution of the loss for different optimization algorithms')
plt.grid(True)
plt.legend()
plt.tight_layout()

if save_plots:
    plt.savefig('example_optimization_algorithms_comparison_' + str(num_params) + 'd')
