import time

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.pyplot import cm
from scipy.optimize import rosen

from slurmcmc.mcmc import slurm_mcmc, get_gelman_rubin_statistic

plt.close('all')
plt.interactive(True)
plt.rcParams.update({'font.size': 12})

save_plots = False
# save_plots = True

np.random.seed(0)

num_params = 5
num_walkers = num_params * 10
num_iters = 50000

minima = np.ones(num_params)

grid_ranges = [(-0.5, 1.5), (-0.5, 1.5), (-0.5, 2.0), (-0.5, 2.5), (-0.5, 4.5)]


def log_prob(x):
    return -rosen(x)


# initial points chosen around known minima
init_points = []
for i in range(num_walkers):
    init_point = minima + 1e-2 * np.random.randn(num_params)
    init_points += [init_point]
init_points = np.array(init_points)

# using emcee backend enables to run long chains faster, without it the time per mcmc iter increases
backend = emcee.backends.HDFBackend("emcee_backend_example.h5")

t_slurm_mcmc_start = time.time()
status = slurm_mcmc(log_prob_fun=log_prob, init_points=init_points, num_iters=num_iters,
                    cluster='local-map', verbosity=2,
                    record_history=False,
                    # status_restart=status,
                    emcee_kwargs={'backend': backend},
                    print_iter_interval=100,  # avoid printing status every single iteration
                    )
slurm_mcmc_run_time = time.time() - t_slurm_mcmc_start
print(f'slurm_mcmc_run_time: {slurm_mcmc_run_time:.3f}s.')

plt.figure(figsize=(8, 6))
plt.scatter(range(len(status['time_per_iter'])), status['time_per_iter'], color='b', alpha=0.5)
plt.xlabel('# mcmc iteration')
plt.ylabel('run time per iteration [s]')
plt.grid(True)
plt.tight_layout()

sampler = status['sampler']
slurm_pool = status['slurm_pool']

print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

tau = sampler.get_autocorr_time(quiet=True)
print("Autocorrelation time (tau) per paramter:", [np.round(t, 2) for t in tau])

thin = 1
burnin = 0
# burnin = int(2 * np.max(tau))

if burnin > 0:
    tau = sampler.get_autocorr_time(discard=burnin, thin=thin, quiet=True)
    print("Post burn-in, autocorrelation time (tau) per parameter:", [np.round(t, 2) for t in tau])

samples = sampler.get_chain(discard=burnin, thin=thin)
samples_flat = sampler.get_chain(discard=burnin, thin=thin, flat=True)
print('len(samples_flat) = ', len(samples_flat))
num_MCMC_samples = len(samples_flat)

# plot the parameter distributions for different iterations, to see how the distrbutions converge.
param_labels = [f'$x_{i + 1}$' for i in range(num_params)]
fig = plt.figure(figsize=(9, 9))
bins = 40
num_splits = 5
num_iters_interval = int(samples.shape[0] / num_splits)
max_iters_list = np.arange(num_iters_interval, samples.shape[0] + 1, num_iters_interval)
color_list = cm.rainbow(np.linspace(0, 1, num_splits))
num_subsamples = int(samples_flat.shape[0] / num_splits)
for ind_n, n in enumerate(max_iters_list):
    samples_flat_subset = samples[:n].reshape(-1, samples.shape[-1])
    random_indices = np.random.permutation(samples_flat_subset.shape[0])[0:num_subsamples]
    samples_flat_subset = samples_flat_subset[random_indices, :]
    corner.corner(samples_flat_subset, color=color_list[ind_n], fig=fig, bins=bins,
                  plot_datapoints=False, plot_density=False, range=grid_ranges,
                  )
axes = np.array(fig.axes).reshape((num_params, num_params))
for i in range(num_params):
    axes[i, i].set_title(param_labels[i], pad=10)
fig.suptitle('mcmc parameters distribution (different iteration stops)')
fig.subplots_adjust(wspace=0.1, hspace=0.1, left=0.05, right=0.95, bottom=0.05, top=0.9)

# add "fake legend" to indicate what the colors mean
legend_elements = [Line2D([0], [0], color=color_list[i], lw=1.5, label=f"MCMC, $L_c={max_iters_list[i]}$") for i in
                   range(num_splits)]

# Place small horizontal legend centered above the whole figure
fig.legend(
    handles=legend_elements,
    loc="upper center",  # try also: "upper left", "upper right"
    bbox_to_anchor=(0.7, 0.9),  # y = 0.96–1.02 usually works well
    frameon=True,  # or False for cleaner look
    handlelength=1.2,
    handletextpad=0.6,
)

if save_plots:
    plt.savefig('example_mcmc_' + str(num_params) + 'd_parameters_distribution_convergence')

# plot metrics for mcmc convergence
print('Calculating metrics convergence')
metrics = ['tau', 'tau_multiples', 'ESS', 'GR']
metric_labels = ['$\\tau$', '$L_c/\\tau$', '$ESS$', '$GR$']
diagnostics = {}
num_iters_interval = 2000
nun_iters_list = np.arange(num_iters_interval, num_iters + 1, num_iters_interval)
for metric in metrics:
    print('metric:', metric)
    diagnostics[metric] = []
    for ind_n, n in enumerate(nun_iters_list):
        print(f"{n}/{nun_iters_list[-1]}")
        if metric == 'tau':
            diagnostics[metric] += [emcee.autocorr.integrated_time(samples[:n], quiet=True)]
        elif metric == 'tau_multiples':
            diagnostics[metric] += [n / diagnostics['tau'][ind_n]]
        elif metric == 'ESS':
            diagnostics[metric] += [num_walkers * diagnostics['tau_multiples'][ind_n]]
        elif metric == 'GR':
            diagnostics[metric] += [get_gelman_rubin_statistic(samples[:n])]
    diagnostics[metric] = np.array(diagnostics[metric])

plt.figure(figsize=(9, 7))
color_list = cm.rainbow(np.linspace(0, 1, num_params))
for ind_metric, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
    plt.subplot(len(metrics), 1, ind_metric + 1)
    for ind_param, param_label in enumerate(param_labels):
        plt.plot(nun_iters_list, diagnostics[metric][:, ind_param], '-o',
                 label=param_label, color=color_list[ind_param])
        plt.ylabel(metric_label)
        plt.grid(True)
        if ind_metric == len(metrics) - 1:
            plt.legend(ncol=3)
        else:
            plt.gca().set_xticklabels([])
plt.xlabel('# iteration')
plt.suptitle('convergence diagnostics')
plt.tight_layout()

if save_plots:
    plt.savefig('example_mcmc_' + str(num_params) + 'd_convergence_diagnostics')

## calculate and plot the probability distribution using brute-force monte-carlo
print('Beginning brute-force Monte-Carlo sampling')
t_grid_mc_start = time.time()

# Create linear axes for each parameter
steps_per_dim = 40
axes_list = [np.linspace(b[0], b[1], steps_per_dim) for b in grid_ranges]

# Create the coordinate grid, 'ij' indexing is crucial for keeping dimensions aligned with the input list
grid = np.meshgrid(*axes_list, indexing='ij')

# Reshape for vectorized evaluation
# This creates a (TotalPoints, NumParams) array
flat_grid = np.stack([g.ravel() for g in grid], axis=-1)
num_MC_samples = len(flat_grid)

# Evaluate log-probabilities
log_probs = np.zeros(num_MC_samples)
for i, p in enumerate(flat_grid):
    if np.mod(i, int(num_MC_samples / 100)) == 0:
        print(f'{i}/{num_MC_samples} = {i / num_MC_samples * 100:g}%')
    log_probs[i] = log_prob(p)

# Convert to Probabilities
# Subtract max log_prob for numerical stability (prevents overflow)
probs = np.exp(log_probs - np.max(log_probs))

# Normalize to create a valid PMF
probs /= np.sum(probs)

# Reshape back to the grid shape
probs_grid = probs.reshape([steps_per_dim] * num_params)

grid_mc_run_time = time.time() - t_grid_mc_start
print(f'grid_mc_run_time: {grid_mc_run_time:.3f}s.')

# coarse MC with an approximately equivalent number of samples of MCMC, based on the samples already made.
steps_per_dim_coarse = 19  # The reduced steps per dimension

coarse_axes_list = [np.linspace(b[0], b[1], steps_per_dim_coarse) for b in grid_ranges]

# Compute the subsampled indices (approximating the positions in the coarser grid)
indices = np.round(np.linspace(0, steps_per_dim - 1, steps_per_dim_coarse)).astype(int)

# Reshape log_probs into a grid for multidimensional indexing
log_probs_grid = log_probs.reshape([steps_per_dim] * num_params)

# Subsample using the indices for each dimension
indices_list = [indices] * num_params
coarse_log_probs_grid = log_probs_grid[np.ix_(*indices_list)]

# Flatten and compute probs as in the original code
coarse_log_probs = coarse_log_probs_grid.ravel()
coarse_probs = np.exp(coarse_log_probs - np.max(coarse_log_probs))
coarse_probs /= np.sum(coarse_probs)
num_MC_samples_coarse = len(coarse_probs)

# Reshape back to the coarse grid shape
coarse_probs_grid = coarse_probs.reshape([steps_per_dim_coarse] * num_params)

## corner plot of mcmc samples
fig = plt.figure(figsize=(9, 9))
corner.corner(samples_flat, color='k',
              fig=fig, bins=bins,
              plot_datapoints=False, plot_density=False, range=grid_ranges,
              )
axes = np.array(fig.axes).reshape((num_params, num_params))
for i in range(num_params):
    axes[i, i].set_title(param_labels[i], pad=10)
fig.suptitle('mcmc (and mc) parameters distribution')
fig.subplots_adjust(wspace=0.1, hspace=0.1, left=0.05, right=0.95, bottom=0.05, top=0.9)

colors_MC = ['b', 'r']
linewidths_MC = [2, 1]
linestyles_MC = ['-', '--']
for curr_probs_grid, curr_axes_list, color, linewidth, linestyle in zip([probs_grid, coarse_probs_grid],
                                                                        [axes_list, coarse_axes_list], colors_MC,
                                                                        linewidths_MC, linestyles_MC):

    axes = np.array(fig.axes).reshape((num_params, num_params))
    for i in range(num_params):
        for j in range(i + 1):
            ax = axes[i, j]

            # --- Diagonal: 1D Marginals ---
            if i == j:
                # Sum over all axes EXCEPT the current one
                marginal_1d = curr_probs_grid
                for axis_idx in reversed(range(num_params)):
                    if axis_idx != i:
                        marginal_1d = np.sum(marginal_1d, axis=axis_idx)

                # Normalize as a PDF:
                # The integral of the curve must be 1.0
                dx = curr_axes_list[i][1] - curr_axes_list[i][0]  # Assuming uniform grid
                norm_factor = np.sum(marginal_1d) * dx
                marginal_1d = marginal_1d / norm_factor

                # Normalize the height to match the max value of the mcmc historams
                patches = ax.patches
                poly = ax.patches[0]  # the single Polygon
                verts = poly.get_xy()  # (N,2) array — x,y vertices
                y_values = verts[:, 1]  # all y-coordinates of the outline
                marginal_1d /= np.mean(marginal_1d)
                marginal_1d *= y_values.mean()
                ax.plot(curr_axes_list[i], marginal_1d, color=color, lw=linewidth, linestyle=linestyle)

            # --- Off-diagonal: 2D Marginals ---
            else:
                # Sum over all axes EXCEPT i and j
                marginal_2d = curr_probs_grid
                for axis_idx in reversed(range(num_params)):
                    if axis_idx != i and axis_idx != j:
                        marginal_2d = np.sum(marginal_2d, axis=axis_idx)

                # Transpose if necessary to match (x, y) orientation
                # Corner plots: j is x-axis (column), i is y-axis (row)
                Z = marginal_2d.T
                X, Y = np.meshgrid(curr_axes_list[j], curr_axes_list[i])

                # Overlay contours
                ax.contour(X, Y, Z, colors=color, levels=3, linewidths=linewidth, linestyles=linestyle)


def to_latex_sci_notation(x, precision=2):
    if x == 0:
        return "0"

    import math
    sign = "-" if x < 0 else ""
    x = abs(x)

    if x == 0:  # avoid log(0)
        return "0"

    exp = math.floor(math.log10(x))
    mantissa = x / (10 ** exp)

    # Round
    mantissa_rounded = round(mantissa, precision)
    if mantissa_rounded >= 10:
        mantissa_rounded /= 10
        exp += 1

    # Format mantissa nicely
    if precision == 0:
        mant_str = str(int(mantissa_rounded))
    else:
        mant_str = f"{mantissa_rounded:.{precision}f}".rstrip("0").rstrip(".")
        if not mant_str:  # in case it was 1.000 → ""
            mant_str = "1"

    # LaTeX – use double backslash or raw string
    if exp == 0:
        return f"{sign}{mant_str}"
    elif exp == 1:
        return f"{sign}{mant_str} \\cdot 10"
    else:
        return f"{sign}{mant_str} \\cdot 10^{{{exp}}}"


# add "fake legend" to indicate what the colors mean
legend_elements = [
    Line2D([0], [0], color="black", lw=1.5,
           label=f"MCMC, $N_{{samples}}={to_latex_sci_notation(num_MCMC_samples)}, T_{{calc}}={int(slurm_mcmc_run_time)}s$"),
    Line2D([0], [0], color=colors_MC[0], lw=1.5, linestyle=linestyles_MC[0],
           label=f"MC (full), $N_{{samples}}={to_latex_sci_notation(num_MC_samples)}, T_{{calc}}={int(grid_mc_run_time)}s$"),
    Line2D([0], [0], color=colors_MC[1], lw=1.5, linestyle=linestyles_MC[1],
           label=f"MC (coarse), $N_{{samples}}={to_latex_sci_notation(num_MC_samples_coarse)}$"),
]

# Place small horizontal legend centered above the whole figure
fig.legend(
    handles=legend_elements,
    loc="upper center",  # try also: "upper left", "upper right"
    bbox_to_anchor=(0.5, 0.9),  # y = 0.96–1.02 usually works well
    frameon=True,  # or False for cleaner look
    handlelength=1.2,
    handletextpad=0.6,
)

if save_plots:
    plt.savefig('example_mcmc_and_mc_' + str(num_params) + 'd_parameters_distribution')
