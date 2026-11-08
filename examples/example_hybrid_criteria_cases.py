"""
What the two surrogate-accuracy criteria of slurm_mcmc_hybrid see, on four 1D surrogates of a
standard normal target: the weighted log-error epsilon and the weight efficiency eta_w, scored on
N validation points drawn from each surrogate posterior, exactly as a validation round does.
No cluster or surrogate fit is involved; the surrogates are written down analytically.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

plt.close('all')
plt.interactive(True)
plt.rcParams.update({'font.size': 12})

save_plots = False
# save_plots = True

rng = np.random.default_rng(3)

num_validation_points = 200
log_error_threshold = 0.1  # the defaults of slurm_mcmc_hybrid
min_weight_efficiency = 0.9

theta_grid = np.linspace(-4, 8.5, 1000)


def log_prob_expensive(theta):
    return norm.logpdf(theta, 0, 1)


def wiggle(theta, amplitude):
    return amplitude * np.sin(2.3 * theta + 0.4)


def sample_wiggle(amplitude):
    # rejection sampling from N(0,1) * exp(wiggle)
    samples = np.empty(0)
    while len(samples) < num_validation_points:
        theta = rng.normal(0, 1, 4 * num_validation_points)
        accept = rng.random(len(theta)) < np.exp(wiggle(theta, amplitude) - amplitude)
        samples = np.concatenate([samples, theta[accept]])
    return samples[:num_validation_points]


def sample_lobe():
    in_lobe = rng.random(num_validation_points) < 0.2
    return np.where(in_lobe, rng.normal(6, 0.6, num_validation_points), rng.normal(0, 1, num_validation_points))


# each case: title, surrogate log-probability (up to a constant), sampler of its posterior
cases = [
    ('accurate surrogate',
     lambda theta: log_prob_expensive(theta) + wiggle(theta, 0.03),
     lambda: sample_wiggle(0.03)),
    ('noisy surrogate',
     lambda theta: log_prob_expensive(theta) + wiggle(theta, 0.3),
     lambda: sample_wiggle(0.3)),
    ('spurious lobe',
     lambda theta: np.logaddexp(np.log(0.8) + norm.logpdf(theta, 0, 1), np.log(0.2) + norm.logpdf(theta, 6, 0.6)),
     sample_lobe),
    ('misplaced peak',
     lambda theta: norm.logpdf(theta, 3.5, 0.6),
     lambda: rng.normal(3.5, 0.6, num_validation_points)),
]


def criteria(log_ratio):
    """epsilon and eta_w of the validation points, from d_i = log p_exp - log p_sur."""
    weights_normalized = np.exp(log_ratio - logsumexp(log_ratio))
    log_ratio_mean = np.sum(weights_normalized * log_ratio)
    epsilon = np.sqrt(np.sum(weights_normalized * (log_ratio - log_ratio_mean) ** 2))
    weight_efficiency = np.exp(2 * logsumexp(log_ratio) - np.log(len(log_ratio)) - logsumexp(2 * log_ratio))
    return epsilon, weight_efficiency, weights_normalized


fig = plt.figure(figsize=(10, 10))
grid = fig.add_gridspec(2, 2, left=0.09, right=0.98, bottom=0.06, top=0.96, hspace=0.22, wspace=0.16)
for k, (title, log_prob_surrogate, sample_surrogate) in enumerate(cases):
    row, col = k // 2, k % 2
    panels = grid[row, col].subgridspec(2, 1, height_ratios=[1.3, 1], hspace=0.06)
    validation_points = sample_surrogate()
    log_ratio = log_prob_expensive(validation_points) - log_prob_surrogate(validation_points)
    epsilon, weight_efficiency, weights_normalized = criteria(log_ratio)
    print(f'{title:20s} epsilon={epsilon:.3f}, eta_w={weight_efficiency:.3f}, '
          f'N*eta_w={num_validation_points * weight_efficiency:.1f}')

    ax = fig.add_subplot(panels[0])
    ax.plot(theta_grid, np.exp(log_prob_expensive(theta_grid)), color='k', linewidth=2, label='expensive')
    ax.fill_between(theta_grid, np.exp(log_prob_surrogate(theta_grid)), color='tab:blue', alpha=0.35,
                    label='surrogate')
    ax.plot(validation_points, np.full(num_validation_points, -0.03), '|', color='tab:blue', markersize=8,
            alpha=0.6)
    ax.set_xlim(theta_grid[0], theta_grid[-1])
    ax.set_ylim(-0.06, 0.8)
    ax.set_xticklabels([])
    ax.set_title(title)
    verdict = lambda passed: 'PASS' if passed else 'FAIL'
    ax.text(0.97, 0.95,
            f'$\\varepsilon$ = {epsilon:.3f}  ({verdict(epsilon <= log_error_threshold)})\n'
            f'$\\eta_w$ = {weight_efficiency:.3f}  ({verdict(weight_efficiency >= min_weight_efficiency)})',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.7'))
    if k == 0:
        ax.legend(loc='upper left', fontsize=10)
    if col == 0:
        ax.set_ylabel('density')

    ax = fig.add_subplot(panels[1])
    ax.vlines(validation_points, 0, weights_normalized, color='tab:red', linewidth=1)
    ax.axhline(1 / num_validation_points, color='0.4', linestyle='--', linewidth=1)
    ax.set_xlim(theta_grid[0], theta_grid[-1])
    ax.set_yscale('log')
    ax.set_ylim(1e-6, 1.5)
    ax.set_yticks([1e-6, 1e-4, 1e-2, 1])
    ax.set_xlabel(r'$\theta$')
    if col == 0:
        ax.set_ylabel(r'weight $\tilde w_i$')
    if k == 0:
        ax.text(-3.8, 1.6 / num_validation_points, '$1/N$', color='0.4', fontsize=10)
    if title == 'spurious lobe':
        # the lobe's points carry weights far below the plotted range
        lobe_weight_max = np.max(weights_normalized[validation_points > 4.5])
        ax.text(6, 3e-6, f'lobe: $\\tilde w_i < 10^{{{int(np.ceil(np.log10(lobe_weight_max)))}}}$',
                ha='center', fontsize=10, color='tab:blue')

if save_plots:
    plt.savefig('example_hybrid_criteria_cases')
