"""
Minimal template for the hybrid surrogate-MCMC pipeline (slurmcmc.hybrid), on a 3D Rosenbrock.

Every argument of slurm_mcmc_hybrid() is written on its own line, including the ones left at
their default value, so it is obvious what the run is being told and easy to change one thing
at a time. For a real target, replace log_prob_fun with the expensive function and switch
cluster to 'slurm'.

The pipeline takes minutes while the plots take seconds, so everything the figures need is
cached and reloaded on the next run: edit the plotting section and re-run for free.

The figures stay open when the script ends. Set save_plots = True to also write them to
<work_dir>/figs.
"""
import os
import pickle

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import rosen

from slurmcmc.hybrid import slurm_mcmc_hybrid
from slurmcmc.mcmc import slurm_mcmc
from slurmcmc.surrogates import DistributedGPSurrogate

plt.close('all')
plt.rcParams.update({'font.size': 12})

# save_plots = False
save_plots = True

# Reuse the previous run's results instead of running the pipeline again. This is what makes
# iterating on the plots cheap; set False (or delete the cache file) to force a fresh run.
reuse_previous_run = True

# A plain expensive MCMC on the same target, to compare the hybrid posterior against -- affordable
# only because the target is analytic. It converges slowly (tau keeps growing with chain length),
# so check its convergence figure before trusting a comparison. Cached separately from the pipeline.
calc_expensive_reference = True
# calc_expensive_reference = False
num_iters_reference = 100000
num_reference_stops = 5              # points along the reference chain shown in its convergence figure

np.random.seed(0)

work_dir = 'mcmc_hybrid_example'
cache_file = os.path.join(work_dir, 'example_results.pkl')
reference_cache_file = os.path.join(work_dir, 'reference_results.pkl')
num_params = 3
num_walkers = 10 * num_params        # expensive stage: walkers ARE the cluster parallelism
num_surrogate_walkers = 2 * num_params + 2   # cheap stage: emcee's minimum, for a long chain
minima = np.ones(num_params)  # the Rosenbrock maximum of -rosen, for reference in the plots
param_labels = [f'$x_{i}$' for i in range(num_params)]

# named because the plots draw them too, so they cannot drift out of sync with the run
log_error_threshold = 0.1 # default
posterior_shift_tolerance = None # default None = do not test it at all
min_weight_efficiency = 0.9 # default
train_log_prob_trim_range = 100.0
surrogate_burnin_fraction = 0.2  # default
refine_iters_per_tau_floor, verify_iters_per_tau_floor = 10.0, 50.0 # defaults
num_posterior_points_shown = 200     # posterior draws scattered over the log-prob maps


def log_prob_fun(x):
    """The 'expensive' log-probability. Analytic here so the example runs on a laptop."""
    return -rosen(x)


init_points = np.random.uniform(-2, 2, (num_walkers, num_params))


def run_hybrid():
    """Run the pipeline, and cache what the plots need."""
    result = slurm_mcmc_hybrid(
        # --- the problem ---
        log_prob_fun=log_prob_fun,
        init_points=init_points,
        param_bounds=None,                      # None = unbounded, confined by the training data
        constraint_fun=None,
        # --- expensive stage (round 0), where essentially all the cost is ---
        num_expensive_iters=10,
        initial_train_points=None,              # existing evaluations, e.g. from a slurm_mcmc run: skips round 0
        initial_train_values=None,              # their log-probabilities, same length
        num_regularization_points=50,
        num_expensive_iters_per_round=0,        # > 0 re-runs a short expensive chain each round
        # --- surrogate ---
        # 'gp' | 'polynomial' | DistributedGPSurrogate | your own fit/predict object
        surrogate=DistributedGPSurrogate(parallel='joblib', max_points_per_expert=500),
        polynomial_degree=2,
        num_surrogate_iters='auto',             # sized from tau, chain by chain
        num_surrogate_walkers=num_surrogate_walkers,
        surrogate_burnin_fraction=surrogate_burnin_fraction,
        surrogate_uncertainty_penalty=1.0,      # sample mean - kappa*std, to avoid extrapolation
        # --- validation and convergence ---
        num_validation_points='auto',           # max(200, 10 * num_params)
        log_error_threshold=log_error_threshold,
        min_weight_efficiency=min_weight_efficiency,
        posterior_shift_tolerance=posterior_shift_tolerance,
        refine_surrogate_ess='auto',            # refining: ESS floor, 'auto' = 10 * num_params
        refine_iters_per_tau='auto',            # refining: L_c/tau floor, 'auto' = 10
        verify_surrogate_ess='auto',            # verifying: ESS floor, 'auto' = 100 * num_params
        verify_iters_per_tau='auto',            # verifying: L_c/tau floor, 'auto' = 50
        max_verification_attempts=None,   # None = as many as the target needs; num_rounds_max bounds it
        num_rounds_max=30,
        # --- training data handling ---
        train_log_prob_trim_range=train_log_prob_trim_range,
        min_train_points_after_trim='auto',     # 10 * num_params
        max_train_points=None,
        train_log_prob_floor=None,
        # --- saving and restart ---
        save_surrogate='latest',                # None | 'latest' | 'all'
        save_posterior_samples=2000,            # per round, so the run can be replayed after
        save_restart=True,                      # resume a killed run with load_restart=True
        load_restart=os.path.isfile(os.path.join(work_dir, 'hybrid_restart.pkl')),
        restart_file='hybrid_restart.pkl',
        status_restart=None,
        # --- execution ---
        verbosity=1,
        slurm_verbosity=0,
        log_file='hybrid.log',
        extra_arg=None,
        work_dir=work_dir,
        job_name=work_dir,
        cluster='local-map',                    # 'slurm' for the real thing
        submitit_kwargs=None,
        job_fail_value=-1e10,
        expensive_mcmc_kwargs=None,
        keep_run_dirs='all',                    # 'all' | 'failed' | 'none': per-batch directories left in work_dir
        install_signal_handler=True,
        random_seed=0,                          # applied inside the run: remote runs and restarts reproduce too
        remote=False,
        remote_cluster='slurm',
        remote_submitit_kwargs=None,
    )
    if not isinstance(result, dict):
        # remote=True returns a submitit Job at once; .result() waits for the run on the cluster to
        # finish and returns the same dict a local run does
        result = result.result()

    # only what the plots read; the fitted surrogate and the emcee sampler are large, and the
    # sampler holds a reference to the whole runner, so neither is cached
    cached = {'result': {key: result[key] for key in
                         ('converged', 'num_rounds', 'samples', 'num_expensive_evals', 'total_time',
                          'weighted_log_error_per_round', 'weight_efficiency_per_round',
                          'posterior_displacement_per_round', 'num_train_points_per_round',
                          'num_fit_points_per_round', 'round_records', 'surrogate_iters_per_round',
                          'surrogate_ess_per_round', 'surrogate_tau_per_round',
                          'verified_per_round')}}
    os.makedirs(work_dir, exist_ok=True)
    with open(cache_file, 'wb') as handle:
        pickle.dump(cached, handle)
    return cached


def run_reference():
    """A plain expensive MCMC on the same target, no surrogate anywhere, cached on its own."""
    # record_history=False: only the chain is needed, and the history grows with every iteration
    status = slurm_mcmc(log_prob_fun=log_prob_fun, init_points=init_points,
                        num_iters=num_iters_reference, cluster='local-map', verbosity=0,
                        record_history=False)
    burnin = int(surrogate_burnin_fraction * num_iters_reference)  # as the pipeline's own chains
    # the chain keeps its walker axis, which tau needs; float32 halves the file, far below any
    # Monte-Carlo error
    reference = {'iters': num_iters_reference, 'burnin': burnin,
                 'chain': status['sampler'].get_chain(discard=burnin).astype(np.float32)}
    os.makedirs(work_dir, exist_ok=True)
    with open(reference_cache_file, 'wb') as handle:
        pickle.dump(reference, handle)
    return reference


def chain_depth(chain):
    """
    L_c/tau and ESS/tau of an (iterations, walkers, params) chain, measured as the pipeline measures
    its own: tau is the largest over the parameters. tol=0 only silences emcee's too-short
    warning, since L_c/tau is reported here anyway.
    """
    tau = emcee.autocorr.integrated_time(chain, tol=0, quiet=True)
    return tau, chain.shape[0] / np.max(tau), chain.shape[0] * chain.shape[1] / np.max(tau)


if reuse_previous_run and os.path.isfile(cache_file):
    print(f'reusing {cache_file} (delete it, or set reuse_previous_run = False, to run again)')
    with open(cache_file, 'rb') as handle:
        cached = pickle.load(handle)
else:
    print(f'launching the hybrid run in {work_dir} ...')
    cached = run_hybrid()
    print('hybrid run finished')

reference = None
if calc_expensive_reference:
    if os.path.isfile(reference_cache_file):
        with open(reference_cache_file, 'rb') as handle:
            reference = pickle.load(handle)
        if reference.get('iters') != num_iters_reference:   # another length or format: redo it
            reference = None
        else:
            print(f'reusing {reference_cache_file}')
    if reference is None:
        print(f'launching the expensive reference: {num_iters_reference} iterations x {num_walkers} walkers ...')
        reference = run_reference()
        print('expensive reference finished')
reference_samples = None if reference is None else reference['chain'].reshape(-1, num_params)

result = cached['result']
samples = result['samples']
print(f"converged = {result['converged']} after {result['num_rounds']} rounds, "
      f"{result['num_expensive_evals']} expensive evaluations in {result['total_time']:.0f}s")
print('posterior mean:', np.round(samples.mean(axis=0), 3))
print('posterior std: ', np.round(samples.std(axis=0), 3))
if reference is not None:
    # L_c/tau only means something once tau has stopped growing, so it is shown along the chain
    reference_stops = np.linspace(len(reference['chain']) / num_reference_stops,
                                  len(reference['chain']), num_reference_stops).astype(int)
    reference_depth = [chain_depth(reference['chain'][:n]) for n in reference_stops]
    print(f"\nexpensive reference: {reference['iters'] * num_walkers} evaluations, first "
          f"{reference['burnin']} iterations discarded as burn-in")
    print(f"{'iter':>8}{'Lc':>8}  {'tau per param':<22}{'Lc/tau':>7}{'>=50?':>7}{'ESS/tau':>9}")
    for n, (tau, lc_tau, ess) in zip(reference_stops, reference_depth):
        print(f"{reference['burnin'] + n:>8}{n:>8}  {' '.join(f'{t:6.0f}' for t in tau):<22}"
              f"{lc_tau:>7.1f}{'yes' if lc_tau >= 50 else 'NO':>7}{ess:>9.0f}")
    print('reference mean:', np.round(reference_samples.mean(axis=0), 3))
    print('reference std: ', np.round(reference_samples.std(axis=0), 3))

# --- was the cheap chain long enough? (docs/mcmc.md: Lc/tau >= 50, ESS of 10-100 x num_params) ---
# Both are demanded only of verified rounds (marked v); refinement rounds are deliberately short.
print(f'\nchain quality (advised: Lc/tau >= 50, and ESS of 10-100 x num_params = '
      f'{10 * num_params}-{100 * num_params})')
print(f"{'round':>6}{'Lc':>8}{'tau':>7}{'Lc/tau':>9}{'>=50?':>7}{'ESS':>7}{'>=10d?':>8}")
for ind_round, (chain_length, tau, ess, verified) in enumerate(zip(
        result['surrogate_iters_per_round'], result['surrogate_tau_per_round'],
        result['surrogate_ess_per_round'], result['verified_per_round']), 1):
    label = f'{ind_round}v' if verified else str(ind_round)
    print(f'{label:>6}{chain_length:>8}{tau:>7.0f}{chain_length / tau:>9.1f}'
          f'{"yes" if chain_length / tau >= 50 else "NO":>7}{ess:>7.0f}'
          f'{"yes" if ess >= 10 * num_params else "NO":>8}')

figs_dir = os.path.join(work_dir, 'figs')
if save_plots:
    os.makedirs(figs_dir, exist_ok=True)


def save(fig, name):
    if save_plots:
        fig.savefig(os.path.join(figs_dir, 'example_mcmc_hybrid_' + name + '.png'), dpi=150, bbox_inches='tight')


def applied_shift_tolerances(records, requested):
    """
    The tolerance each measurement's posterior shift was held to: recorded since it was kept,
    and otherwise rebuilt as the pipeline computes it -- the requested tolerance raised to the
    noise floor 3 sqrt(1/ESS + 1/ESS_previous), where ESS_previous is the previous round's
    reported chain (its last record).
    """
    applied, previous_ess, last_ess_by_round = [], None, {}
    for rec in records:
        if rec['round'] not in last_ess_by_round and last_ess_by_round:
            previous_ess = list(last_ess_by_round.values())[-1]
        if 'posterior_shift_tolerance' in rec:
            applied.append(rec['posterior_shift_tolerance'])
        elif previous_ess is None:
            applied.append(np.nan)    # nothing to compare the first round with
        else:
            noise = 3.0 * np.sqrt(1.0 / max(rec['chain_ess'], 1.0) + 1.0 / max(previous_ess, 1.0))
            applied.append(max(requested, noise))
        last_ess_by_round[rec['round']] = rec['chain_ess']
    return np.array(applied, dtype=float)


# --- convergence, round by round ---
# Top: the criteria, with the region a converged run must stay out of shaded. Bottom: the depth of
# the surrogate chain against its floors. A verified round has two points, its refinement and its
# verification chain, the latter nudged aside.
records = result['round_records']
rounds = np.array([rec['round'] for rec in records], dtype=float)
verified = np.array([rec['verified'] for rec in records], dtype=bool)
x_round = rounds + np.where(verified, 0.18, 0.0)
round_ticks = np.unique(rounds).astype(int)
round_ticks = round_ticks[::max(1, len(round_ticks) // 10)]   # a tick per round is unreadable by 30
# (values, y label, y scale, [(level, legend label, colour)], shade the failing side of the first
# level: 'above', 'below' or None)
panels = [
    ([rec['weighted_log_error'] for rec in records], '$\\epsilon$ [nats]', 'log',
     [(log_error_threshold, 'threshold', 'red')], 'above'),
    ([rec['weight_efficiency'] for rec in records], 'weight efficiency $\\eta_w$', 'linear',
     [(min_weight_efficiency, 'threshold', 'red')], 'below'),
    # its threshold differs per round (see below), so it is drawn after the loop
    ([rec['posterior_displacement'] for rec in records], 'posterior shift [sd]', 'log', [], None),
    ([rec['surrogate_iters'] / rec['surrogate_tau'] for rec in records], '$L_c/\\tau$', 'linear',
     [(refine_iters_per_tau_floor, f'refinement floor, $L_c/\\tau \\geq {refine_iters_per_tau_floor:.0f}$',
       'orange'),
      (verify_iters_per_tau_floor, f'verification floor, $L_c/\\tau \\geq {verify_iters_per_tau_floor:.0f}$',
       'red')],
     None),
    ([rec['chain_ess'] for rec in records], 'ESS/$\\tau$', 'linear',
     [(10 * num_params, 'refinement floor, ESS/$\\tau \\geq 10\\,d$', 'orange'),
      (100 * num_params, 'verification floor, ESS/$\\tau \\geq 100\\,d$', 'red')], None)]
if posterior_shift_tolerance is None:
    # stability is not a criterion of this run, so its panel would only invite misreading
    panels = [panel for panel in panels if panel[1] != 'posterior shift [sd]']
num_cols = int(np.ceil(len(panels) / 2))
fig, axes = plt.subplots(2, num_cols, figsize=(5.7 * num_cols, 9))
for ax in axes.ravel()[len(panels):]:
    ax.axis('off')
for ax, (values, ylabel, yscale, levels, shade) in zip(axes.ravel(), panels):
    values = np.array(values, dtype=float)
    shown = np.isfinite(values) & ((values > 0) if yscale == 'log' else True)
    ax.plot(x_round[shown], values[shown], '-', color='b', alpha=0.4)
    ax.plot(x_round[shown & ~verified], values[shown & ~verified], 'o', color='b',
            label='refinement chain')
    ax.plot(x_round[shown & verified], values[shown & verified], 's', color='g', markersize=9,
            label='verification chain')
    ax.set_yscale(yscale)
    for level, label, color in levels:  # a criterion can be switched off, and then has no line
        ax.axhline(level, color=color, ls='--', label=label)
    if shade and levels:
        low, high = ax.get_ylim()  # captured before the span, which would otherwise widen them
        if shade == 'above':
            ax.axhspan(levels[0][0], high, color='r', alpha=0.08)
        else:
            low = min(low, levels[0][0] - 0.02)
            ax.axhspan(low, levels[0][0], color='r', alpha=0.08)
        ax.set_ylim(low, high)
    if ylabel == 'posterior shift [sd]' and posterior_shift_tolerance is not None:
        # the tolerance applied is posterior_shift_tolerance raised to the noise floor of the two
        # chains compared, so it is a threshold per round, not one line
        applied = applied_shift_tolerances(records, posterior_shift_tolerance)
        low, high = ax.get_ylim()
        high = max(high, 1.5 * np.nanmax(applied))
        ax.plot(x_round, applied, '--', color='red', marker='_', markersize=12,
                label='tolerance applied: requested, raised to the noise floor')
        ax.fill_between(x_round, applied, high, color='r', alpha=0.08)
        ax.axhline(posterior_shift_tolerance, color='0.4', ls=':', label='tolerance requested')
        ax.set_ylim(min(low, 0.8 * np.nanmin(applied), 0.8 * posterior_shift_tolerance), high)
    ax.legend(fontsize=8, framealpha=0.85)
    ax.set_xticks(round_ticks)
    ax.set_xlabel('round')
    ax.set_ylabel(ylabel)
    ax.grid(True)
fig.suptitle('hybrid convergence')
fig.tight_layout()
save(fig, 'convergence')

# --- the expensive target, as 2d slices through the known optimum, with posterior draws ---
# The background is a slice (other parameters at the optimum) while the dots are a projection of
# the full posterior, so a bright patch without dots is not by itself a missed mode.
# The window spans both chains, by percentiles so a few far-tail samples cannot squash the valley.
both = samples if reference_samples is None else np.vstack([samples, reference_samples])
plot_ranges = []
for ind_param in range(num_params):
    low, high = np.percentile(both[:, ind_param], [0.1, 99.9])
    pad = 0.15 * (high - low)
    plot_ranges.append((low - pad, high + pad))
rng = np.random.default_rng(0)
shown = samples[rng.choice(len(samples), num_posterior_points_shown, replace=False)]
shown_reference = (None if reference_samples is None else
                   reference_samples[rng.choice(len(reference_samples), num_posterior_points_shown,
                                                replace=False)])

fig, axes = plt.subplots(num_params - 1, num_params - 1,
                         figsize=(4.2 * (num_params - 1), 3.8 * (num_params - 1)))
axes = np.atleast_2d(axes)
for row in range(num_params - 1):
    for col in range(num_params - 1):
        ax = axes[row, col]
        ind_x, ind_y = col, row + 1
        if ind_x >= ind_y:
            ax.axis('off')
            continue
        grid_x = np.linspace(*plot_ranges[ind_x], 120)
        grid_y = np.linspace(*plot_ranges[ind_y], 120)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        points = np.tile(minima, (mesh_x.size, 1))
        points[:, ind_x] = mesh_x.ravel()
        points[:, ind_y] = mesh_y.ravel()
        values = np.array([log_prob_fun(p) for p in points]).reshape(mesh_x.shape)
        # e^-200 is negligible; clipping keeps the colour scale on the region that matters
        high = float(values.max())
        mesh = ax.pcolormesh(mesh_x, mesh_y, np.clip(values, high - 200.0, high), cmap='viridis')
        if shown_reference is not None:  # drawn first, so the hybrid's draws sit on top
            ax.scatter(shown_reference[:, ind_x], shown_reference[:, ind_y], s=8,
                       c='deepskyblue', alpha=0.7)
        ax.scatter(shown[:, ind_x], shown[:, ind_y], s=8, c='r', alpha=0.6)
        ax.plot(minima[ind_x], minima[ind_y], marker='*', markersize=14, markerfacecolor='none',
                markeredgecolor='w', markeredgewidth=1.5)
        ax.set_xlabel(param_labels[ind_x])
        ax.set_ylabel(param_labels[ind_y])
# every panel shares the same 200-nat scale, so one colorbar serves them all; it and the
# legend go in the empty upper-right triangle of the corner layout
handles = [plt.Line2D([], [], color='r', marker='o', ls='', label='hybrid posterior')]
if shown_reference is not None:
    handles.append(plt.Line2D([], [], color='deepskyblue', marker='o', ls='',
                              label='expensive reference'))
handles.append(plt.Line2D([], [], color='k', marker='*', markerfacecolor='none', ls='',
                          markersize=12, label='true optimum'))
fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.78, 0.90), fontsize=11)
fig.suptitle('expensive log-probability, sliced at the optimum, with posterior draws')
fig.tight_layout()
fig.colorbar(mesh, ax=axes, label='log-probability [nats]', fraction=0.06, shrink=0.45,
             anchor=(0.0, 0.85))
save(fig, 'log_prob_maps')

# --- posterior, against the expensive reference when it was computed ---
# density=True on the histograms: corner plots raw counts, so two chains of different length
# would give marginals of different height even where they agree exactly.
corner_range = plot_ranges  # already spans both chains, so the two corner plots align
fig = plt.figure(figsize=(7, 7))
if reference_samples is not None:
    corner.corner(reference_samples, color='b', fig=fig, bins=50, labels=param_labels,
                  range=corner_range, plot_datapoints=False, plot_density=False,
                  hist_kwargs=dict(density=True))
corner.corner(samples, color='k', truths=minima, truth_color='r', fig=fig, bins=50,
              labels=param_labels, range=corner_range, plot_datapoints=False,
              plot_density=False, hist_kwargs=dict(density=True))
handles = [plt.Line2D([], [], color='k', label='hybrid surrogate posterior')]
if reference_samples is not None:
    handles.append(plt.Line2D([], [], color='b', label='expensive reference'))
handles.append(plt.Line2D([], [], color='r', label='true optimum'))
fig.legend(handles=handles, loc='upper right', fontsize=10)
fig.suptitle('posterior')
fig.tight_layout()
save(fig, 'posterior')

# --- the hybrid posterior, round by round (for a verified round, its verification chain's) ---
# Each round saved the same number of draws, far fewer than the reference, hence coarser bins.
posterior_dir = os.path.join(work_dir, 'posteriors')
saved_rounds = (sorted(int(f[len('posterior_round'):-4]) for f in os.listdir(posterior_dir)
                       if f.startswith('posterior_round'))
                if os.path.isdir(posterior_dir) else [])
if saved_rounds:
    per_round = [np.load(os.path.join(posterior_dir, f'posterior_round{r}.npy')) for r in saved_rounds]
    fig = plt.figure(figsize=(7, 7))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(saved_rounds)))
    for color, round_samples in zip(colors, per_round):
        corner.corner(round_samples, color=color, fig=fig, bins=30, labels=param_labels,
                      range=corner_range, plot_datapoints=False, plot_density=False,
                      hist_kwargs=dict(density=True))
    drawn = list(per_round)
    if reference_samples is not None:  # on top and dashed, as the thing to converge to
        corner.corner(reference_samples, color='k', fig=fig, bins=30, labels=param_labels,
                      range=corner_range, plot_datapoints=False, plot_density=False,
                      hist_kwargs=dict(density=True, ls='--', lw=1.5),
                      contour_kwargs=dict(linestyles='--'))
        drawn.append(reference_samples)
    # corner sizes each marginal to whichever histogram it drew last, which would clip the peaks
    # of the noisier rounds; size it to the tallest of them instead
    diagonal = np.array(fig.axes).reshape(num_params, num_params).diagonal()
    for ind_param, ax in enumerate(diagonal):
        peak = max(np.histogram(d[:, ind_param], bins=30, range=corner_range[ind_param],
                                density=True)[0].max() for d in drawn)
        ax.set_ylim(0, 1.05 * peak)
    handles = [plt.Line2D([], [], color=color,
                          label=f"round {r}{'v' if result['verified_per_round'][r - 1] else ''}: "
                                f"$\\epsilon$ = {result['weighted_log_error_per_round'][r - 1]:.3f}")
               for color, r in zip(colors, saved_rounds)]
    if reference_samples is not None:
        handles.append(plt.Line2D([], [], color='k', ls='--', label='expensive reference'))
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.98, 0.88), fontsize=9)
    fig.suptitle('hybrid posterior in different rounds')
    fig.tight_layout()
    save(fig, 'hybrid_posterior_per_round')

# --- is the reference itself converged? The posterior at several stops along its chain ---
# Agreement between stops is necessary, not sufficient: read it with L_c/tau in the legend.
if reference is not None:
    fig = plt.figure(figsize=(7, 7))
    colors = plt.cm.rainbow(np.linspace(0, 1, num_reference_stops))
    num_per_stop = reference_stops[0] * reference['chain'].shape[1]
    rng = np.random.default_rng(0)
    for color, n in zip(colors, reference_stops):
        stop_samples = reference['chain'][:n].reshape(-1, num_params)
        stop_samples = stop_samples[rng.permutation(len(stop_samples))[:num_per_stop]]
        corner.corner(stop_samples, color=color, fig=fig, bins=50, labels=param_labels,
                      range=corner_range, plot_datapoints=False, plot_density=False,
                      hist_kwargs=dict(density=True))
    handles = [plt.Line2D([], [], color=color,
                          label=f"iter {reference['burnin'] + n}: $L_c/\\tau$ = {lc_tau:.0f}, "
                                f"ESS/$\\tau$ = {ess:.0f}")
               for color, n, (_, lc_tau, ess) in zip(colors, reference_stops, reference_depth)]
    # in the empty upper-right triangle of the corner layout, clear of the title
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.98, 0.88), fontsize=9)
    fig.suptitle('expensive reference at different iteration stops')
    fig.tight_layout()
    save(fig, 'reference_convergence')

plt.show()  # blocking, so the figures stay open when the script ends
