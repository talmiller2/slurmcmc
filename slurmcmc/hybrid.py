"""
Hybrid surrogate-MCMC: sample the posterior of an expensive log-probability with far fewer
expensive evaluations than a plain slurm_mcmc, by iterating a fast surrogate until measured
criteria say its posterior can be trusted.

1. Round 0: a short expensive MCMC on the cluster, plus optional regularization points, builds
   the training set -- or existing evaluations are passed as initial_train_points/values.
2. Fit a surrogate (a Gaussian process by default) to the training set.
3. Sample the surrogate with a cheap, vectorized MCMC.
4. Validate: evaluate the expensive function on points drawn from the surrogate posterior.
5. If the criteria fail, add those evaluations to the training set and repeat from 2.
6. A pass on the cheap chain is verified before it is accepted: the same surrogate is re-sampled
   to reporting depth and validated again.

Convergence metrics
-------------------
With d_i = log p_expensive(x_i) - log p_surrogate(x_i) on the validation points and importance
weights w_i = exp(d_i), weighted_log_error is the w-weighted RMS of d about its weighted mean, in
nats (0.02 ~ probability ratios accurate to 2%). The weights make it an average under the
expensive posterior, and the mean is removed because a log-probability is only defined up to a
constant. A run converges when

    weighted_log_error <= log_error_threshold,
    ess_weights = (sum w)^2 / sum w^2 >= min_ess_weights,  (a few dominant weights can make the
                                                             error spuriously small)
    posterior shift <= max(posterior_shift_tolerance, noise floor)   (when enabled).

ess_weights counts validation points; it is unrelated to the ESS of a chain, N / tau.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import os
import pickle
import shutil
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import emcee
import numpy as np
import submitit

# Re-exported so `from slurmcmc.hybrid import GaussianProcessSurrogate` keeps working, and so
# that artifacts pickled before the surrogates moved out of this module still unpickle:
# a saved surrogate references its optimizer by module path, and restart files, cached results
# and saved fits from earlier runs all name slurmcmc.hybrid.
from slurmcmc.surrogates import (DistributedGPSurrogate, GaussianProcessSurrogate,
                                 PolynomialSurrogate, _DetrendedSurrogateBase, _fit_expert,
                                 _lbfgs_with_more_iterations)
from slurmcmc.general_utils import (set_logging, combine_args, point_to_tuple, signal_handler,
                                    save_restart_file, load_restart_file, seed_random_generators,
                                    get_random_state, set_random_state)
from slurmcmc.import_utils import deferred_import_function_wrapper
from slurmcmc.mcmc import slurm_mcmc
from slurmcmc.slurm_utils import Cluster, KeepRunDirs, SlurmPool, submit_remote_run


# ---------------------------------------------------------------------------
# Helpers and constants
# ---------------------------------------------------------------------------

def stats_norm_ppf(q: float) -> float:
    """Standard-normal quantile (kept local so the hot path never imports scipy.stats)."""
    from scipy import stats
    return float(stats.norm.ppf(q))


# Confinement of an unbounded run (param_bounds=None). Away from its training data a surrogate
# tends to its trend -- a quadratic that can run off to +inf -- or to a constant, which is
# improper, and either way the cheap chain walks away. With R the largest Mahalanobis radius of
# the training points, a quartic penalty acts beyond INFLATION*R and the log-prob is -inf beyond
# HARD_RADIUS*R; both grow as the training set spreads. Constants rather than arguments: with a
# GP the uncertainty penalty turns the walkers back first, so the envelope only matters for
# surrogates without predict_std.
_CONFINEMENT_INFLATION = 2.0
_CONFINEMENT_STRENGTH = 10.0      # penalty scale, in units of the training log-prob range
_CONFINEMENT_HARD_RADIUS = 10.0

# The posterior-shift tolerance never falls below this many standard errors of the difference
# between two finite chains; below that the criterion would be unsatisfiable, not strict.
_POSTERIOR_NOISE_SIGMAS = 3.0

# ---------------------------------------------------------------------------
# Config + runner
# ---------------------------------------------------------------------------

@dataclass
class HybridMCMCConfig:
    """
    Complete, picklable description of a slurm_mcmc_hybrid run.
    """
    log_prob_fun: Union[Callable, Dict]
    # Round-0 walkers, shape (num_walkers, num_params). Optional with initial_train_points, except
    # that num_expensive_iters_per_round > 0 takes its walker count from it.
    init_points: Optional[np.ndarray] = None
    # [[lo, hi], ...]: a hard box that constrains the posterior. None runs unbounded, confined by
    # an envelope grown from the training data (see _CONFINEMENT_*).
    param_bounds: Optional[List] = None
    constraint_fun: Optional[Union[Callable, Dict]] = None
    num_expensive_iters: int = 20
    # Existing evaluations, shapes (n, num_params) and (n,), e.g. a previous slurm_mcmc's
    # points_history and values_history[:, 0]. Round 0 is skipped; they are not counted in
    # num_expensive_evals.
    initial_train_points: Optional[np.ndarray] = None
    initial_train_values: Optional[np.ndarray] = None
    num_regularization_points: int = 0
    num_expensive_iters_per_round: int = 0  # optional expensive MCMC each round
    surrogate: Union[str, Any] = 'gp'  # 'gp' | 'polynomial' | object with fit(X, y) / predict(X)
    polynomial_degree: int = 3
    num_surrogate_iters: int = 2000
    # Walkers of the cheap chain, independent of the expensive stage. At fixed work W = N_c L_c,
    # ESS = W/tau does not depend on the walker count but L_c/tau falls with it, so None uses
    # emcee's minimum, 2*num_params + 2.
    num_surrogate_walkers: Optional[int] = None
    surrogate_burnin_fraction: float = 0.2  # discarded before tau and the samples are taken
    num_validation_points: int = 100
    log_error_threshold: float = 0.1  # posterior-weighted RMS log-prob error, in nats
    # below this many effective validation points the weighted error is not trusted
    min_ess_weights: float = 10.0
    # Largest change of any parameter's posterior mean or std between rounds, in posterior
    # standard deviations; never tested below the chains' noise floor. None disables it.
    posterior_shift_tolerance: Optional[float] = 0.1
    # ESS floors of the cheap chain: refining (posterior discarded at the next fit; None -> 10 d)
    # and verifying (posterior reported; None -> 100 d). The chain is extended, up to
    # max_surrogate_iters_multiplier x num_surrogate_iters, until they hold.
    refine_surrogate_ess: Optional[float] = None
    final_surrogate_ess: Optional[float] = None
    # Rejected verifications allowed before the run gives up and ends; None = unlimited.
    max_verification_attempts: Optional[int] = None
    # L_c/tau floors of the two depths; None while refining = not demanded, since that posterior
    # is discarded and tau is unreliable on a short chain anyway.
    refine_iters_per_tau: Optional[float] = None
    final_iters_per_tau: float = 50.0
    max_surrogate_iters_multiplier: int = 8
    num_rounds_max: int = 5
    # Stall warning: judged from the trend of log(error) over all rounds, after stall_window of
    # them, and only when the improvement rate is significantly below stall_relative_improvement
    # -- a single round's error scatters too much to compare with the previous one.
    stall_window: int = 12
    stall_significance: float = 0.05
    stall_relative_improvement: float = 0.05
    # warn when this fraction of the posterior lies within bounds_warning_tolerance of a bound
    bounds_warning_fraction: float = 0.10
    bounds_warning_tolerance: float = 0.01
    # The surrogate MCMC samples mean - kappa*std, so walkers avoid regions where the surrogate
    # extrapolates. Inactive for surrogates without predict_std (e.g. the polynomial).
    surrogate_uncertainty_penalty: float = 1.0
    # Drop training points this many nats below the best: the MCMC never visits them, and they
    # only slow the fit. Trimming much harder leaves the surrogate without evidence that the
    # probability is low away from the peak. None disables it.
    train_log_prob_trim_range: Optional[float] = 100.0
    min_train_points_after_trim: int = 50
    # Cap on points entering a fit, whose cost grows roughly as n^2.4. A uniform random subsample,
    # which keeps the far-field points that stop the surrogate extrapolating. None = no cap.
    max_train_points: Optional[int] = None
    train_log_prob_floor: Optional[float] = None  # drop training points with log-prob <= floor (penalty values)
    # Fitted surrogates in <work_dir>/surrogates/round{k}.pkl: 'latest' keeps one, enough for a
    # resumed round to skip its refit; 'all' keeps every round (~8 n^2 bytes each). None = off.
    save_surrogate: Optional[Literal['latest', 'all']] = None
    # Draws kept per round in <work_dir>/posteriors/round{k}.npy. None = off.
    save_posterior_samples: Optional[int] = None
    # The restart file holds the training set and diagnostics, not the surrogate (refitted) or
    # the cheap chain's sampler (only its walker positions); written after round 0 and every round.
    save_restart: bool = False
    load_restart: bool = False
    restart_file: str = 'hybrid_restart.pkl'
    status_restart: Optional[Dict] = None  # restart from a status dict instead of a file
    verbosity: int = 1
    slurm_verbosity: int = 0
    log_file: Optional[str] = None
    extra_arg: Any = None
    work_dir: str = 'mcmc_hybrid'
    job_name: str = 'mcmc_hybrid'
    cluster: Cluster = 'slurm'
    submitit_kwargs: Optional[Dict] = None
    job_fail_value: float = -1e10
    expensive_mcmc_kwargs: Optional[Dict] = None  # extra kwargs forwarded to the inner slurm_mcmc calls
    keep_run_dirs: KeepRunDirs = 'all'  # per-batch directories of the expensive evaluations, see SlurmPool
    install_signal_handler: bool = True
    # seeds every random choice, inside the process that runs the loop (so remote runs too)
    random_seed: Optional[int] = None
    # remote run params:
    remote: bool = False
    remote_cluster: Literal['slurm', 'local'] = 'slurm'
    remote_submitit_kwargs: Optional[Dict] = None


class HybridMCMCRunner:
    """
    Runner for the automated hybrid surrogate-MCMC pipeline (see module docstring).

    Usage:
        result = HybridMCMCRunner(HybridMCMCConfig(log_prob_fun=..., init_points=...,
                                                   ...)).run()

    Most callers use the `slurm_mcmc_hybrid(...)` convenience wrapper instead.
    """

    def __init__(self, config: HybridMCMCConfig) -> None:
        self.cfg = config
        self.constraint_fun: Optional[Callable] = None
        self.surrogate: Any = None
        self.train_X: List = []
        self.train_y: List[float] = []
        self._train_points_set: set = set()
        self.num_expensive_evals: int = 0
        self._unbounded = config.param_bounds is None
        if self._unbounded:
            self._lower = None
            self._upper = None
        else:
            self._lower = np.array([b[0] for b in config.param_bounds], dtype=float)
            self._upper = np.array([b[1] for b in config.param_bounds], dtype=float)
        if config.init_points is None and config.initial_train_points is None:
            err_msg = ('init_points is required unless initial_train_points is given: without '
                       'either there is nothing to start the run from.')
            logging.error(err_msg)
            raise ValueError(err_msg)
        if config.init_points is None and config.num_expensive_iters_per_round > 0:
            err_msg = ('num_expensive_iters_per_round > 0 needs init_points: its length sets the '
                       'number of walkers of the expensive refresh chains.')
            logging.error(err_msg)
            raise ValueError(err_msg)
        dims = {name: int(np.atleast_2d(np.asarray(points)).shape[1])
                for name, points in (('init_points', config.init_points),
                                     ('initial_train_points', config.initial_train_points))
                if points is not None}
        if len(set(dims.values())) > 1:
            err_msg = (f'init_points and initial_train_points disagree on the number of '
                       f'parameters: {dims}.')
            logging.error(err_msg)
            raise ValueError(err_msg)
        self._num_params = next(iter(dims.values()))
        # confinement geometry, refreshed from the training set at each fit (unbounded mode only)
        self._confinement: Optional[Dict[str, Any]] = None
        self._filter_stats: Optional[Dict[str, int]] = None
        if (config.initial_train_points is None) != (config.initial_train_values is None):
            err_msg = 'initial_train_points and initial_train_values must be given together.'
            logging.error(err_msg)
            raise ValueError(err_msg)
        if config.initial_train_points is not None and \
                len(config.initial_train_points) != len(config.initial_train_values):
            err_msg = (f'initial_train_points has {len(config.initial_train_points)} rows but '
                       f'initial_train_values has {len(config.initial_train_values)}.')
            logging.error(err_msg)
            raise ValueError(err_msg)
        self.timings: Dict[str, float] = {}
        self.timings_per_round: List[Dict[str, float]] = []
        self._last_surrogate_iters: int = 0
        self._last_surrogate_tau: float = float('nan')
        self._last_surrogate_chain_ess: float = float('nan')
        self.num_fit_points_per_round: List[int] = []
        self._resumed: bool = False
        self._surrogate_switched_at: Optional[int] = None
        self._config_changes_on_restart: Dict[str, Any] = {}
        self._previous_surrogate_class: Optional[str] = None

    # ------------------------------------------------------------------
    # Directory layout
    # ------------------------------------------------------------------

    def _stage_dir(self, stage_name: str) -> str:
        """Per-stage sub-directory of work_dir, e.g. round0_expensive_mcmc, round1_validation."""
        return os.path.join(self.cfg.work_dir, stage_name)

    def _fresh_stage_dir(self, stage_name: str) -> str:
        """
        The stage directory, cleared of an earlier attempt at the same stage: a round interrupted
        before its restart was written, or an earlier run in the same work_dir. Its evaluations
        never reached the training set, and SlurmPool refuses a directory holding old output. It
        is moved aside to <stage>_interrupted<k> for inspection, or removed when keep_run_dirs='none'.
        """
        path = self._stage_dir(stage_name)
        if os.path.isdir(path) and os.listdir(path):
            if self.cfg.keep_run_dirs == 'none':
                shutil.rmtree(path, ignore_errors=True)
            else:
                k = 1
                while os.path.exists(f'{path}_interrupted{k}'):
                    k += 1
                os.rename(path, f'{path}_interrupted{k}')
                if self.cfg.verbosity >= 1:
                    logging.info(f'    moved the output of an earlier attempt at {stage_name} '
                                 f'to {stage_name}_interrupted{k}.')
        return path

    # The per-round table: name, column width, and the description printed above the table.
    _DIAGNOSTICS_COLUMNS = [
        ('round', 5, 'refinement round'),
        ('n_expensive', 12, 'expensive training points accumulated before this round'),
        ('n_train', 8, 'points actually fitted, after trimming and max_train_points'),
        ('surrogate', 22, 'surrogate class fitted this round'),
        ('w_log_err', 10, 'posterior-weighted RMS log-prob error [nats]'),
        ('post_shift', 11, 'largest parameter mean/std change vs the previous round, in '
                           'posterior standard deviations'),
        ('surr_iters', 11, 'cheap-chain length L_c actually run; grows until both chain '
                           'criteria below are met'),
        ('Lc/tau', 8, 'chain length in autocorrelation times; advised >= 50'),
        ('ess/tau', 9, 'effective samples in the surrogate-based chain, N_total/tau; '
                       'advised 10-100 x num_params'),
        ('t_fit_s', 9, 'time: surrogate fit [s]'),
        ('t_mcmc_s', 9, 'time: cheap surrogate MCMC [s]'),
        ('t_valid_s', 10, 'time: expensive validation batch [s]'),
        ('t_refresh_s', 12, 'time: expensive MCMC refresh [s], 0 unless '
                            'num_expensive_iters_per_round > 0'),
        ('t_round_s', 10, 'time: round total [s]'),
    ]

    def _diagnostics_header(self) -> str:
        """
        The run's settings and its expensive-evaluation budget, written above the per-round
        table so everything needed to read a run sits in one file.
        """
        cfg = self.cfg
        seeded = cfg.initial_train_points is not None
        num_walkers = (len(np.atleast_2d(np.asarray(cfg.init_points)))
                       if cfg.init_points is not None else None)

        def row(name: str, value: Any) -> str:
            return f'  {name:<38} {value}\n'

        def fmt(value: Any) -> str:
            return 'none' if value is None else str(value)

        surrogate_name = (type(self.surrogate).__name__.replace('Surrogate', '')
                          if self.surrogate is not None else str(cfg.surrogate))
        text = 'hybrid surrogate-MCMC diagnostics\n\nrun parameters\n'
        text += row('num_params', self._num_params)
        text += row('num_walkers', fmt(num_walkers))
        text += row('param_bounds', 'none (unbounded, confinement envelope)'
                    if self._unbounded else f'{len(cfg.param_bounds)} x [lo, hi]')
        text += row('surrogate', surrogate_name)
        for name in ('num_expensive_iters', 'num_expensive_iters_per_round',
                     'num_regularization_points', 'num_surrogate_iters',
                     'num_validation_points', 'num_rounds_max', 'log_error_threshold',
                     'posterior_shift_tolerance', 'min_ess_weights', 'refine_surrogate_ess',
                     'max_surrogate_iters_multiplier', 'surrogate_uncertainty_penalty',
                     'train_log_prob_trim_range', 'min_train_points_after_trim',
                     'max_train_points', 'train_log_prob_floor'):
            text += row(name, fmt(getattr(cfg, name)))

        # num_walkers is always known when a refresh is requested: __init__ refuses otherwise
        num_refresh = ((cfg.num_expensive_iters_per_round + 1) * num_walkers
                       if cfg.num_expensive_iters_per_round > 0 else 0)
        if seeded:
            num_seeded = len(cfg.initial_train_points)
            text += ('\nexpensive evaluations\n'
                     f'  round 0 skipped: seeded with {num_seeded} existing evaluations '
                     f'(initial_train_points),\n'
                     f'  not counted in num_expensive_evals; num_expensive_iters and '
                     f'num_regularization_points are unused\n'
                     f'  => raw training points at the first surrogate fit = {num_seeded}\n'
                     f'  each refinement round then adds num_validation_points = '
                     f'{cfg.num_validation_points}')
        else:
            num_round0 = (cfg.num_expensive_iters + 1) * num_walkers
            num_first_fit = num_round0 + cfg.num_regularization_points
            text += ('\nexpensive evaluations\n'
                     f'  round 0 MCMC: (num_expensive_iters + 1) * num_walkers = '
                     f'({cfg.num_expensive_iters} + 1) * {num_walkers} = {num_round0}\n'
                     f'  regularization points: {cfg.num_regularization_points}\n'
                     f'  => raw training points at the first surrogate fit = {num_round0} + '
                     f'{cfg.num_regularization_points} = {num_first_fit}\n'
                     f'  each refinement round then adds num_validation_points = '
                     f'{cfg.num_validation_points}')
        if num_refresh:
            text += (f', plus (num_expensive_iters_per_round + 1) * num_walkers = '
                     f'{num_refresh} for the expensive refresh')
        text += ('\n  (exact unless a proposed point repeats one already evaluated, or an '
                 'evaluation\n   fails, in which case these are upper bounds)\n')

        text += '\ncolumns\n'
        for name, _, description in self._DIAGNOSTICS_COLUMNS:
            text += f'  {name:<13}{description}\n'
        text += ('  a verified round prints two rows: n is its refinement chain, whose metrics\n'
                 '  passed and so triggered the verification, and nv is the verification chain\n'
                 '  that then judged the round. Both share one fit and training set, so n_expensive,\n'
                 '  n_train, surrogate and t_fit_s appear on the first row only.\n')
        text += '\n'
        return text

    def _write_diagnostics(self, rows: List[Dict[str, Any]]) -> None:
        """
        Write the per-round metrics as a human-readable text table, refreshed each round so a
        killed run still leaves the full history on disk.
        """
        path = os.path.join(self.cfg.work_dir, 'hybrid_diagnostics.txt')
        columns = [(name, width) for name, width, _ in self._DIAGNOSTICS_COLUMNS]
        try:
            os.makedirs(self.cfg.work_dir, exist_ok=True)
            with open(path, 'w') as f:
                f.write(self._diagnostics_header())
                f.write(''.join(f'{name:>{width}}' for name, width in columns) + '\n')
                f.write('-' * sum(width for _, width in columns) + '\n')
                for row in rows:
                    f.write(''.join(f'{row.get(name, ""):>{width}}' for name, width in columns) + '\n')
        except OSError as e:
            logging.warning(f'could not write diagnostics file {path}: {e}')

    # ------------------------------------------------------------------
    # Training-data management
    # ------------------------------------------------------------------

    def _point_is_trainable(self, point, value) -> bool:
        if point_to_tuple(point) in self._train_points_set:
            return False
        value = float(np.ravel(value)[0])
        if not np.isfinite(value):
            return False
        if value == self.cfg.job_fail_value:
            return False
        if self.cfg.train_log_prob_floor is not None and value <= self.cfg.train_log_prob_floor:
            return False
        if self.constraint_fun is not None:
            if self.constraint_fun(*combine_args(point, self.cfg.extra_arg)) > 0:
                return False
        return True

    def _add_training_points(self, points, values) -> int:
        num_added = 0
        for point, value in zip(points, values):
            if self._point_is_trainable(point, value):
                self._train_points_set.add(point_to_tuple(point))
                self.train_X.append(np.array(point, dtype=float))
                self.train_y.append(float(np.ravel(value)[0]))
                num_added += 1
        return num_added

    def _training_arrays(self):
        """The points entering the fit: trimmed by train_log_prob_trim_range, capped by max_train_points."""
        X = np.array(self.train_X)
        y = np.array(self.train_y)
        num_raw = len(y)
        num_trimmed = 0
        num_subsampled = 0
        trim_range = self.cfg.train_log_prob_trim_range
        if trim_range is not None:
            keep = y > np.max(y) - trim_range
            if keep.sum() >= self.cfg.min_train_points_after_trim:
                if self.cfg.verbosity >= 2 and keep.sum() < len(y):
                    logging.info(f'    trimmed {len(y) - keep.sum()} training points more than '
                                 f'{trim_range} nats below the peak ({keep.sum()} remain).')
                num_trimmed = int(len(y) - keep.sum())
                X, y = X[keep], y[keep]
            elif self.cfg.verbosity >= 1:
                logging.warning(f'    train_log_prob_trim_range={trim_range} would leave only '
                                f'{keep.sum()} points; skipping the trim.')

        cap = self.cfg.max_train_points
        if cap is not None and len(y) > cap:
            # uniform subsample: keeps the spatial spread, unlike keeping the top-N by log-prob
            rng = np.random.default_rng(0)
            sel = rng.choice(len(y), size=cap, replace=False)
            if self.cfg.verbosity >= 2:
                logging.info(f'    subsampled the training set from {len(y)} to {cap} points '
                             f'(max_train_points) to bound the O(n^3) fit')
            num_subsampled = int(len(y) - cap)
            X, y = X[sel], y[sel]
        self._filter_stats = {'raw': num_raw, 'trimmed': num_trimmed,
                              'subsampled': num_subsampled, 'kept': len(y)}
        return X, y

    # ------------------------------------------------------------------
    # Surrogate handling
    # ------------------------------------------------------------------

    def _create_surrogate(self):
        cfg = self.cfg
        surrogate = self._build_surrogate()
        # a surrogate that distributes its fits needs scratch space the compute nodes can read;
        # work_dir already is, so hand it down, with the run's job name, unless the caller chose them
        if getattr(surrogate, 'work_dir', 'unset') is None:
            surrogate.work_dir = cfg.work_dir
        if getattr(surrogate, 'job_name', 'unset') is None:
            surrogate.job_name = cfg.job_name
        return surrogate

    def _build_surrogate(self):
        cfg = self.cfg
        if isinstance(cfg.surrogate, str):
            if cfg.surrogate == 'gp':
                return GaussianProcessSurrogate()
            elif cfg.surrogate == 'polynomial':
                return PolynomialSurrogate(degree=cfg.polynomial_degree)
            else:
                err_msg = f"invalid surrogate: {cfg.surrogate}. use 'gp', 'polynomial' or a fit/predict object."
                logging.error(err_msg)
                raise ValueError(err_msg)
        return cfg.surrogate  # user-supplied object with fit/predict

    def _surrogate_path(self, ind_round: int) -> str:
        """Where this round's fit is saved; named after its round in both save modes."""
        return os.path.join(self.cfg.work_dir, 'surrogates', f'round{ind_round}.pkl')

    #: the surrogate log's fixed columns; the free-text details column closes each line
    _SURROGATE_LOG_COLUMNS = [('round', 6), ('surrogate', 17), ('n_raw', 7), ('n_trim', 7),
                              ('n_sub', 6), ('n_fit', 7), ('t_fit_s', 9), ('file', 12)]

    @staticmethod
    def _surrogate_details(surrogate: Any) -> str:
        """
        The settings this surrogate was built with, and what its fit came out as: the constructor
        arguments that differ from their defaults, then the fitted kernel (one expert's, plus the
        expert sizes, for an ensemble). Best effort -- an unknown object contributes nothing.
        """
        parts = []
        try:
            for name, parameter in inspect.signature(type(surrogate).__init__).parameters.items():
                if name in ('self', 'work_dir', 'job_name', 'submitit_kwargs', 'kwargs'):
                    continue
                value = getattr(surrogate, name, parameter.default)
                if value is not parameter.default and value != parameter.default:
                    parts.append(f'{name}={value}')
        except (TypeError, ValueError):
            pass
        settings = ' '.join(parts) if parts else 'default settings'

        experts = getattr(surrogate, '_experts', None)
        fitted = surrogate if experts is None else (experts[0] if experts else None)
        if experts:
            sizes = [len(getattr(e, 'X_train_', [])) for e in experts]
            settings += f' | {len(experts)} experts on {sizes} points'
        kernel = getattr(getattr(fitted, '_gpr', fitted), 'kernel_', None)
        if kernel is not None:
            kernel = str(kernel).replace('\n', ' ')
            settings += ' | kernel' + ('[expert 0]' if experts else '') + '=' + kernel[:150]
        return settings

    def _append_surrogate_log(self, ind_round: int, num_fit: int, elapsed: Optional[float]) -> None:
        """
        One line per fit in <work_dir>/surrogates/surrogate_log.txt: which surrogate was fitted on
        how many points, how long it took, and what it was fitted with. Appended, so it survives
        save_surrogate='latest' pruning the .pkl files and a restart continuing the run.
        """
        stats = self._filter_stats or {}
        path = os.path.join(self.cfg.work_dir, 'surrogates', 'surrogate_log.txt')
        values = {'round': ind_round,
                  'surrogate': type(self.surrogate).__name__.replace('Surrogate', ''),
                  'n_raw': stats.get('raw', len(self.train_y)), 'n_trim': stats.get('trimmed', ''),
                  'n_sub': stats.get('subsampled', ''), 'n_fit': num_fit,
                  't_fit_s': '-' if elapsed is None else f'{elapsed:.1f}',
                  'file': os.path.basename(self._surrogate_path(ind_round))}
        details = ('reused the fit saved before the interruption' if elapsed is None
                   else self._surrogate_details(self.surrogate))
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'a') as f:
                if f.tell() == 0:
                    f.write('surrogate log: one line per surrogate fit, appended across restarts\n\n')
                    f.write(''.join(f'{name:>{width}}' for name, width in self._SURROGATE_LOG_COLUMNS)
                            + '  details\n')
                f.write(''.join(f'{values[name]:>{width}}' for name, width in self._SURROGATE_LOG_COLUMNS)
                        + '  ' + details + '\n')
        except OSError as e:
            logging.warning(f'could not write the surrogate log {path}: {e}')

    def _save_surrogate(self, ind_round: int, num_train: int, num_fit: int) -> None:
        """Pickle this round's fit with a fingerprint of its data; written atomically."""
        if self.cfg.save_surrogate is None:
            return
        path = self._surrogate_path(ind_round)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {'surrogate': self.surrogate, 'ind_round': ind_round,
                   'surrogate_class': type(self.surrogate).__name__,
                   'num_train_points': num_train, 'num_fit_points': num_fit,
                   '_slurmcmc_version': __import__('slurmcmc').__version__}
        try:
            with open(path + '.tmp', 'wb') as f:
                pickle.dump(payload, f)
            os.replace(path + '.tmp', path)
            if self.cfg.save_surrogate == 'latest':
                # keep only this round's file; pruning after the write means a crash mid-save
                # never leaves zero surrogates on disk
                for name in os.listdir(os.path.dirname(path)):
                    stale = os.path.join(os.path.dirname(path), name)
                    if name.startswith('round') and name.endswith('.pkl') and stale != path:
                        os.remove(stale)
        except Exception:
            # a surrogate that cannot be pickled (e.g. a user object holding a lambda) must
            # not take the run down with it; the fit itself is unaffected
            logging.warning(f'could not save the surrogate for round {ind_round}; continuing.',
                            exc_info=True)

    def _load_saved_surrogate(self, ind_round: int, num_train: int) -> Optional[Any]:
        """A previously saved fit for this exact round and training set, or None."""
        path = self._surrogate_path(ind_round)
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'rb') as f:
                payload = pickle.load(f)
        except Exception:
            logging.warning(f'could not read {path}; refitting instead.', exc_info=True)
            return None
        if payload.get('ind_round') != ind_round or payload.get('num_train_points') != num_train:
            return None  # stale: a different round, or the training set has moved on
        expected = type(self._create_surrogate()).__name__
        if payload.get('surrogate_class') != expected:
            # resuming with a different surrogate: refit rather than continue with the old model
            logging.info(f'    saved surrogate is a {payload.get("surrogate_class")} but this '
                         f'run uses {expected}; refitting instead of reusing it.')
            return None
        return payload

    def _fit_surrogate(self, ind_round: Optional[int] = None) -> float:
        t_start = time.time()
        X, y = self._training_arrays()

        # a run that died inside a round may have saved that round's fit; reuse it only when
        # resuming and when the round and both data sizes match
        if ind_round is not None and self._resumed and self.cfg.save_surrogate is not None:
            payload = self._load_saved_surrogate(ind_round, len(self.train_y))
            if payload is not None and payload.get('num_fit_points') == len(y):
                self.surrogate = payload['surrogate']
                self._update_confinement(X, y)
                self.num_fit_points_per_round.append(len(y))
                if self.cfg.verbosity >= 1:
                    logging.info(f'    reusing the saved surrogate for round {ind_round} '
                                 f'({len(y)} points); no refit needed.')
                self._append_surrogate_log(ind_round, len(y), elapsed=None)
                self._resumed = False  # only the first round after a resume can reuse a fit
                return 0.0
        self._resumed = False

        self.surrogate = self._create_surrogate()
        self.surrogate.fit(X, y)
        self._update_confinement(X, y)
        elapsed = time.time() - t_start
        self.num_fit_points_per_round.append(len(y))
        if self.cfg.verbosity >= 1:
            logging.info(f'    surrogate fit on {len(y)} points: {elapsed:.2f}s.')
        if ind_round is not None:
            self._save_surrogate(ind_round, len(self.train_y), len(y))
            if self.cfg.save_surrogate is not None:
                self._append_surrogate_log(ind_round, len(y), elapsed)
        return elapsed

    def _update_confinement(self, X: np.ndarray, y: np.ndarray) -> None:
        """Refresh the confinement geometry from the points the surrogate was just fitted on."""
        if not self._unbounded:
            return
        X = np.atleast_2d(np.asarray(X, dtype=float))
        num_params = X.shape[1]
        mean = X.mean(axis=0)
        centered = X - mean
        cov = np.atleast_2d(np.cov(X, rowvar=False)) if len(X) > 1 else np.zeros((num_params, num_params))
        # a ridge keeps this invertible when the training points are degenerate in some
        # direction (collinear, or simply fewer points than parameters)
        scale = float(np.trace(cov)) / max(num_params, 1)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        precision = np.linalg.pinv(cov + 1e-6 * scale * np.eye(num_params))
        radii = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', centered, precision, centered), 0.0))
        radius = float(np.max(radii)) if len(radii) else 0.0
        if not np.isfinite(radius) or radius <= 0:
            radius = 1.0
        span = float(np.max(y) - np.min(y)) if len(y) else 0.0
        if not np.isfinite(span):
            span = 0.0
        self._confinement = {'mean': mean, 'precision': precision, 'radius': radius,
                             'span': max(span, 1.0)}

    def _confinement_radius(self, coords: np.ndarray) -> np.ndarray:
        """Mahalanobis radius of each point, as a multiple of the training set's own radius."""
        conf = self._confinement
        centered = np.atleast_2d(coords) - conf['mean']
        r2 = np.einsum('ij,jk,ik->i', centered, conf['precision'], centered)
        return np.sqrt(np.maximum(r2, 0.0)) / conf['radius']

    def _surrogate_log_prob_batch(self, coords: np.ndarray) -> np.ndarray:
        """
        Vectorized log-prob for emcee: -inf outside param_bounds (or beyond the confinement
        envelope when unbounded) and where the constraint fails; mean - kappa * std inside.
        """
        coords = np.atleast_2d(coords)
        log_prob = np.full(len(coords), -np.inf)
        relative_radius = None
        if self._unbounded:
            if self._confinement is None:
                inside = np.ones(len(coords), dtype=bool)  # nothing fitted yet
            else:
                relative_radius = self._confinement_radius(coords)
                # beyond the hard radius the surrogate is pure invention, and excluding it here
                # also spares the GP the cost of predicting there
                inside = relative_radius <= _CONFINEMENT_HARD_RADIUS
        else:
            inside = np.all((coords >= self._lower) & (coords <= self._upper), axis=1)
        if self.constraint_fun is not None:
            for i in np.where(inside)[0]:
                if self.constraint_fun(*combine_args(coords[i], self.cfg.extra_arg)) > 0:
                    inside[i] = False
        if np.any(inside):
            values = self.surrogate.predict(coords[inside])
            kappa = self.cfg.surrogate_uncertainty_penalty
            if kappa > 0 and hasattr(self.surrogate, 'predict_std'):
                values = values - kappa * np.asarray(self.surrogate.predict_std(coords[inside]))
            if relative_radius is not None:
                excess = np.maximum(relative_radius[inside] / _CONFINEMENT_INFLATION - 1.0, 0.0)
                values = values - _CONFINEMENT_STRENGTH * self._confinement['span'] * excess ** 4
            log_prob[inside] = values
        return log_prob

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _run_expensive_mcmc(self, init_points: np.ndarray, num_iters: int, stage_name: str) -> float:
        """Short expensive MCMC (parallel via SlurmPool); its evaluations join the training set."""
        cfg = self.cfg
        t_start = time.time()
        if cfg.verbosity >= 1:
            logging.info(f'--- hybrid: expensive MCMC stage ({stage_name}), {num_iters} iters.')
        status = slurm_mcmc(log_prob_fun=cfg.log_prob_fun, init_points=init_points, num_iters=num_iters,
                            verbosity=cfg.slurm_verbosity, slurm_verbosity=cfg.slurm_verbosity,
                            work_dir=self._fresh_stage_dir(stage_name), job_name=cfg.job_name + '_' + stage_name,
                            cluster=cfg.cluster, submitit_kwargs=cfg.submitit_kwargs,
                            extra_arg=cfg.extra_arg, job_fail_value=cfg.job_fail_value,
                            install_signal_handler=False,  # the pipeline already installed one
                            **{'keep_run_dirs': cfg.keep_run_dirs, **(cfg.expensive_mcmc_kwargs or {})})
        pool = status['slurm_pool']
        self.num_expensive_evals += pool.num_evaluated_points
        num_added = self._add_training_points(pool.points_history, pool.values_history[:, 0])
        elapsed = time.time() - t_start
        if cfg.verbosity >= 1:
            logging.info(f'    added {num_added} training points '
                         f'({pool.num_evaluated_points} expensive evaluations) in {elapsed:.2f}s.')
        return elapsed

    def _evaluate_expensive_batch(self, points: np.ndarray, stage_name: str) -> np.ndarray:
        """Evaluate the expensive log-prob on a batch of points in parallel."""
        cfg = self.cfg
        pool = SlurmPool(work_dir=self._fresh_stage_dir(stage_name), job_name=cfg.job_name + '_' + stage_name,
                         cluster=cfg.cluster, verbosity=cfg.slurm_verbosity,
                         extra_arg=cfg.extra_arg, submitit_kwargs=dict(cfg.submitit_kwargs or {}),
                         dim_input=self._num_params, dim_output=1,
                         job_fail_value=cfg.job_fail_value, keep_run_dirs=cfg.keep_run_dirs)
        values = pool.map(cfg.log_prob_fun, list(points))
        self.num_expensive_evals += len(points)
        return np.array([float(np.ravel(v)[0]) for v in values])

    def _sample_regularization_points(self) -> np.ndarray:
        """
        Points placed without reference to a posterior: uniform in the box, or, when unbounded,
        from a Gaussian inflated around the training data. Rejection-sampled on the constraint.
        """
        cfg = self.cfg
        if self._unbounded:
            X = np.atleast_2d(np.asarray(self.train_X, dtype=float))
            mean = X.mean(axis=0)
            cov = np.atleast_2d(np.cov(X, rowvar=False)) if len(X) > 1 else np.eye(X.shape[1])
            cov = cov * _CONFINEMENT_INFLATION ** 2 + 1e-9 * np.eye(len(mean))
            draw = lambda: np.random.multivariate_normal(mean, cov)
        else:
            draw = lambda: self._lower + (self._upper - self._lower) * np.random.rand(len(self._lower))
        points = []
        max_tries = 1000 * cfg.num_regularization_points
        tries = 0
        while len(points) < cfg.num_regularization_points and tries < max_tries:
            tries += 1
            point = draw()
            if self.constraint_fun is not None:
                if self.constraint_fun(*combine_args(point, cfg.extra_arg)) > 0:
                    continue
            points.append(point)
        return np.array(points)

    def _num_surrogate_walkers(self) -> int:
        """Walkers for the cheap chain: the configured count, or emcee's minimum."""
        minimum = 2 * self._num_params + 2  # emcee wants an even count strictly above 2*ndim
        if self.cfg.num_surrogate_walkers is None:
            return minimum
        nwalkers = int(self.cfg.num_surrogate_walkers)
        if nwalkers < minimum or nwalkers % 2:
            adjusted = max(nwalkers + (nwalkers % 2), minimum)
            logging.warning(f'    num_surrogate_walkers={nwalkers} is not usable by emcee '
                            f'(needs an even count above {2 * self._num_params}); using {adjusted}.')
            return adjusted
        return nwalkers

    def _chain_burnin(self, num_iters: int) -> int:
        """Steps discarded from the front of this round's chain."""
        return min(int(self.cfg.surrogate_burnin_fraction * num_iters), num_iters - 1)

    def _chain_tau(self, sampler: emcee.EnsembleSampler, num_iters: int) -> float:
        """Autocorrelation time of this round's chain after burn-in (largest over parameters)."""
        try:
            chain = sampler.get_chain(discard=self._chain_burnin(num_iters))
            return float(np.max(emcee.autocorr.integrated_time(chain, quiet=True)))
        except Exception:
            return float(num_iters) / 10.0

    def _chain_ess(self, sampler: emcee.EnsembleSampler, num_iters: int) -> float:
        """Effective sample size of this round's post-burn-in samples."""
        return len(self._burnin_samples(sampler, num_iters)) / max(self._chain_tau(sampler, num_iters), 1.0)

    def _ess_floor(self, final: bool = False) -> float:
        """Effective-sample-size floor: 10*num_params while refining, 100*num_params to report."""
        configured = self.cfg.final_surrogate_ess if final else self.cfg.refine_surrogate_ess
        if configured is not None:
            return float(configured)
        return (100.0 if final else 10.0) * self._num_params

    def _iters_per_tau_floor(self, final: bool = False) -> float:
        """Required L_c/tau. Zero means the criterion is not applied at this tier."""
        configured = self.cfg.final_iters_per_tau if final else self.cfg.refine_iters_per_tau
        return 0.0 if configured is None else float(configured)

    def _chain_is_sufficient(self, sampler: emcee.EnsembleSampler, num_iters: int,
                             final: bool) -> Tuple[bool, float, float]:
        """Whether the cheap chain satisfies both chain criteria; also returns tau and the ESS."""
        tau = self._chain_tau(sampler, num_iters)
        ess = len(self._burnin_samples(sampler, num_iters)) / max(tau, 1.0)
        ess_floor = self._ess_floor(final)
        iters_per_tau = self._iters_per_tau_floor(final)
        enough_samples = (not ess_floor) or ess >= ess_floor
        long_enough = (not iters_per_tau) or num_iters >= iters_per_tau * tau
        return bool(enough_samples and long_enough), tau, ess

    def _run_surrogate_mcmc(self, init_walkers: np.ndarray, sampler: Optional[emcee.EnsembleSampler],
                            final: bool = False):
        """
        Cheap MCMC on the surrogate, extended in blocks on the same sampler until the criteria
        of its depth hold: refining (final=False) or verifying (final=True).
        """
        cfg = self.cfg
        nwalkers, ndim = init_walkers.shape
        if sampler is None:
            sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim,
                                            log_prob_fn=self._surrogate_log_prob_batch, vectorize=True)
        ess_floor = self._ess_floor(final=final)
        state = sampler.run_mcmc(initial_state=init_walkers, nsteps=cfg.num_surrogate_iters,
                                 skip_initial_state_check=True)
        num_iters = cfg.num_surrogate_iters
        max_iters = cfg.num_surrogate_iters * max(1, cfg.max_surrogate_iters_multiplier)
        sufficient, tau, ess = self._chain_is_sufficient(sampler, num_iters, final)
        while not sufficient and num_iters < max_iters:
            extra = min(cfg.num_surrogate_iters, max_iters - num_iters)
            state = sampler.run_mcmc(initial_state=state, nsteps=extra,
                                     skip_initial_state_check=True)
            num_iters += extra
            sufficient, tau, ess = self._chain_is_sufficient(sampler, num_iters, final)
            if cfg.verbosity >= 2:
                logging.info(f'    extended the surrogate chain to {num_iters} iterations '
                             f'(ESS {ess:.0f}, L_c/tau {num_iters / max(tau, 1.0):.0f})')
        self._last_surrogate_iters = num_iters
        self._last_surrogate_tau = tau
        self._last_surrogate_chain_ess = ess

        # hitting the cap short of a criterion must not pass silently: fewer effective samples
        # raise the stability test's noise floor, which makes it easier to pass, not harder
        if cfg.verbosity >= 1 and not sufficient:
            if ess_floor and ess < ess_floor:
                logging.warning(
                    f'    the cheap chain hit its cap of {max_iters} iterations with an effective '
                    f'sample size of {ess:.0f}, short of the floor of {ess_floor:.0f}. '
                    f'Posterior stability is therefore being tested '
                    f'with wider error bars than intended, which makes it easier to pass, not '
                    f'harder. Raise max_surrogate_iters_multiplier (currently '
                    f'{cfg.max_surrogate_iters_multiplier}) or lower refine_surrogate_ess deliberately.')
            iters_per_tau = self._iters_per_tau_floor(final)
            if iters_per_tau and num_iters < iters_per_tau * tau:
                logging.warning(
                    f'    the cheap chain is {num_iters / max(tau, 1.0):.0f} autocorrelation times '
                    f'long (tau = {tau:.0f}), short of L_c >= {iters_per_tau:.0f} '
                    f'tau, so tau itself is probably under-estimated and the effective sample size '
                    f'above is optimistic. This needs a longer chain, not more walkers: raise '
                    f'num_surrogate_iters or max_surrogate_iters_multiplier, or lower '
                    f'num_surrogate_walkers (currently {sampler.nwalkers}).')
        return sampler, state

    def _burnin_samples(self, sampler: emcee.EnsembleSampler, num_iters: int) -> np.ndarray:
        """Post burn-in flat samples. The sampler holds this round only, so nothing else to skip."""
        return sampler.get_chain(discard=self._chain_burnin(num_iters), flat=True)

    def _validate_posterior(self, samples: np.ndarray, samples_previous: Optional[np.ndarray],
                            num_effective_samples: float, ess_previous: Optional[float],
                            stage_name: str) -> Dict[str, Any]:
        """
        Score a posterior against the expensive target: draw validation points from it, evaluate
        them, and return the accuracy and stability metrics with the points themselves.
        """
        cfg = self.cfg
        unique_samples = np.unique(samples, axis=0)
        num_val = min(cfg.num_validation_points, len(unique_samples))
        val_inds = np.random.choice(len(unique_samples), size=num_val, replace=False)
        validation_points = unique_samples[val_inds]
        log_prob_expensive = self._evaluate_expensive_batch(validation_points, stage_name)
        log_prob_surrogate = self.surrogate.predict(validation_points)

        finite = np.isfinite(log_prob_expensive) & (log_prob_expensive != cfg.job_fail_value)
        metrics = self._accuracy_metrics(log_prob_expensive[finite], log_prob_surrogate[finite])

        # the surrogate's own uncertainty on the validation points, to compare with its error
        surrogate_std = np.zeros(0)
        if hasattr(self.surrogate, 'predict_std'):
            try:
                surrogate_std = np.asarray(self.surrogate.predict_std(validation_points))[finite]
            except Exception:
                logging.debug('surrogate predict_std failed; skipping the diagnostic.', exc_info=True)
        metrics['surrogate_std_median'] = float(np.median(surrogate_std)) if len(surrogate_std) else float('nan')
        metrics['posterior_displacement'] = self._posterior_displacement(samples, samples_previous)
        inverse_ess = 1.0 / max(num_effective_samples, 1.0) + 1.0 / max(ess_previous or 1.0, 1.0)
        metrics['posterior_noise_floor'] = _POSTERIOR_NOISE_SIGMAS * float(np.sqrt(inverse_ess))
        return {'metrics': metrics, 'validation_points': validation_points,
                'log_prob_expensive': log_prob_expensive, 'num_val': num_val,
                'log_importance_weights': log_prob_expensive[finite] - log_prob_surrogate[finite],
                'unique_samples': unique_samples}

    def _initial_surrogate_walkers(self) -> np.ndarray:
        """Start walkers at the highest-probability training points (in-bounds by construction)."""
        nwalkers = self._num_surrogate_walkers()
        if len(self.train_y) < nwalkers:
            err_msg = (f'not enough valid training points ({len(self.train_y)}) to initialize '
                       f'{nwalkers} walkers — increase num_expensive_iters or check the filters.')
            logging.error(err_msg)
            raise RuntimeError(err_msg)
        top_inds = np.argsort(self.train_y)[-nwalkers:]
        return np.array([self.train_X[i] for i in top_inds])

    # ------------------------------------------------------------------
    # Convergence metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _accuracy_metrics(log_prob_expensive: np.ndarray, log_prob_surrogate: np.ndarray) -> Dict[str, float]:
        """
        Surrogate-accuracy metrics on validation points drawn from the surrogate posterior.

        d = log p_expensive - log p_surrogate. A *constant* offset in d is irrelevant
        (the posterior is defined up to normalization), so the spread of d is what matters.
        """
        d = log_prob_expensive - log_prob_surrogate

        # the points come from the surrogate posterior q, so expectations under the true p
        # need the importance weights w = p/q = exp(d)
        w = np.exp(d - np.max(d))
        w_sum = np.sum(w)
        if w_sum <= 0 or not np.isfinite(w_sum):
            w = np.ones_like(d)
            w_sum = float(len(d))
        w_norm = w / w_sum

        # how many points effectively carry the weight
        ess_weights = float(1.0 / np.sum(w_norm ** 2))

        # posterior-weighted RMS error about the weighted mean, in nats
        d_mean = float(np.sum(w_norm * d))
        weighted_log_error = float(np.sqrt(np.sum(w_norm * (d - d_mean) ** 2)))

        return {'weighted_log_error': weighted_log_error,
                'ess_weights': ess_weights,
                'log_weight_std': float(np.std(d))}

    @staticmethod
    def _posterior_displacement(samples_new: np.ndarray, samples_old: Optional[np.ndarray]) -> float:
        """
        The largest change in any parameter's mean or standard deviation, in units of that
        parameter's standard deviation: 0.1 means nothing moved by a tenth of a posterior width.
        """
        if samples_old is None or len(samples_old) == 0 or samples_new is None:
            return float('inf')
        mean_new, mean_old = np.mean(samples_new, axis=0), np.mean(samples_old, axis=0)
        std_new, std_old = np.std(samples_new, axis=0), np.std(samples_old, axis=0)
        scale = np.maximum(0.5 * (std_new + std_old), 1e-12)
        return float(np.max(np.maximum(np.abs(mean_new - mean_old) / scale,
                                       np.abs(std_new - std_old) / scale)))

    def _blocking_criteria(self, metrics: Dict[str, float]) -> List[str]:
        """Which criteria are currently unmet -- names the actual bottleneck of a round."""
        cfg = self.cfg
        blocking = []
        if metrics['weighted_log_error'] > cfg.log_error_threshold:
            blocking.append('accuracy')
        if metrics['ess_weights'] < cfg.min_ess_weights:
            blocking.append('weight-degeneracy')
        if (cfg.posterior_shift_tolerance is not None
                and metrics.get('posterior_displacement', float('inf'))
                > max(cfg.posterior_shift_tolerance, metrics.get('posterior_noise_floor', 0.0))):
            blocking.append('posterior-stability')
        return blocking

    def _is_converged(self, metrics: Dict[str, float]) -> bool:
        cfg = self.cfg
        # a few dominant weights make the weighted error spuriously small: never converged then
        weights_usable = metrics['ess_weights'] >= cfg.min_ess_weights
        if not weights_usable and cfg.verbosity >= 1:
            logging.warning(
                f'    importance weights are degenerate (effective count '
                f'{metrics["ess_weights"]:.1f} < {cfg.min_ess_weights}): the weighted '
                f'error is not meaningful, treating this round as not converged.')
        err_ok = weights_usable and metrics['weighted_log_error'] <= cfg.log_error_threshold
        stable_ok = (cfg.posterior_shift_tolerance is None
                     or metrics.get('posterior_displacement', float('inf'))
                     <= max(cfg.posterior_shift_tolerance, metrics.get('posterior_noise_floor', 0.0)))
        return err_ok and stable_ok

    @staticmethod
    def _error_trend(errors: List[float]) -> Dict[str, float]:
        """
        Least-squares trend of log(error) against round index.

        ``rate`` is d log(eps)/d round, so it is negative while the surrogate is improving
        and ``1 - exp(rate)`` is the fractional improvement per round.
        """
        from scipy import stats

        y = np.log(np.asarray(errors, dtype=float))
        x = np.arange(len(y), dtype=float)
        fit = stats.linregress(x, y)
        return {'rate': float(fit.slope), 'stderr': float(fit.stderr),
                'improvement_per_round': float(1.0 - np.exp(fit.slope))}

    def _warn_if_stalled(self, error_per_round: List[float], converged: bool) -> None:
        """
        Warn when refinement is demonstrably not paying for itself.

        eps scatters by a factor of ~2 between rounds, so progress is read from the trend of
        log(eps) over the whole history, and the warning needs positive evidence that the
        improvement rate is below stall_relative_improvement -- noise stays silent.
        """
        cfg = self.cfg
        if converged or len(error_per_round) < cfg.stall_window:
            return
        errors = [e for e in error_per_round if np.isfinite(e) and e > 0]
        if len(errors) < cfg.stall_window:
            return

        trend = self._error_trend(errors)
        rate, stderr = trend['rate'], trend['stderr']
        if not np.isfinite(stderr) or stderr <= 0:
            return

        # the rate the user would consider worth paying for, as a (negative) log rate
        useful_rate = np.log(1.0 - cfg.stall_relative_improvement)
        # one-sided z for "the true rate is worse than useful_rate"
        z = float(stats_norm_ppf(1.0 - cfg.stall_significance))
        if rate <= useful_rate + z * stderr:
            return  # cannot demonstrate inadequate progress; say nothing

        latest = error_per_round[-1]
        rounds_needed = ((np.log(cfg.log_error_threshold) - np.log(latest)) / rate
                         if rate < 0 and latest > 0 else float('inf'))
        outlook = (f'at the fitted rate that is about {rounds_needed:.0f} more rounds to reach '
                   f'{cfg.log_error_threshold}'
                   if np.isfinite(rounds_needed) else 'at the fitted rate it never gets there')
        logging.warning(
            f'    refinement is not improving fast enough to be worth continuing: over '
            f'{len(errors)} rounds weighted_log_error is changing by '
            f'{100 * trend["improvement_per_round"]:+.1f}% per round (target: at least '
            f'{100 * cfg.stall_relative_improvement:.0f}% improvement per round), and '
            f'{outlook}. Either the surrogate cannot represent this target much better, or '
            f'the new training points are landing where they add little — inspect where the '
            f'error lives before spending more rounds.')

    def _warn_if_posterior_at_bounds(self, samples: np.ndarray) -> None:
        """Warn when posterior mass piles up against param_bounds (or the confinement envelope)."""
        if samples is None or len(samples) == 0:
            return
        if self._unbounded:
            self._warn_if_posterior_at_confinement(samples)
            return
        span = self._upper - self._lower
        tolerance = self.cfg.bounds_warning_tolerance
        for ind_param in range(samples.shape[1]):
            column = samples[:, ind_param]
            at_low = np.mean(column <= self._lower[ind_param] + tolerance * span[ind_param])
            at_high = np.mean(column >= self._upper[ind_param] - tolerance * span[ind_param])
            fraction = max(at_low, at_high)
            if fraction >= self.cfg.bounds_warning_fraction:
                edge = 'lower' if at_low >= at_high else 'upper'
                logging.warning(
                    f'    posterior of parameter {ind_param} piles up against its {edge} bound '
                    f'({fraction:.0%} of samples within {tolerance:.0%} of it). This can indicate '
                    f'model-form error, an unidentifiable parameter, or too-narrow param_bounds.')

    def _warn_if_posterior_at_confinement(self, samples: np.ndarray) -> None:
        """Unbounded counterpart: warn when the posterior reaches where the envelope acts."""
        if self._confinement is None:
            return
        fraction = float(np.mean(self._confinement_radius(samples) > _CONFINEMENT_INFLATION))
        if fraction >= self.cfg.bounds_warning_fraction:
            logging.warning(
                f'    {fraction:.0%} of the posterior lies beyond '
                f'{_CONFINEMENT_INFLATION:g}x the training data Mahalanobis radius, where '
                f'the confinement envelope already penalises the surrogate, so the tails are being '
                f'shaped by the envelope rather than by the target. Push expensive evaluations '
                f'further out (num_regularization_points, or more rounds) before reading them at '
                f'face value.')

    # ------------------------------------------------------------------
    # Restart
    # ------------------------------------------------------------------

    #: keys of the per-round accumulators carried across a restart
    _ACCUMULATOR_KEYS = ('weighted_log_error_per_round',
                         'log_weight_std_per_round', 'surrogate_std_per_round',
                         'posterior_displacement_per_round', 'blocking_per_round',
                         'surrogate_ess_per_round', 'surrogate_iters_per_round',
                         'surrogate_tau_per_round', 'verified_per_round', 'round_records',
                         'num_train_points_per_round', 'diagnostics_rows')

    #: config fields that do not change what a run computes, ignored when reporting changes on restart
    _NON_BEHAVIOURAL_FIELDS = frozenset({
        'log_prob_fun', 'init_points', 'constraint_fun', 'surrogate', 'extra_arg',
        'verbosity', 'slurm_verbosity', 'log_file', 'work_dir', 'job_name', 'cluster',
        'submitit_kwargs', 'expensive_mcmc_kwargs', 'job_fail_value',
        'save_restart', 'load_restart', 'restart_file', 'status_restart', 'save_surrogate', 'keep_run_dirs',
        'install_signal_handler', 'remote', 'remote_cluster', 'remote_submitit_kwargs',
    })

    def _config_snapshot(self) -> Dict[str, Any]:
        """The settings that actually change what a run computes, for restart provenance."""
        snapshot = {}
        for field in dataclasses.fields(self.cfg):
            if field.name in self._NON_BEHAVIOURAL_FIELDS:
                continue
            value = getattr(self.cfg, field.name)
            if isinstance(value, (int, float, str, bool, type(None))):
                snapshot[field.name] = value
        return snapshot

    def _save_posterior_samples(self, ind_round: int, samples: np.ndarray) -> None:
        """Keep a capped draw from this round's posterior; nothing else records it."""
        num_samples = self.cfg.save_posterior_samples
        if not num_samples:
            return
        directory = os.path.join(self.cfg.work_dir, 'posteriors')
        try:
            os.makedirs(directory, exist_ok=True)
            if len(samples) > num_samples:
                kept = np.random.default_rng(0).choice(len(samples), num_samples, replace=False)
                samples = samples[kept]
            np.save(os.path.join(directory, f'round{ind_round}.npy'), samples)
        except OSError as e:
            logging.warning(f'could not save posterior samples for round {ind_round}: {e}')

    def _save_restart(self, ini_round: int, accumulators: Dict[str, list],
                      init_walkers: np.ndarray, samples_previous_round: Optional[np.ndarray],
                      ess_previous_round: Optional[float] = None,
                      converged: bool = False) -> None:
        """Write the state needed to resume at ``ini_round``. Atomic (see save_restart_file)."""
        cfg = self.cfg
        if not cfg.save_restart:
            return
        status = {
            'ini_round': ini_round,
            'converged': converged,
            # recorded so a surrogate switched on resume is reported
            'surrogate_class': type(self.surrogate).__name__ if self.surrogate else None,
            'config_snapshot': self._config_snapshot(),
            'train_X': list(self.train_X),
            'train_y': list(self.train_y),
            'train_points_set': set(self._train_points_set),
            'num_expensive_evals': self.num_expensive_evals,
            'init_walkers': np.asarray(init_walkers),
            # only the previous posterior's mean and std are needed: a two-point stand-in
            # {m-s, m+s} reproduces both exactly
            'posterior_summary': (None if samples_previous_round is None or
                                  len(samples_previous_round) == 0 else
                                  {'mean': np.mean(samples_previous_round, axis=0),
                                   'std': np.std(samples_previous_round, axis=0),
                                   'ess': ess_previous_round}),
            'num_fit_points_per_round': list(self.num_fit_points_per_round),
            'timings': dict(self.timings),
            'timings_per_round': list(self.timings_per_round),
            # taken after the round's last random draw, so a resume reproduces an uninterrupted run
            'random_state': get_random_state(),
        }
        status.update({key: list(accumulators[key]) for key in self._ACCUMULATOR_KEYS})
        if cfg.verbosity >= 2:
            logging.info(f'    saving restart file: {os.path.join(cfg.work_dir, cfg.restart_file)}')
        save_restart_file(status, cfg.work_dir, cfg.restart_file)

    def _load_restart(self, accumulators: Dict[str, list]) -> tuple:
        """Restore the state written by _save_restart. Returns (ini_round, walkers, samples, ess)."""
        cfg = self.cfg
        if cfg.status_restart is not None:
            if cfg.verbosity >= 1:
                logging.info('--- hybrid: restarting from status_restart argument.')
            status = cfg.status_restart
        else:
            if cfg.verbosity >= 1:
                logging.info(f'--- hybrid: loading restart file: '
                             f'{os.path.join(cfg.work_dir, cfg.restart_file)}')
            status = load_restart_file(cfg.work_dir, cfg.restart_file)

        self.train_X = list(status['train_X'])
        self.train_y = list(status['train_y'])
        self._train_points_set = set(status['train_points_set'])
        self.num_expensive_evals = status['num_expensive_evals']
        self.num_fit_points_per_round = list(status.get('num_fit_points_per_round', []))
        self.timings = dict(status['timings'])
        self.timings_per_round = list(status['timings_per_round'])
        set_random_state(status.get('random_state'))
        # lists added in later versions are missing from older restart files: pad them
        num_previous_rounds = len(status.get('weighted_log_error_per_round', []))
        for key in self._ACCUMULATOR_KEYS:
            if key in status:
                accumulators[key][:] = list(status[key])
            else:
                fill = False if key == 'verified_per_round' else float('nan')
                accumulators[key][:] = [fill] * num_previous_rounds
        # changing settings on resume is legitimate, but has to be reported
        saved_config = status.get('config_snapshot') or {}
        current_config = self._config_snapshot()
        changes = {key: (saved_config[key], current_config[key])
                   for key in sorted(set(saved_config) & set(current_config))
                   if saved_config[key] != current_config[key]}
        if changes:
            summary = ', '.join(f'{key}: {was!r} -> {now!r}' for key, (was, now) in changes.items())
            logging.warning(f'--- hybrid: SETTINGS CHANGED ON RESTART at round '
                            f'{status["ini_round"]}: {summary}')
        self._config_changes_on_restart = changes

        previous = status.get('surrogate_class')
        current = type(self._create_surrogate()).__name__
        if previous and previous != current:
            logging.warning(
                f'--- hybrid: SURROGATE CHANGED ON RESTART: rounds 1-{status["ini_round"] - 1} '
                f'were run with {previous}, rounds {status["ini_round"]}+ will use {current}. '
                f'The training data carries over, but per-round diagnostics before and after '
                f'this point come from different models and are not directly comparable.')
        self._surrogate_switched_at = (status['ini_round']
                                       if previous and previous != current else None)
        self._previous_surrogate_class = previous

        if status.get('converged'):
            logging.warning(
                'restart file is from a run that already converged; resuming it will not '
                'repeat the final round, so the returned samples will be empty. Raise '
                'num_rounds_max only if you intend to refine an already-converged surrogate.')
        if status['ini_round'] > cfg.num_rounds_max:
            logging.warning(
                f'restart file resumes at round {status["ini_round"]}, but num_rounds_max is '
                f'{cfg.num_rounds_max}: no refinement round will run and the returned samples '
                f'will be empty. Raise num_rounds_max to continue this run.')
        elif cfg.verbosity >= 1:
            logging.info(f'--- hybrid: resumed at round {status["ini_round"]} with '
                         f'{len(self.train_y)} training points and '
                         f'{self.num_expensive_evals} expensive evaluations already spent.')
        summary = status.get('posterior_summary')
        samples_previous_round = (None if summary is None else
                                  np.vstack([summary['mean'] - summary['std'],
                                             summary['mean'] + summary['std']]))
        ess_previous_round = None if summary is None else summary.get('ess')
        return (status['ini_round'], np.asarray(status['init_walkers']), samples_previous_round,
                ess_previous_round)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        cfg = self.cfg
        set_logging(cfg.work_dir, cfg.log_file)
        if cfg.install_signal_handler:
            signal.signal(signal.SIGTERM, signal_handler)

        t_start_total = time.time()
        # seeded in this process, so remote runs too; a resume then restores the saved state
        seed_random_generators(cfg.random_seed)

        if cfg.constraint_fun is not None:
            self.constraint_fun = deferred_import_function_wrapper(cfg.constraint_fun)

        weighted_log_error_per_round: List[float] = []
        log_weight_std_per_round: List[float] = []
        surrogate_std_per_round: List[float] = []
        surrogate_tau_per_round: List[float] = []
        verified_per_round: List[bool] = []
        round_records: List[Dict[str, Any]] = []
        verification_attempts = 0
        ess_previous_round: Optional[float] = None
        verification_budget_spent = False
        posterior_displacement_per_round: List[float] = []
        blocking_per_round: List[List[str]] = []
        surrogate_ess_per_round: List[float] = []
        surrogate_iters_per_round: List[int] = []
        num_train_points_per_round: List[int] = []
        diagnostics_rows: List[Dict[str, Any]] = []
        accumulators = {
            'weighted_log_error_per_round': weighted_log_error_per_round,
            'log_weight_std_per_round': log_weight_std_per_round,
            'surrogate_std_per_round': surrogate_std_per_round,
            'surrogate_tau_per_round': surrogate_tau_per_round,
            'verified_per_round': verified_per_round,
            'round_records': round_records,
            'posterior_displacement_per_round': posterior_displacement_per_round,
            'blocking_per_round': blocking_per_round,
            'surrogate_ess_per_round': surrogate_ess_per_round,
            'surrogate_iters_per_round': surrogate_iters_per_round,
            'num_train_points_per_round': num_train_points_per_round,
            'diagnostics_rows': diagnostics_rows,
        }

        if cfg.verbosity >= 1:
            # the log file is appended to across restarts, so mark where each run starts
            logging.info('=' * 70)
            logging.info(f'--- hybrid: {"RESUMING" if (cfg.load_restart or cfg.status_restart) else "STARTING"} '
                         f'run, surrogate = {type(self._create_surrogate()).__name__}, '
                         f'work_dir = {cfg.work_dir}')

        if cfg.load_restart or cfg.status_restart is not None:
            (ini_round, init_walkers, samples_previous_round,
             ess_previous_round) = self._load_restart(accumulators)
            self._resumed = True
            if len(init_walkers) != self._num_surrogate_walkers():
                # num_surrogate_walkers changed on resume: restart the walkers from the training set
                logging.warning(
                    f'--- hybrid: num_surrogate_walkers changed on restart '
                    f'({len(init_walkers)} saved, {self._num_surrogate_walkers()} configured); '
                    f're-seeding the cheap chain from the best training points.')
                init_walkers = self._initial_surrogate_walkers()
        else:
            self.timings['expensive_mcmc_0'] = 0.0
            self.timings['regularization'] = 0.0
            if cfg.initial_train_points is not None:
                # existing evaluations replace round 0
                num_added = self._add_training_points(cfg.initial_train_points,
                                                      cfg.initial_train_values)
                if cfg.verbosity >= 1:
                    logging.info(f'--- hybrid: seeded with {len(cfg.initial_train_values)} existing '
                                 f'evaluations ({num_added} usable after filtering); round 0 skipped.')
                if cfg.num_regularization_points > 0:
                    logging.warning(f'--- hybrid: num_regularization_points={cfg.num_regularization_points} '
                                    f'is ignored: the regularization points are part of round 0, '
                                    f'which a start from initial_train_points skips.')
            else:
                # ---- round 0: expensive MCMC + optional regularization points ----
                self.timings['expensive_mcmc_0'] = self._run_expensive_mcmc(
                    np.array(cfg.init_points), cfg.num_expensive_iters, 'round0_expensive_mcmc')

                self.timings['regularization'] = 0.0
                if cfg.num_regularization_points > 0:
                    t_start = time.time()
                    reg_points = self._sample_regularization_points()
                    if len(reg_points) > 0:
                        reg_values = self._evaluate_expensive_batch(reg_points, 'round0_regularization')
                        num_added = self._add_training_points(reg_points, reg_values)
                        self.timings['regularization'] = time.time() - t_start
                        if cfg.verbosity >= 1:
                            logging.info(f'--- hybrid: regularization points: added {num_added} '
                                         f'({len(reg_points)} expensive evaluations) in '
                                         f'{self.timings["regularization"]:.2f}s.')
            ini_round = 1
            init_walkers = self._initial_surrogate_walkers()
            samples_previous_round = None
            # round 0 is expensive in its own right, so protect it before any refinement starts
            self._save_restart(ini_round, accumulators, init_walkers, samples_previous_round,
                               ess_previous_round)

        # ---- refinement rounds ----
        # each round samples a different surrogate, so only the walker positions carry over;
        # bound here so a resumed run whose loop body never executes still has these
        sampler = None
        converged = False
        samples = None
        validation_points = None
        log_importance_weights = None

        for ind_round in range(ini_round, cfg.num_rounds_max + 1):
            t_start_round = time.time()
            if cfg.verbosity >= 1:
                logging.info(f'--- hybrid: refinement round {ind_round}/{cfg.num_rounds_max} '
                             f'({len(self.train_y)} training points).')
            num_train_points_per_round.append(len(self.train_y))

            # fit surrogate
            time_fit = self._fit_surrogate(ind_round)

            # run the cheap MCMC on the surrogate
            t_start = time.time()
            sampler, state = self._run_surrogate_mcmc(init_walkers, None)
            init_walkers = state.coords  # next round (if any) continues from here
            samples = self._burnin_samples(sampler, self._last_surrogate_iters)
            time_surrogate_mcmc = time.time() - t_start
            if cfg.verbosity >= 1:
                logging.info(f'    surrogate MCMC ({self._last_surrogate_iters} iters, '
                             f'ESS {self._chain_ess(sampler, self._last_surrogate_iters):.0f}'
                             f'/{self._ess_floor():.0f}): {time_surrogate_mcmc:.2f}s.')

            # validation: expensive evaluations on held-out surrogate-posterior samples
            t_start = time.time()
            tau_samples = self._last_surrogate_tau   # computed once by the extension loop
            if not np.isfinite(tau_samples) or tau_samples <= 0:
                tau_samples = float(self._last_surrogate_iters) / 10.0
            num_effective_samples = len(samples) / max(tau_samples, 1.0)
            scored = self._validate_posterior(samples, samples_previous_round,
                                              num_effective_samples, ess_previous_round,
                                              f'round{ind_round}_validation')
            metrics = scored['metrics']
            validation_points = scored['validation_points']
            log_prob_expensive = scored['log_prob_expensive']
            log_importance_weights = scored['log_importance_weights']
            unique_samples = scored['unique_samples']
            num_val = scored['num_val']
            time_validation = time.time() - t_start

            weighted_log_error_per_round.append(metrics['weighted_log_error'])
            log_weight_std_per_round.append(metrics['log_weight_std'])
            surrogate_std_per_round.append(metrics['surrogate_std_median'])
            posterior_displacement_per_round.append(metrics['posterior_displacement'])
            if cfg.verbosity >= 1:
                logging.info(f'    validation ({num_val} expensive evals): {time_validation:.2f}s.')
                logging.info(f'    weighted_log_error = {metrics["weighted_log_error"]:.4f} nats '
                             f'(threshold {cfg.log_error_threshold}), '
                             f'surrogate std = {metrics["surrogate_std_median"]:.4f} nats, '
                             f'posterior shift = {metrics["posterior_displacement"]:.3f} sd '
                             f'(tolerance {max(cfg.posterior_shift_tolerance or 0.0, metrics["posterior_noise_floor"]):.3f}, '
                             f'of which noise floor {metrics["posterior_noise_floor"]:.3f}).')

            # verification overwrites `metrics`, so keep the refinement pass's row now
            refinement_record = {'round': ind_round, 'verified': False,
                                 'weighted_log_error': metrics['weighted_log_error'],
                                 'posterior_displacement': metrics['posterior_displacement'],
                                 'surrogate_iters': self._last_surrogate_iters,
                                 'surrogate_tau': tau_samples,
                                 'chain_ess': num_effective_samples}
            refinement_row = {
                'round': ind_round,
                'w_log_err': f'{metrics["weighted_log_error"]:.4f}',
                'post_shift': f'{metrics["posterior_displacement"]:.3f}',
                'surr_iters': self._last_surrogate_iters,
                'Lc/tau': f'{self._last_surrogate_iters / max(tau_samples, 1.0):.1f}',
                'ess/tau': f'{num_effective_samples:.0f}',
                't_mcmc_s': f'{time_surrogate_mcmc:.1f}',
                't_valid_s': f'{time_validation:.1f}',
            }
            refinement_mcmc_seconds = time_surrogate_mcmc
            refinement_validation_seconds = time_validation

            converged = self._is_converged(metrics)
            verified_this_round = False

            # verification: a pass on the cheap chain is provisional until this same surrogate,
            # re-sampled to reporting depth, passes one more validation batch
            attempts_left = (cfg.max_verification_attempts is None
                             or verification_attempts < cfg.max_verification_attempts)
            if converged and attempts_left:
                verification_attempts += 1
                verified_this_round = True
                # recycle the refinement batch now, before verification replaces it; it cannot
                # bias the verification, which scores the surrogate already fitted
                self._add_training_points(validation_points, log_prob_expensive)
                if cfg.verbosity >= 1:
                    logging.info(
                        f'    provisionally converged; verification '
                        f'{verification_attempts}/{cfg.max_verification_attempts}: re-sampling the '
                        f'surrogate to reporting depth (ESS floor {self._ess_floor(final=True):.0f}, '
                        f'L_c >= {cfg.final_iters_per_tau:.0f} tau).')
                t_start = time.time()
                sampler, state = self._run_surrogate_mcmc(init_walkers, None, final=True)
                init_walkers = state.coords
                samples = self._burnin_samples(sampler, self._last_surrogate_iters)
                time_surrogate_mcmc += time.time() - t_start
                tau_samples = self._last_surrogate_tau
                if not np.isfinite(tau_samples) or tau_samples <= 0:
                    tau_samples = float(self._last_surrogate_iters) / 10.0
                num_effective_samples = len(samples) / max(tau_samples, 1.0)

                t_start = time.time()
                scored = self._validate_posterior(samples, samples_previous_round,
                                                  num_effective_samples, ess_previous_round,
                                                  f'round{ind_round}_verification')
                metrics = scored['metrics']
                validation_points = scored['validation_points']
                log_prob_expensive = scored['log_prob_expensive']
                log_importance_weights = scored['log_importance_weights']
                unique_samples = scored['unique_samples']
                num_val = scored['num_val']
                time_validation += time.time() - t_start
                converged = self._is_converged(metrics)

                # the verification measures this same round, better; it replaces the round's
                # recorded metrics rather than adding a phantom round
                weighted_log_error_per_round[-1] = metrics['weighted_log_error']
                log_weight_std_per_round[-1] = metrics['log_weight_std']
                surrogate_std_per_round[-1] = metrics['surrogate_std_median']
                posterior_displacement_per_round[-1] = metrics['posterior_displacement']
                if cfg.verbosity >= 1:
                    logging.info(
                        f'    verification: {self._last_surrogate_iters} iters, ESS '
                        f'{num_effective_samples:.0f}, weighted_log_error = '
                        f'{metrics["weighted_log_error"]:.4f}, posterior shift = '
                        f'{metrics["posterior_displacement"]:.3f} -> '
                        f'{"accepted" if converged else "rejected, refining again"}.')

            elif converged:
                # no verification attempts left: an unverified pass is not a pass, and no later
                # round could be verified either, so stop
                converged = False
                verification_budget_spent = True
                if cfg.verbosity >= 1:
                    logging.warning(
                        f'    provisionally converged, but all {cfg.max_verification_attempts} '
                        f'verification attempts are spent, so this posterior was never re-sampled '
                        f'to reporting depth. Stopping without reporting convergence: raise '
                        f'max_verification_attempts (None = as many as needed) or loosen the '
                        f'thresholds.')

            samples_previous_round = samples
            ess_previous_round = num_effective_samples
            self._save_posterior_samples(ind_round, samples)
            blocking = self._blocking_criteria(metrics)
            blocking_per_round.append(blocking)
            surrogate_ess_per_round.append(num_effective_samples)
            surrogate_iters_per_round.append(self._last_surrogate_iters)
            surrogate_tau_per_round.append(tau_samples)
            verified_per_round.append(verified_this_round)
            if cfg.verbosity >= 1 and not converged:
                logging.info(f'    not converged; blocking criteria: {", ".join(blocking)}')
            self._warn_if_stalled(weighted_log_error_per_round, converged)
            time_refresh = 0.0

            if not converged:
                # recycle the validation evaluations as training data — they sit exactly
                # where the surrogate needs to improve
                self._add_training_points(validation_points, log_prob_expensive)

                # optional expensive-MCMC refresh from the current surrogate posterior
                if cfg.num_expensive_iters_per_round > 0:
                    nwalkers = len(cfg.init_points)
                    refresh_inds = np.random.choice(len(unique_samples),
                                                    size=min(nwalkers, len(unique_samples)), replace=False)
                    time_refresh = self._run_expensive_mcmc(
                        unique_samples[refresh_inds], cfg.num_expensive_iters_per_round,
                        f'round{ind_round}_expensive_refresh')

            time_round = time.time() - t_start_round
            round_timings = {'surrogate_fit': time_fit, 'surrogate_mcmc': time_surrogate_mcmc,
                             'validation': time_validation, 'expensive_refresh': time_refresh,
                             'total': time_round}
            self.timings_per_round.append(round_timings)
            # one record per measurement: a verified round contributes its refinement and its
            # verification; the per-round lists keep only the latter
            if verified_this_round:
                round_records.append(refinement_record)
            round_records.append({'round': ind_round, 'verified': verified_this_round,
                                  'weighted_log_error': metrics['weighted_log_error'],
                                  'posterior_displacement': metrics['posterior_displacement'],
                                  'surrogate_iters': self._last_surrogate_iters,
                                  'surrogate_tau': tau_samples,
                                  'chain_ess': num_effective_samples})
            common = {'n_expensive': num_train_points_per_round[-1],
                      'n_train': self.num_fit_points_per_round[-1],
                      'surrogate': type(self.surrogate).__name__.replace('Surrogate', '')}
            if verified_this_round:
                refinement_row.update(common)
                refinement_row['t_fit_s'] = f'{time_fit:.1f}'   # the fit belongs to the round, once
                diagnostics_rows.append(refinement_row)
                row_mcmc_seconds = time_surrogate_mcmc - refinement_mcmc_seconds
                row_validation_seconds = time_validation - refinement_validation_seconds
            else:
                row_mcmc_seconds, row_validation_seconds = time_surrogate_mcmc, time_validation
            diagnostics_rows.append({
                **({} if verified_this_round else common),  # already on the refinement row
                'round': f'{ind_round}v' if verified_this_round else ind_round,
                'w_log_err': f'{metrics["weighted_log_error"]:.4f}',
                'post_shift': f'{metrics["posterior_displacement"]:.3f}',
                'surr_iters': self._last_surrogate_iters,
                'Lc/tau': f'{self._last_surrogate_iters / max(tau_samples, 1.0):.1f}',
                'ess/tau': f'{num_effective_samples:.0f}',
                't_fit_s': '' if verified_this_round else f'{time_fit:.1f}',
                't_mcmc_s': f'{row_mcmc_seconds:.1f}',
                't_valid_s': f'{row_validation_seconds:.1f}',
                't_refresh_s': f'{time_refresh:.1f}',
                't_round_s': f'{time_round:.1f}',
            })
            self._write_diagnostics(diagnostics_rows)
            self._save_restart(ind_round + 1, accumulators, init_walkers,
                               samples_previous_round, ess_previous_round, converged=converged)

            # a spent verification budget ends the run too: no later round could be verified
            if converged or verification_budget_spent:
                break

        self._warn_if_posterior_at_bounds(samples)

        total_time = time.time() - t_start_total
        if cfg.verbosity >= 1:
            logging.info(f'--- hybrid: done in {total_time:.2f}s. converged={converged}, '
                         f'num_expensive_evals={self.num_expensive_evals}.')

        return {
            'converged': converged,
            'num_rounds': len(weighted_log_error_per_round),
            'samples': samples,  # flat post-burn-in surrogate-posterior samples of the final round
            'validation_points': validation_points,
            'log_importance_weights': log_importance_weights,  # for the final validation subset
            'weighted_log_error_per_round': weighted_log_error_per_round,
            'log_weight_std_per_round': log_weight_std_per_round,
            'surrogate_std_per_round': surrogate_std_per_round,
            'surrogate_tau_per_round': surrogate_tau_per_round,
            'verified_per_round': verified_per_round,
            'round_records': round_records,
            'posterior_displacement_per_round': posterior_displacement_per_round,
            'blocking_criteria_per_round': blocking_per_round,
            'surrogate_ess_per_round': surrogate_ess_per_round,
            'surrogate_iters_per_round': surrogate_iters_per_round,
            'num_train_points_per_round': num_train_points_per_round,
            'num_fit_points_per_round': list(self.num_fit_points_per_round),
            'surrogate_class_per_round': [row['surrogate'] for row in diagnostics_rows if 'surrogate' in row],
            'surrogate_switched_at_round': getattr(self, '_surrogate_switched_at', None),
            'config_changes_on_restart': dict(self._config_changes_on_restart),
            'num_expensive_evals': self.num_expensive_evals,
            'verification_attempts': verification_attempts,
            'surrogate': self.surrogate,
            'sampler': sampler,
            'train_X': np.array(self.train_X),
            'train_y': np.array(self.train_y),
            'timings': self.timings,
            'timings_per_round': self.timings_per_round,
            'total_time': total_time,
        }


def run_hybrid_mcmc(config: HybridMCMCConfig) -> Dict:
    """Run a HybridMCMCRunner from a config object. Module-level so remote mode can pickle it by reference."""
    return HybridMCMCRunner(config).run()


def slurm_mcmc_hybrid(
        log_prob_fun: Union[Callable, Dict],
        init_points: Optional[np.ndarray] = None,
        param_bounds: Optional[List] = None,
        constraint_fun: Optional[Union[Callable, Dict]] = None,
        num_expensive_iters: int = 20,
        initial_train_points: Optional[np.ndarray] = None,
        initial_train_values: Optional[np.ndarray] = None,
        num_regularization_points: int = 0,
        num_expensive_iters_per_round: int = 0,
        surrogate: Union[str, Any] = 'gp',
        polynomial_degree: int = 3,
        num_surrogate_iters: int = 2000,
        num_surrogate_walkers: Optional[int] = None,
        final_surrogate_ess: Optional[float] = None,
        max_verification_attempts: Optional[int] = None,
        refine_iters_per_tau: Optional[float] = None,
        final_iters_per_tau: float = 50.0,
        surrogate_burnin_fraction: float = 0.2,
        num_validation_points: int = 100,
        surrogate_uncertainty_penalty: float = 1.0,
        log_error_threshold: float = 0.1,
        min_ess_weights: float = 10.0,
        posterior_shift_tolerance: Optional[float] = 0.1,
        refine_surrogate_ess: Optional[float] = None,
        max_surrogate_iters_multiplier: int = 8,
        num_rounds_max: int = 5,
        train_log_prob_trim_range: Optional[float] = 100.0,
        min_train_points_after_trim: int = 50,
        max_train_points: Optional[int] = None,
        train_log_prob_floor: Optional[float] = None,
        save_surrogate: Optional[Literal['latest', 'all']] = None,
        save_posterior_samples: Optional[int] = None,
        save_restart: bool = False,
        load_restart: bool = False,
        restart_file: str = 'hybrid_restart.pkl',
        status_restart: Optional[Dict] = None,
        verbosity: int = 1,
        slurm_verbosity: int = 0,
        log_file: Optional[str] = None,
        extra_arg: Any = None,
        work_dir: str = 'mcmc_hybrid',
        job_name: str = 'mcmc_hybrid',
        cluster: Cluster = 'slurm',
        submitit_kwargs: Optional[Dict] = None,
        job_fail_value: float = -1e10,
        expensive_mcmc_kwargs: Optional[Dict] = None,
        keep_run_dirs: KeepRunDirs = 'all',
        install_signal_handler: bool = True,
        random_seed: Optional[int] = None,
        # remote run params:
        remote: bool = False,
        remote_cluster: Literal['slurm', 'local'] = 'slurm',
        remote_submitit_kwargs: Optional[Dict] = None,
) -> Union[Dict, submitit.Job]:
    """
    Automated hybrid surrogate-MCMC: iteratively build a fast surrogate of an
    expensive log-probability function that is accurate on the posterior, and
    return a converged MCMC distribution — using far fewer expensive
    evaluations than a full expensive MCMC.

    See the module docstring of slurmcmc.hybrid for the algorithm and the convergence
    metrics. This is a thin convenience wrapper around HybridMCMCConfig + HybridMCMCRunner.

    Key parameters
    --------------
    init_points : array (num_walkers, num_params), optional
        Starting walkers of the round-0 expensive MCMC. Required unless initial_train_points
        (with initial_train_values) is given, in which case round 0 is skipped; still required
        then if num_expensive_iters_per_round > 0, whose chains take their walker count from it.
    param_bounds : list of [lo, hi] per parameter, optional
        A hard box that constrains the posterior. None (default) runs unbounded, confined by an
        envelope grown from the training data. Give a box only for real physical limits; a
        guessed one can truncate the posterior.
    surrogate : 'gp' (default, recommended) | 'polynomial' | object
        Any object with fit(X, y) and predict(X) can be plugged in.
    random_seed : int or None
        Seed for every random choice of the run, applied inside the process that runs the loop,
        so remote runs are reproducible too; restart files store the generator state.
    keep_run_dirs : 'all' | 'failed' | 'none'
        What to keep of each batch's directory under work_dir once its results are in: 'all'
        (default) keeps everything, for investigating crashes or re-using the points' own output
        files; 'failed' keeps only batches with a failed evaluation; 'none' removes them all.
        Every point and result is appended to points_history.txt / values_history.txt either way.
    log_error_threshold : float
        Convergence threshold on the posterior-weighted RMS error of the surrogate
        log-probability, in nats (0.02 means "probability ratios accurate to ~2%").
    num_expensive_iters_per_round : int
        If > 0, each refinement round also runs this many expensive MCMC steps
        initialized from the current surrogate posterior (extra training data
        for hard posteriors). Default 0 — validation recycling is usually enough.

    Returns
    -------
    dict with keys: 'converged', 'samples' (flat post-burn-in surrogate-posterior
    samples), 'log_importance_weights' + 'validation_points' (exact reweighting
    on the validation subset), 'weighted_log_error_per_round', 'surrogate',
    'num_expensive_evals', 'timings',
    'timings_per_round', and more (see HybridMCMCRunner.run).
    With remote=True, returns a submitit Job whose .result() is that dict.
    """
    config = HybridMCMCConfig(
        log_prob_fun=log_prob_fun, init_points=init_points, param_bounds=param_bounds,
        constraint_fun=constraint_fun,
        num_expensive_iters=num_expensive_iters,
        initial_train_points=initial_train_points,
        initial_train_values=initial_train_values,
        num_regularization_points=num_regularization_points,
        num_expensive_iters_per_round=num_expensive_iters_per_round,
        surrogate=surrogate, polynomial_degree=polynomial_degree,
        num_surrogate_iters=num_surrogate_iters,
        num_surrogate_walkers=num_surrogate_walkers,
        final_surrogate_ess=final_surrogate_ess,
        max_verification_attempts=max_verification_attempts,
        refine_iters_per_tau=refine_iters_per_tau,
        final_iters_per_tau=final_iters_per_tau,
        surrogate_burnin_fraction=surrogate_burnin_fraction,
        num_validation_points=num_validation_points,
        surrogate_uncertainty_penalty=surrogate_uncertainty_penalty,
        log_error_threshold=log_error_threshold,
        min_ess_weights=min_ess_weights,
        posterior_shift_tolerance=posterior_shift_tolerance,
        refine_surrogate_ess=refine_surrogate_ess,
        max_surrogate_iters_multiplier=max_surrogate_iters_multiplier,
        num_rounds_max=num_rounds_max,
        train_log_prob_trim_range=train_log_prob_trim_range,
        min_train_points_after_trim=min_train_points_after_trim,
        max_train_points=max_train_points,
        train_log_prob_floor=train_log_prob_floor,
        save_surrogate=save_surrogate,
        save_posterior_samples=save_posterior_samples,
        save_restart=save_restart, load_restart=load_restart,
        restart_file=restart_file, status_restart=status_restart,
        verbosity=verbosity, slurm_verbosity=slurm_verbosity, log_file=log_file,
        extra_arg=extra_arg, work_dir=work_dir, job_name=job_name, cluster=cluster,
        submitit_kwargs=submitit_kwargs, job_fail_value=job_fail_value,
        expensive_mcmc_kwargs=expensive_mcmc_kwargs, keep_run_dirs=keep_run_dirs,
        install_signal_handler=install_signal_handler, random_seed=random_seed,
        remote=remote, remote_cluster=remote_cluster, remote_submitit_kwargs=remote_submitit_kwargs,
    )

    if config.remote:
        set_logging(config.work_dir, config.log_file)
        if config.install_signal_handler:
            signal.signal(signal.SIGTERM, signal_handler)
        print('Running slurm_mcmc_hybrid remotely.')
        return submit_remote_run(run_hybrid_mcmc, dataclasses.replace(config, remote=False),
                                 config.work_dir, config.job_name,
                                 config.remote_cluster, config.remote_submitit_kwargs)

    return HybridMCMCRunner(config).run()
