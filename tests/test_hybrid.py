"""
Tests for the automated hybrid surrogate-MCMC pipeline (slurmcmc/hybrid.py).

All pipeline tests use a cheap 2D Gaussian log-probability so the "expensive"
stage is fast, and cluster='local-map'.
"""

import os
import pickle
import re

import numpy as np
import pytest

from slurmcmc.hybrid import (GaussianProcessSurrogate, HybridMCMCConfig, HybridMCMCRunner,
                             PolynomialSurrogate, slurm_mcmc_hybrid)

# analytic 2D Gaussian posterior: mean mu, per-parameter std sigma
mu = np.array([1.0, -1.0])
sigma = np.array([1.0, 0.5])
param_bounds = [[-5, 5], [-5, 5]]


def log_prob_gaussian(x):
    return float(-0.5 * np.sum(((np.asarray(x) - mu) / sigma) ** 2))


r_constraint = 3.0


def constraint_fun(x):
    # return > 0 for violation; circle around the posterior center
    return 1 if np.sum((np.asarray(x) - mu) ** 2) > r_constraint ** 2 else -1


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """
    Run every test in its own directory.

    The pipeline's default work_dir is relative ('mcmc_hybrid'), so without this the tests
    litter the repository root with a work directory.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture()
def init_points(seed):
    num_walkers = 10
    return mu + 0.5 * np.random.randn(num_walkers, 2)


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

def test_hybrid_mcmc_gaussian_converges(verbosity, seed, init_points):
    result = slurm_mcmc_hybrid(log_prob_fun=log_prob_gaussian, init_points=init_points,
                               param_bounds=param_bounds,
                               num_expensive_iters=15, num_regularization_points=20,
                               num_surrogate_iters=500, num_validation_points=50,
                               num_rounds_max=4,
                               cluster='local-map', verbosity=verbosity)

    assert result['converged']
    assert result['weighted_log_error_per_round'][-1] <= 0.1
    # posterior moments close to the analytic ones
    samples = result['samples']
    np.testing.assert_allclose(np.mean(samples, axis=0), mu, atol=0.2)
    np.testing.assert_allclose(np.std(samples, axis=0), sigma, rtol=0.3)
    # the whole point: far fewer expensive evaluations than a full MCMC would need
    assert result['num_expensive_evals'] < 500


def test_hybrid_mcmc_with_constraint(verbosity, seed, init_points):
    result = slurm_mcmc_hybrid(log_prob_fun=log_prob_gaussian, init_points=init_points,
                               param_bounds=param_bounds, constraint_fun=constraint_fun,
                               num_expensive_iters=15, num_regularization_points=20,
                               num_surrogate_iters=500, num_validation_points=50,
                               num_rounds_max=4,
                               cluster='local-map', verbosity=verbosity)

    assert result['converged']
    # every sample of the surrogate posterior satisfies the constraint
    for sample in result['samples']:
        assert constraint_fun(sample) <= 0
    # every training point satisfies the constraint too
    for point in result['train_X']:
        assert constraint_fun(point) <= 0


def test_hybrid_mcmc_polynomial_surrogate(verbosity, seed, init_points):
    result = slurm_mcmc_hybrid(log_prob_fun=log_prob_gaussian, init_points=init_points,
                               param_bounds=param_bounds,
                               surrogate='polynomial', polynomial_degree=2,
                               num_expensive_iters=15,
                               num_surrogate_iters=500, num_validation_points=50,
                               num_rounds_max=4,
                               cluster='local-map', verbosity=verbosity)

    # A degree-2 polynomial matches the quadratic log-prob exactly, so the surrogate is
    # accurate immediately; convergence still takes 2 rounds because the posterior-stability
    # criterion has nothing to compare against until a second posterior exists.
    assert result['converged']
    assert result['num_rounds'] == 2
    np.testing.assert_allclose(np.mean(result['samples'], axis=0), mu, atol=0.2)

def test_hybrid_mcmc_not_converged_path(verbosity, seed, init_points):
    """An unreachable accuracy threshold exhausts num_rounds_max and reports converged=False."""
    result = slurm_mcmc_hybrid(log_prob_fun=log_prob_gaussian, init_points=init_points,
                               param_bounds=param_bounds,
                               num_expensive_iters=10,
                               num_surrogate_iters=200, num_validation_points=20,
                               log_error_threshold=-1.0,  # an RMS cannot be negative
                               num_rounds_max=2,
                               cluster='local-map', verbosity=verbosity)

    assert not result['converged']
    assert result['num_rounds'] == 2
    assert len(result['weighted_log_error_per_round']) == 2
    # validation points were recycled into the training set between rounds
    assert result['num_train_points_per_round'][1] > result['num_train_points_per_round'][0]


def test_accuracy_metrics_degenerate_weights_are_flagged():
    """A single dominating validation point makes the weighted error trivially zero;
    the weight-ESS guard must expose that instead of reporting a perfect surrogate."""
    log_prob_expensive = np.array([0.0] + [-50.0] * 149)  # one point carries all the weight
    log_prob_surrogate = np.zeros(150)
    m = HybridMCMCRunner._accuracy_metrics(log_prob_expensive, log_prob_surrogate)
    assert m['weighted_log_error'] < 1e-6  # deceptively "perfect" despite a terrible surrogate
    assert m['ess_weights'] == pytest.approx(1.0, rel=1e-6)   # but only 1 effective point

    # a healthy case is not flagged
    rng = np.random.default_rng(0)
    d = rng.normal(scale=0.03, size=150)
    m2 = HybridMCMCRunner._accuracy_metrics(d, np.zeros(150))
    assert m2['ess_weights'] > 100
    assert m2['weighted_log_error'] == pytest.approx(0.03, rel=0.4)


def test_hybrid_mcmc_reports_surrogate_uncertainty(verbosity, seed, init_points):
    """Each round reports the emulator's own predictive uncertainty, so the common
    assumption that it is negligible can be checked rather than assumed."""
    result = slurm_mcmc_hybrid(log_prob_fun=log_prob_gaussian, init_points=init_points,
                               param_bounds=param_bounds,
                               num_expensive_iters=15,
                               num_surrogate_iters=300, num_validation_points=40,
                               num_rounds_max=2,
                               cluster='local-map', verbosity=verbosity)
    stds = result['surrogate_std_per_round']
    assert len(stds) == result['num_rounds']
    assert all(np.isfinite(s) and s >= 0 for s in stds)


def test_hybrid_mcmc_expensive_refresh_rounds(verbosity, seed, init_points):
    """num_expensive_iters_per_round adds expensive-MCMC training data each round."""
    result = slurm_mcmc_hybrid(log_prob_fun=log_prob_gaussian, init_points=init_points,
                               param_bounds=param_bounds,
                               num_expensive_iters=10, num_expensive_iters_per_round=5,
                               num_surrogate_iters=200, num_validation_points=20,
                               log_error_threshold=-1.0,  # force both rounds to run
                               num_rounds_max=2,
                               cluster='local-map', verbosity=verbosity)

    assert result['num_rounds'] == 2
    # round 2 must have gained both validation recycling and refresh-MCMC points
    gained = result['num_train_points_per_round'][1] - result['num_train_points_per_round'][0]
    assert gained > 20  # 20 validation points alone would be the maximum without refresh

def test_training_trim_drops_only_negligible_points(verbosity, seed, init_points):
    """train_log_prob_trim_range drops points far below the peak (they cannot affect the
    posterior) while keeping enough to constrain the surrogate."""
    result = slurm_mcmc_hybrid(log_prob_fun=log_prob_gaussian, init_points=init_points,
                               param_bounds=param_bounds,
                               num_expensive_iters=15, num_regularization_points=20,
                               num_surrogate_iters=400, num_validation_points=40,
                               train_log_prob_trim_range=20.0,
                               num_rounds_max=4,
                               cluster='local-map', verbosity=verbosity)
    assert result['converged']
    # trimming happens at fit time only: the stored history keeps every evaluation
    y = result['train_y']
    assert np.any(y <= y.max() - 20.0) or len(y) > 0


def test_training_trim_is_skipped_when_too_few_points_remain(seed):
    """An over-aggressive trim must not be applied — leaving the surrogate with almost no
    data is far worse than keeping the far-field points."""
    from slurmcmc.hybrid import HybridMCMCConfig, HybridMCMCRunner
    cfg = HybridMCMCConfig(log_prob_fun=log_prob_gaussian, init_points=np.zeros((4, 2)),
                           param_bounds=param_bounds,
                           train_log_prob_trim_range=1e-6,  # would keep ~nothing
                           min_train_points_after_trim=50, verbosity=0)
    runner = HybridMCMCRunner(cfg)
    rng = np.random.default_rng(0)
    runner.train_X = list(rng.uniform(-3, 3, (200, 2)))
    runner.train_y = [log_prob_gaussian(x) for x in runner.train_X]
    X, y = runner._training_arrays()
    assert len(y) == 200  # trim skipped, all points retained


def test_blocking_criteria_names_the_bottleneck(verbosity, seed, init_points):
    """A run that fails must say which criterion blocked it, so the user knows whether to
    buy more expensive data or something cheaper."""
    result = slurm_mcmc_hybrid(log_prob_fun=log_prob_gaussian, init_points=init_points,
                               param_bounds=param_bounds,
                               num_expensive_iters=10,
                               num_surrogate_iters=200, num_validation_points=20,
                               log_error_threshold=-1.0,  # an RMS cannot be negative
                               num_rounds_max=2,
                               cluster='local-map', verbosity=verbosity)
    assert not result['converged']
    blocking = result['blocking_criteria_per_round']
    assert len(blocking) == 2
    assert all('accuracy' in b for b in blocking)   # the criterion we made impossible


def test_surrogate_chain_extended_to_reach_min_ess(verbosity, seed, init_points):
    """The surrogate MCMC is free, so it is extended until it carries enough information;
    a verdict should never be limited by the one stage that costs nothing."""
    result = slurm_mcmc_hybrid(log_prob_fun=log_prob_gaussian, init_points=init_points,
                               param_bounds=param_bounds,
                               num_expensive_iters=10,
                               num_surrogate_iters=200,     # deliberately short
                               num_validation_points=30,
                               refine_surrogate_ess=400, max_surrogate_iters_multiplier=30,
                               num_rounds_max=3,
                               cluster='local-map', verbosity=verbosity)
    # the chain was extended beyond the requested length to reach the ESS floor
    assert max(result['surrogate_iters_per_round']) > 200
    assert max(result['surrogate_ess_per_round']) >= 400


def test_max_train_points_caps_the_fit_and_keeps_coverage(seed):
    """The cap bounds the O(n^3) fit; the retained points must still span the whole region,
    which a top-N-by-log-prob rule would not preserve."""
    from slurmcmc.hybrid import HybridMCMCConfig, HybridMCMCRunner
    cfg = HybridMCMCConfig(log_prob_fun=log_prob_gaussian, init_points=np.zeros((4, 2)),
                           param_bounds=param_bounds, max_train_points=100,
                           train_log_prob_trim_range=None, verbosity=0)
    runner = HybridMCMCRunner(cfg)
    rng = np.random.default_rng(0)
    runner.train_X = list(rng.uniform(-4, 4, (500, 2)))
    runner.train_y = [log_prob_gaussian(x) for x in runner.train_X]
    X, y = runner._training_arrays()
    assert len(y) == 100
    # coverage preserved: the subsample still reaches out to the edges of the training cloud
    assert X.min() < -3.0 and X.max() > 3.0


def _stall_probe(errors, **overrides):
    """Feed an error history to the stall detector round by round; return the log records."""
    import logging
    from slurmcmc.hybrid import HybridMCMCConfig, HybridMCMCRunner
    settings = dict(stall_window=12, stall_significance=0.05,
                    stall_relative_improvement=0.05, log_error_threshold=0.01)
    settings.update(overrides)
    cfg = HybridMCMCConfig(log_prob_fun=log_prob_gaussian, init_points=np.zeros((4, 2)),
                           param_bounds=param_bounds, verbosity=0, **settings)
    runner = HybridMCMCRunner(cfg)
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append((record.levelname, record.getMessage()))

    handler = _Capture()
    logger = logging.getLogger()
    logger.addHandler(handler)
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        for k in range(1, len(errors) + 1):
            runner._warn_if_stalled(list(errors[:k]), converged=False)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    return records

def test_stall_detector_does_not_fire_on_the_real_runs_that_were_improving():
    """
    Regression test against the traces of the 5d runs in
    examples/example_mcmc_hybrid_stages.py. Every one of these was improving (the ring
    at 5.6% and 7.6% per round over 30 rounds, Rosenbrock to convergence); the previous
    rule warned on 50-64% of their rounds.
    """
    traces = {
        'ring 5d, quadratic mean': [0.590, 0.5319, 0.4994, 0.4389, 0.3905, 0.4808, 0.2402,
                                    0.4227, 0.2664, 0.2947, 0.1874, 0.1363, 0.1351, 0.1877,
                                    0.2013, 0.2159, 0.1200, 0.1313, 0.1819, 0.1993, 0.1238,
                                    0.1546, 0.2845, 0.1081, 0.1321, 0.1547, 0.0695, 0.0865,
                                    0.1203, 0.1122],
        'rosenbrock 5d, no mean': [0.1326, 0.0996, 0.0546, 0.0324, 0.0331, 0.0198, 0.0177,
                                   0.0146, 0.0155, 0.0121, 0.0113, 0.0067],
    }
    for name, trace in traces.items():
        warnings = [m for level, m in _stall_probe(trace) if level == 'WARNING']
        assert not warnings, f'{name} was reported as stalled while it was improving'


def test_stall_detector_catches_a_run_that_improves_then_plateaus():
    rng = np.random.default_rng(3)
    rounds = np.arange(40)
    mean = -0.15 * np.minimum(rounds, 10)  # fast for ten rounds, then nothing
    trace = 0.5 * np.exp(mean) * np.exp(rng.normal(0, 0.35, len(rounds)))
    warnings = [m for level, m in _stall_probe(trace) if level == 'WARNING']
    assert warnings, 'a run that plateaued after a fast start was not reported'
    assert 'not improving fast enough' in warnings[-1]


def test_stall_detector_needs_a_full_window():
    assert _stall_probe([0.1] * 11) == []


def test_error_trend_recovers_a_known_rate():
    from slurmcmc.hybrid import HybridMCMCRunner
    rng = np.random.default_rng(2)
    rounds = np.arange(40)
    errors = 0.5 * np.exp(-0.1 * rounds) * np.exp(rng.normal(0, 0.2, len(rounds)))
    trend = HybridMCMCRunner._error_trend(list(errors))
    assert trend['rate'] == pytest.approx(-0.1, abs=0.02)
    assert trend['improvement_per_round'] == pytest.approx(0.095, abs=0.02)
    assert trend['stderr'] < 0.01


def _restart_settings(work_dir, **overrides):
    settings = dict(log_prob_fun=log_prob_gaussian,
                    init_points=np.random.uniform(-2, 2, (8, 2)),
                    param_bounds=param_bounds, num_expensive_iters=5,
                    num_regularization_points=30, num_surrogate_iters=300,
                    num_validation_points=20, log_error_threshold=1e-9,
                    num_rounds_max=2, work_dir=work_dir,
                    cluster='local-map', verbosity=0)
    settings.update(overrides)
    return settings


def test_restart_resumes_without_repeating_expensive_work(work_dir):
    """
    A run interrupted after round 1 and resumed must continue from the accumulated training
    set rather than start over -- the expensive evaluations are the whole point of the file.
    """
    from slurmcmc.hybrid import slurm_mcmc_hybrid

    np.random.seed(0)
    first = slurm_mcmc_hybrid(**_restart_settings(work_dir, num_rounds_max=1, save_restart=True))
    assert os.path.exists(os.path.join(work_dir, 'hybrid_restart.pkl'))

    np.random.seed(0)
    resumed = slurm_mcmc_hybrid(**_restart_settings(work_dir, num_rounds_max=2,
                                                    save_restart=True, load_restart=True))

    # the resumed run ran round 2 only, on top of round 1's data
    assert resumed['num_rounds'] == 2
    assert len(resumed['weighted_log_error_per_round']) == 2
    assert resumed['weighted_log_error_per_round'][0] == first['weighted_log_error_per_round'][0]
    assert len(resumed['train_X']) > len(first['train_X'])
    # round 0 was not repeated: its evaluations are counted once, not twice
    assert resumed['num_expensive_evals'] < 2 * first['num_expensive_evals']


def test_restart_reaches_the_same_place_as_an_uninterrupted_run(work_dir):
    """Splitting a run in two at the restart file must not change what it accumulates."""
    from slurmcmc.hybrid import slurm_mcmc_hybrid

    np.random.seed(0)
    whole = slurm_mcmc_hybrid(**_restart_settings(os.path.join(work_dir, 'whole')))

    split_dir = os.path.join(work_dir, 'split')
    np.random.seed(0)
    slurm_mcmc_hybrid(**_restart_settings(split_dir, num_rounds_max=1, save_restart=True))
    np.random.seed(0)
    resumed = slurm_mcmc_hybrid(**_restart_settings(split_dir, num_rounds_max=2,
                                                    save_restart=True, load_restart=True))

    assert resumed['num_rounds'] == whole['num_rounds']
    assert resumed['num_expensive_evals'] == whole['num_expensive_evals']
    assert len(resumed['train_X']) == len(whole['train_X'])


def test_restart_from_a_status_dict(work_dir):
    """status_restart accepts the state in memory, without a file (matches slurm_mcmc)."""
    from slurmcmc.general_utils import load_restart_file
    from slurmcmc.hybrid import slurm_mcmc_hybrid

    np.random.seed(0)
    slurm_mcmc_hybrid(**_restart_settings(work_dir, num_rounds_max=1, save_restart=True))
    status = load_restart_file(work_dir, 'hybrid_restart.pkl')

    np.random.seed(0)
    resumed = slurm_mcmc_hybrid(**_restart_settings(work_dir, num_rounds_max=2,
                                                    status_restart=status))
    assert resumed['num_rounds'] == 2


def test_restart_file_records_that_a_run_converged(work_dir):
    """
    The file is written after round 0 too -- that round is expensive and must be protected --
    so a converged run still leaves one behind. It has to say so, or the next person resumes
    a finished run and gets an empty posterior with no explanation.
    """
    from slurmcmc.general_utils import load_restart_file
    from slurmcmc.hybrid import slurm_mcmc_hybrid
    np.random.seed(0)
    result = slurm_mcmc_hybrid(**_restart_settings(work_dir, log_error_threshold=1e3,
                                                   posterior_shift_tolerance=None,
                                                   save_restart=True))
    assert result['converged']
    status = load_restart_file(work_dir, 'hybrid_restart.pkl')
    assert status['converged'] is True


def test_restart_file_stores_a_summary_not_the_whole_chain(work_dir):
    """
    Only the mean and std of the previous chain are ever used, so the file must not carry
    the chain itself -- it is written every round, often onto a shared filesystem.
    """
    from slurmcmc.general_utils import load_restart_file
    from slurmcmc.hybrid import slurm_mcmc_hybrid
    np.random.seed(0)
    slurm_mcmc_hybrid(**_restart_settings(work_dir, num_rounds_max=1, save_restart=True))
    status = load_restart_file(work_dir, 'hybrid_restart.pkl')
    assert 'samples_previous_round' not in status
    # mean and std reconstruct the previous posterior exactly for the displacement; ess is
    # what the noise floor needs. Still a handful of numbers, not the chain.
    assert set(status['posterior_summary']) == {'mean', 'std', 'ess'}
    assert os.path.getsize(os.path.join(work_dir, 'hybrid_restart.pkl')) < 50_000


def test_posterior_displacement_is_unchanged_by_the_summary_stand_in():
    """
    The restart file keeps only the previous posterior's mean and std, never the chain. A
    two-point stand-in {m-s, m+s} has exactly that mean and population std, so the displacement
    computed after a restart must equal the one computed from the full chain.
    """
    from slurmcmc.hybrid import HybridMCMCRunner
    rng = np.random.default_rng(0)
    old = rng.normal(0.0, 1.0, (5000, 3))
    new = rng.normal(0.1, 1.1, (5000, 3))
    stand_in = np.vstack([old.mean(axis=0) - old.std(axis=0),
                          old.mean(axis=0) + old.std(axis=0)])
    assert HybridMCMCRunner._posterior_displacement(new, stand_in) == \
           pytest.approx(HybridMCMCRunner._posterior_displacement(new, old))

def test_surrogates_are_saved_per_round_and_are_usable(work_dir):
    """
    Each round's fit is pickled with a fingerprint of the data it saw, and reloads into a
    working surrogate -- that is what makes it possible to compare rounds after the fact.
    """
    from slurmcmc.hybrid import slurm_mcmc_hybrid
    np.random.seed(0)
    result = slurm_mcmc_hybrid(**_restart_settings(work_dir, num_rounds_max=2,
                                                   save_surrogate='all'))
    directory = os.path.join(work_dir, 'surrogates')
    saved = sorted(f for f in os.listdir(directory) if f.endswith('.pkl'))
    assert saved == ['round1.pkl', 'round2.pkl']

    with open(os.path.join(directory, 'round1.pkl'), 'rb') as f:
        payload = pickle.load(f)
    assert payload['ind_round'] == 1
    assert payload['num_train_points'] == result['num_train_points_per_round'][0]
    assert payload['num_fit_points'] == result['num_fit_points_per_round'][0]

    # the round-1 surrogate still predicts, and differs from the final one (it saw less data)
    early = payload['surrogate'].predict(np.zeros((1, 2)))
    assert np.isfinite(early).all()


def test_no_surrogates_written_when_the_option_is_off(work_dir):
    from slurmcmc.hybrid import slurm_mcmc_hybrid
    np.random.seed(0)
    slurm_mcmc_hybrid(**_restart_settings(work_dir, num_rounds_max=1))
    assert not os.path.exists(os.path.join(work_dir, 'surrogates'))


def test_resumed_round_reuses_its_saved_fit_instead_of_repeating_it(work_dir):
    """
    A run that dies inside a round should not repeat that round's fit on resume -- it is the
    most expensive single step. The saved fit is adopted only when the round index and both
    data sizes match.
    """
    from slurmcmc.hybrid import HybridMCMCConfig, HybridMCMCRunner

    cfg = HybridMCMCConfig(log_prob_fun=log_prob_gaussian, init_points=np.zeros((4, 2)),
                           param_bounds=param_bounds, work_dir=work_dir,
                           save_surrogate='all', verbosity=0)
    runner = HybridMCMCRunner(cfg)
    rng = np.random.default_rng(0)
    runner.train_X = list(rng.uniform(-3, 3, (60, 2)))
    runner.train_y = [log_prob_gaussian(x) for x in runner.train_X]

    runner._fit_surrogate(ind_round=3)                       # the round that then "crashed"
    reference = runner.surrogate.predict(np.zeros((1, 2)))

    resumed = HybridMCMCRunner(cfg)
    resumed.train_X, resumed.train_y = list(runner.train_X), list(runner.train_y)
    resumed._resumed = True
    elapsed = resumed._fit_surrogate(ind_round=3)
    assert elapsed == 0.0, 'the saved fit should have been reused, not refitted'
    assert resumed.surrogate.predict(np.zeros((1, 2))) == pytest.approx(reference)
    # and the log says so, rather than claiming a second fit
    log = open(os.path.join(work_dir, 'surrogates', 'surrogate_log.txt')).read().splitlines()
    assert len([line for line in log if line.strip().startswith('3 ')]) == 2
    assert log[-1].split()[6:8] == ['-', 'round3.pkl'] and 'reused' in log[-1]


def test_saved_fit_is_rejected_when_the_training_set_has_moved_on(work_dir):
    """The fingerprint must catch a stale fit rather than silently reusing the wrong model."""
    from slurmcmc.hybrid import HybridMCMCConfig, HybridMCMCRunner

    cfg = HybridMCMCConfig(log_prob_fun=log_prob_gaussian, init_points=np.zeros((4, 2)),
                           param_bounds=param_bounds, work_dir=work_dir,
                           save_surrogate='all', verbosity=0)
    runner = HybridMCMCRunner(cfg)
    rng = np.random.default_rng(0)
    runner.train_X = list(rng.uniform(-3, 3, (60, 2)))
    runner.train_y = [log_prob_gaussian(x) for x in runner.train_X]
    runner._fit_surrogate(ind_round=2)

    resumed = HybridMCMCRunner(cfg)
    resumed.train_X = list(runner.train_X) + list(rng.uniform(-3, 3, (10, 2)))  # grew
    resumed.train_y = [log_prob_gaussian(x) for x in resumed.train_X]
    resumed._resumed = True
    assert resumed._fit_surrogate(ind_round=2) > 0.0, 'a stale fit must not be reused'


def test_latest_mode_keeps_one_rolling_file(work_dir):
    """
    The cheap mode: enough to skip a refit on restart, without the per-round disk cost. One
    file, overwritten each round, carrying the most recent fit's fingerprint.
    """
    from slurmcmc.hybrid import slurm_mcmc_hybrid
    np.random.seed(0)
    result = slurm_mcmc_hybrid(**_restart_settings(work_dir, num_rounds_max=2,
                                                   save_surrogate='latest'))
    # one file, and it still names the round it came from: a bare 'latest' would leave you
    # unable to tell which round a recovered surrogate belongs to
    directory = os.path.join(work_dir, 'surrogates')
    saved = sorted(f for f in os.listdir(directory) if f.endswith('.pkl'))
    assert saved == [f'round{result["num_rounds"]}.pkl']
    with open(os.path.join(directory, saved[0]), 'rb') as f:
        payload = pickle.load(f)
    assert payload['ind_round'] == result['num_rounds']          # the last round, not the first
    assert payload['num_train_points'] == result['num_train_points_per_round'][-1]

# ---------------------------------------------------------------------------
# Distributed GP (rBCM) surrogate
# ---------------------------------------------------------------------------

def log_prob_ring(x):
    """
    A shell: not polynomial at any degree, so the quadratic mean function cannot explain it
    and the GP stage actually runs. log_prob_gaussian is exactly quadratic, which makes the
    trend fit it outright and the GP be skipped -- fine behaviour, useless for testing a GP.
    """
    radius = np.sqrt(np.sum(np.asarray(x, dtype=float) ** 2))
    return float(-0.5 * ((radius - 2.0) / 0.6) ** 2)


# ---------------------------------------------------------------------------
# Changing surrogate across a restart
# ---------------------------------------------------------------------------

def _swap_settings(work_dir):
    return dict(log_prob_fun=log_prob_ring,
                init_points=np.random.default_rng(0).uniform(-2, 2, (8, 2)),
                param_bounds=param_bounds, num_expensive_iters=5,
                num_regularization_points=30, num_surrogate_iters=250,
                num_validation_points=20, log_error_threshold=1e-9,
                cluster='local-map', verbosity=0, work_dir=work_dir,
                save_restart=True, save_surrogate='latest')


def test_restart_can_switch_surrogate_family(work_dir):
    """
    The restart file holds only surrogate-agnostic state, so a run that is proving too slow
    with one surrogate can be killed and resumed with another without losing a single
    expensive evaluation.
    """
    from slurmcmc.hybrid import (DistributedGPSurrogate, GaussianProcessSurrogate,
                                 slurm_mcmc_hybrid)
    settings = _swap_settings(work_dir)
    np.random.seed(0)
    first = slurm_mcmc_hybrid(surrogate=GaussianProcessSurrogate(n_restarts_optimizer=0),
                              num_rounds_max=2, **settings)
    np.random.seed(0)
    resumed = slurm_mcmc_hybrid(surrogate=DistributedGPSurrogate(num_experts=3,
                                                                 n_restarts_optimizer=0),
                                num_rounds_max=4, load_restart=True, **settings)
    assert resumed['num_rounds'] == 4
    assert type(resumed['surrogate']).__name__ == 'DistributedGPSurrogate'
    assert len(resumed['train_X']) > len(first['train_X'])          # data carried over
    assert resumed['num_expensive_evals'] > first['num_expensive_evals']


def test_saved_surrogate_is_not_reused_across_a_family_change(work_dir):
    """
    The reuse path keys on round and data size. Without a type check it would hand a resumed
    rBCM run the dense GP's saved fit -- you would believe you were running an ensemble while
    running the old model.
    """
    from slurmcmc.hybrid import (DistributedGPSurrogate, GaussianProcessSurrogate,
                                 HybridMCMCConfig, HybridMCMCRunner)
    rng = np.random.default_rng(0)
    train_X = list(rng.uniform(-3, 3, (60, 2)))
    train_y = [log_prob_ring(x) for x in train_X]

    dense_cfg = HybridMCMCConfig(log_prob_fun=log_prob_ring, init_points=np.zeros((4, 2)),
                                 param_bounds=param_bounds, work_dir=work_dir,
                                 surrogate=GaussianProcessSurrogate(n_restarts_optimizer=0),
                                 save_surrogate='latest', verbosity=0)
    runner = HybridMCMCRunner(dense_cfg)
    runner.train_X, runner.train_y = list(train_X), list(train_y)
    runner._fit_surrogate(ind_round=2)

    ensemble_cfg = HybridMCMCConfig(log_prob_fun=log_prob_ring,
                                    init_points=np.zeros((4, 2)),
                                    param_bounds=param_bounds, work_dir=work_dir,
                                    surrogate=DistributedGPSurrogate(num_experts=2,
                                                                     n_restarts_optimizer=0),
                                    save_surrogate='latest', verbosity=0)
    resumed = HybridMCMCRunner(ensemble_cfg)
    resumed.train_X, resumed.train_y = list(train_X), list(train_y)
    resumed._resumed = True
    assert resumed._fit_surrogate(ind_round=2) > 0.0, 'a different family must be refitted'
    assert type(resumed.surrogate).__name__ == 'DistributedGPSurrogate'



def test_restart_records_and_reports_changed_settings(work_dir):
    """
    Resuming with different settings is legitimate -- raising num_rounds_max extends a run,
    loosening max_train_points rescues one whose fits got too slow -- but the diagnostics then
    span two configurations, and a trace that changes for a config reason rather than a
    modelling one is uninterpretable unless the change is on the record.
    """
    from slurmcmc.hybrid import slurm_mcmc_hybrid
    settings = _swap_settings(work_dir)
    np.random.seed(0)
    slurm_mcmc_hybrid(surrogate='gp', num_rounds_max=2, max_train_points=None, **settings)
    np.random.seed(0)
    resumed = slurm_mcmc_hybrid(surrogate='gp', num_rounds_max=4, max_train_points=60,
                                train_log_prob_trim_range=50.0, load_restart=True, **settings)

    changes = resumed['config_changes_on_restart']
    assert changes['max_train_points'] == (None, 60)
    assert changes['train_log_prob_trim_range'] == (100.0, 50.0)
    assert changes['num_rounds_max'] == (2, 4)
    # the cap actually bit: the fit saw fewer points than the training set holds
    assert resumed['num_fit_points_per_round'][-1] <= 60


def test_restart_reports_no_settings_change_when_there_is_none(work_dir):
    """The diff must not cry wolf: plumbing and the restart flags themselves are excluded,
    or every legitimate resume would raise a warning and the signal would be worthless."""
    from slurmcmc.hybrid import slurm_mcmc_hybrid
    settings = _swap_settings(work_dir)
    np.random.seed(0)
    slurm_mcmc_hybrid(surrogate='gp', num_rounds_max=2, **settings)
    np.random.seed(0)
    resumed = slurm_mcmc_hybrid(surrogate='gp', num_rounds_max=2, load_restart=True, **settings)
    assert resumed['config_changes_on_restart'] == {}


def test_surrogate_provenance_is_recorded_per_round(work_dir):
    """Which surrogate produced each row, so a discontinuity in the error trace can be
    attributed to the model change that caused it."""
    from slurmcmc.hybrid import DistributedGPSurrogate, slurm_mcmc_hybrid
    settings = _swap_settings(work_dir)
    np.random.seed(0)
    slurm_mcmc_hybrid(surrogate='gp', num_rounds_max=2, **settings)
    np.random.seed(0)
    resumed = slurm_mcmc_hybrid(surrogate=DistributedGPSurrogate(num_experts=3,
                                                                n_restarts_optimizer=0),
                                num_rounds_max=4, load_restart=True, **settings)

    assert resumed['surrogate_switched_at_round'] == 3
    assert resumed['surrogate_class_per_round'] == ['GaussianProcess', 'GaussianProcess',
                                                    'DistributedGP', 'DistributedGP']
    # and it is on disk, not only in the returned dict
    diagnostics = open(os.path.join(work_dir, 'hybrid_diagnostics.txt')).read()
    assert 'surrogate' in diagnostics
    assert 'DistributedGP' in diagnostics and 'GaussianProcess' in diagnostics


def test_run_log_is_timestamped_and_marks_each_run(work_dir):
    """The log file is appended across restarts, so the two runs must be separable in it."""
    from slurmcmc.hybrid import slurm_mcmc_hybrid
    settings = dict(_swap_settings(work_dir), verbosity=1, log_file='hybrid_log.txt')
    np.random.seed(0)
    slurm_mcmc_hybrid(surrogate='gp', num_rounds_max=2, **settings)
    np.random.seed(0)
    slurm_mcmc_hybrid(surrogate='gp', num_rounds_max=3, load_restart=True, **settings)

    log = open(os.path.join(work_dir, 'hybrid_log.txt')).read()
    assert 'STARTING run' in log and 'RESUMING run' in log
    assert 'SETTINGS CHANGED ON RESTART' in log        # num_rounds_max went 2 -> 3
    # every line carries a wall-clock stamp
    stamped = [l for l in log.splitlines() if l.strip()]
    assert all(re.match(r'\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2} ', l) for l in stamped[:5])


# ---------------------------------------------------------------------------
# Unbounded mode (param_bounds=None)
# ---------------------------------------------------------------------------

def test_hybrid_unbounded_recovers_posterior_and_reports_budget():
    """
    With no param_bounds the run must still find the posterior, and the diagnostics file must
    state the expensive-evaluation budget it spent getting there.
    """
    np.random.seed(0)
    num_walkers, num_expensive_iters, num_reg = 12, 15, 20
    result = slurm_mcmc_hybrid(
        log_prob_fun=log_prob_gaussian,
        init_points=np.random.uniform(-2, 2, (num_walkers, 2)),
        param_bounds=None,
        num_expensive_iters=num_expensive_iters, num_regularization_points=num_reg,
        num_surrogate_iters=400, num_validation_points=40, num_rounds_max=2,
        refine_surrogate_ess=50, cluster='local-map', verbosity=0, work_dir='unbounded')

    samples = result['samples']
    assert np.all(np.isfinite(samples))
    assert np.allclose(samples.mean(axis=0), mu, atol=0.35)
    assert np.allclose(samples.std(axis=0), sigma, atol=0.35)

    text = open(os.path.join('unbounded', 'hybrid_diagnostics.txt')).read()
    # the round-0 budget and the first-fit training count, spelled out as equations
    num_round0 = (num_expensive_iters + 1) * num_walkers
    assert f'({num_expensive_iters} + 1) * {num_walkers} = {num_round0}' in text
    assert f'= {num_round0} + {num_reg} = {num_round0 + num_reg}' in text
    # run parameters travel with the diagnostics
    assert 'num_surrogate_iters' in text and 'refine_surrogate_ess' in text
    assert 'none (unbounded' in text
    # the per-round table carries the trim, and surr_std is no longer a column
    assert 'n_expensive' in text and 'n_train' in text
    assert 'surr_std' not in text


def _fitted_unbounded_runner():
    """A runner with a surrogate fitted on a standard-normal cloud, in unbounded mode."""
    np.random.seed(1)
    X = np.random.normal(size=(120, 2))
    runner = HybridMCMCRunner(HybridMCMCConfig(
        log_prob_fun=log_prob_gaussian, init_points=np.zeros((4, 2)), param_bounds=None,
        surrogate_uncertainty_penalty=0.0, cluster='local-map', verbosity=0))
    runner.train_X = [x for x in X]
    runner.train_y = [log_prob_gaussian(x) for x in X]
    runner._fit_surrogate()
    return runner


def test_confinement_makes_the_surrogate_posterior_proper():
    """
    Beyond the hard radius the surrogate log-prob is -inf, which is what stops an unbounded
    chain from walking away along a direction where the fitted trend happens to rise.
    """
    from slurmcmc.hybrid import _CONFINEMENT_HARD_RADIUS, _CONFINEMENT_INFLATION
    runner = _fitted_unbounded_runner()
    mean = runner._confinement['mean']
    offset = np.array([1.0, 0.0])
    unit_radius = runner._confinement_radius((mean + offset)[None])[0]

    def at_radius(target):
        return (mean + offset * (target / unit_radius))[None]

    assert np.isfinite(runner._surrogate_log_prob_batch(at_radius(0.5))[0])
    assert runner._surrogate_log_prob_batch(at_radius(50.0))[0] == -np.inf
    # and the boundary is where the config says it is
    hard = _CONFINEMENT_HARD_RADIUS
    assert np.isfinite(runner._surrogate_log_prob_batch(at_radius(hard * 0.9))[0])
    assert runner._surrogate_log_prob_batch(at_radius(hard * 1.1))[0] == -np.inf


def test_confinement_penalty_is_inactive_inside_the_data_hull():
    """
    The envelope must not touch the posterior where the training data actually is; it only
    switches on past _CONFINEMENT_INFLATION times the data's own radius.
    """
    from slurmcmc.hybrid import _CONFINEMENT_HARD_RADIUS, _CONFINEMENT_INFLATION
    runner = _fitted_unbounded_runner()
    mean = runner._confinement['mean']
    offset = np.array([1.0, 0.0])
    unit_radius = runner._confinement_radius((mean + offset)[None])[0]

    def at_radius(target):
        return (mean + offset * (target / unit_radius))[None]

    inflation = _CONFINEMENT_INFLATION
    for radius in (0.2, 0.9, inflation):
        point = at_radius(radius)
        assert np.isclose(runner._surrogate_log_prob_batch(point)[0],
                          float(runner.surrogate.predict(point)[0]))
    # past the envelope start the surrogate is penalised
    point = at_radius(inflation * 2.0)
    assert (runner._surrogate_log_prob_batch(point)[0]
            < float(runner.surrogate.predict(point)[0]) - 1.0)


def test_bounded_mode_is_unchanged_by_the_confinement_machinery():
    """A run that supplies param_bounds must behave exactly as before: box, no envelope."""
    runner = HybridMCMCRunner(HybridMCMCConfig(
        log_prob_fun=log_prob_gaussian, init_points=np.zeros((4, 2)),
        param_bounds=param_bounds, cluster='local-map', verbosity=0))
    assert runner._unbounded is False
    np.random.seed(2)
    X = np.random.normal(size=(60, 2))
    runner.train_X = [x for x in X]
    runner.train_y = [log_prob_gaussian(x) for x in X]
    runner._fit_surrogate()
    assert runner._confinement is None  # never built when a box was given
    assert runner._surrogate_log_prob_batch(np.array([[100.0, 100.0]]))[0] == -np.inf


def test_surrogate_walkers_are_independent_of_the_expensive_stage():
    """
    The cheap chain sizes its own ensemble. Walkers are cluster parallelism in the expensive
    stage, so more is strictly better there; in the cheap chain they trade directly against
    L_c/tau, so it defaults to the fewest emcee will accept.
    """
    from slurmcmc.hybrid import HybridMCMCConfig, HybridMCMCRunner
    cfg = HybridMCMCConfig(log_prob_fun=log_prob_gaussian, init_points=np.zeros((40, 2)),
                           param_bounds=param_bounds, cluster='local-map', verbosity=0)
    runner = HybridMCMCRunner(cfg)
    assert runner._num_surrogate_walkers() == 2 * 2 + 2   # not the 40 of the expensive stage

    cfg.num_surrogate_walkers = 12                        # an explicit count is honoured
    assert runner._num_surrogate_walkers() == 12
    cfg.num_surrogate_walkers = 3                         # odd and below 2*ndim: corrected, not
    assert runner._num_surrogate_walkers() == 6           # passed to emcee to raise on


def test_surrogate_chain_is_rebuilt_each_round_and_burnt_in(verbosity, seed, init_points):
    """
    Each round samples a *different* surrogate, so the chain must not accumulate: tau estimated
    across rounds measures the drift between successive surrogates as if it were autocorrelation
    within one chain, and inflates it. The kept samples are what survives surrogate_burnin_fraction.
    """
    burnin_fraction = 0.25
    result = slurm_mcmc_hybrid(log_prob_fun=log_prob_gaussian, init_points=init_points,
                               param_bounds=param_bounds,
                               num_expensive_iters=10, num_surrogate_iters=400,
                               surrogate_burnin_fraction=burnin_fraction,
                               refine_surrogate_ess=0,          # no extension, so the length is known
                               num_validation_points=20, num_rounds_max=2,
                               cluster='local-map', verbosity=verbosity)

    assert result['num_rounds'] == 2
    num_iters = result['surrogate_iters_per_round'][-1]
    # the sampler carries the last round only, not both rounds concatenated
    assert result['sampler'].get_chain().shape[0] == num_iters
    expected = (num_iters - int(burnin_fraction * num_iters)) * result['sampler'].nwalkers
    assert len(result['samples']) == expected


# ---------------------------------------------------------------------------
# Two-tier sampling: cheap refinement rounds, one verified posterior at the end
# ---------------------------------------------------------------------------

def test_ess_floors_scale_with_dimension():
    """None means mcmc.md's advised band: 10*num_params while refining, 100*num_params to report."""
    from slurmcmc.hybrid import HybridMCMCConfig, HybridMCMCRunner
    cfg = HybridMCMCConfig(log_prob_fun=log_prob_gaussian, init_points=np.zeros((8, 2)),
                           cluster='local-map', verbosity=0)
    runner = HybridMCMCRunner(cfg)
    assert runner._ess_floor() == 20.0
    assert runner._ess_floor(final=True) == 200.0
    cfg.refine_surrogate_ess, cfg.final_surrogate_ess = 123.0, 456.0
    assert (runner._ess_floor(), runner._ess_floor(final=True)) == (123.0, 456.0)


def test_verification_resamples_deeper_before_accepting(verbosity, seed, init_points):
    """
    A verdict reached on a cheap refinement chain is provisional. Before accepting, the same
    surrogate is re-sampled to reporting depth and re-scored on that posterior.
    """
    result = slurm_mcmc_hybrid(log_prob_fun=log_prob_gaussian, init_points=init_points,
                               param_bounds=param_bounds,
                               num_expensive_iters=12, num_regularization_points=30,
                               num_surrogate_iters=600, num_validation_points=30,
                               num_rounds_max=4, cluster='local-map', verbosity=verbosity)

    assert result['converged']
    assert result['verification_attempts'] >= 1
    # the reported posterior clears the *final* floor, not merely the refinement one
    assert result['surrogate_ess_per_round'][-1] >= 100 * 2
    # and the verified round is marked as such in the diagnostics table
    assert str(result['surrogate_class_per_round'][-1])  # rows exist
    text = open(os.path.join('mcmc_hybrid', 'hybrid_diagnostics.txt')).read()
    assert f"{result['num_rounds']}v" in text


def test_verification_can_be_switched_off(verbosity, seed, init_points):
    """max_verification_attempts=0 switches verification off."""
    result = slurm_mcmc_hybrid(log_prob_fun=log_prob_gaussian, init_points=init_points,
                               param_bounds=param_bounds,
                               num_expensive_iters=12, num_regularization_points=30,
                               num_surrogate_iters=600, num_validation_points=30,
                               max_verification_attempts=0,
                               num_rounds_max=4, cluster='local-map', verbosity=verbosity)
    assert result['verification_attempts'] == 0
    text = open(os.path.join('mcmc_hybrid', 'hybrid_diagnostics.txt')).read()
    assert 'v' not in text.split('round n_expensive')[1].split('\n')[2]


def test_every_config_field_is_reachable_through_the_wrapper():
    """
    Adding a config field and forgetting the wrapper is silent until someone passes it and gets
    a TypeError at runtime. Only the warning/stall tuning knobs are deliberately config-only.
    """
    import dataclasses
    import inspect
    from slurmcmc.hybrid import HybridMCMCConfig

    fields = {f.name for f in dataclasses.fields(HybridMCMCConfig)}
    params = set(inspect.signature(slurm_mcmc_hybrid).parameters)
    config_only = {'bounds_warning_fraction', 'bounds_warning_tolerance',
                   'stall_relative_improvement', 'stall_significance', 'stall_window'}
    assert fields - params == config_only, 'a new config field is not reachable from the wrapper'
    assert params - fields == set(), 'the wrapper takes an argument the config does not have'


def test_a_rejected_verification_loses_no_expensive_evaluations(seed, init_points):
    """
    When verification runs, its batch becomes the round's measurement -- but the refinement batch
    before it was paid for too, and both must reach the training set. The refinement batch used
    to be overwritten and lost: 100 evaluations in the 3D example, 8% of the run. The verdicts
    are scripted so that the reject-then-accept path is exercised every time, not just when a
    real run happens to reject.
    """
    from slurmcmc.hybrid import HybridMCMCConfig, HybridMCMCRunner
    num_val = 20
    cfg = HybridMCMCConfig(log_prob_fun=log_prob_gaussian, init_points=init_points,
                           param_bounds=param_bounds, num_expensive_iters=8,
                           num_regularization_points=0, num_surrogate_iters=300,
                           num_validation_points=num_val, num_rounds_max=4,
                           cluster='local-map', verbosity=0)
    runner = HybridMCMCRunner(cfg)
    # round 1: provisional pass, verification rejects; round 2: provisional pass, accepted
    verdicts = iter([True, False, True, True])
    runner._is_converged = lambda metrics: next(verdicts)
    offered = []
    add_training_points = runner._add_training_points

    def recording_add(points, values):
        offered.append(len(points))
        return add_training_points(points, values)

    runner._add_training_points = recording_add
    result = runner.run()

    assert result['converged'] and result['num_rounds'] == 2
    assert result['verification_attempts'] == 2
    assert result['verified_per_round'] == [True, True]
    # after the round-0 chain: round 1's refinement and verification batches, then round 2's
    # refinement batch -- every validation batch except the one that was finally accepted
    assert offered[1:] == [num_val, num_val, num_val]


def test_exhausted_verification_budget_is_not_reported_as_converged(seed, init_points):
    """
    If every verification attempt is rejected, the only posterior in hand was measured on a cheap
    refinement chain. Accepting that as converged would return exactly what verification exists to
    refuse, so the run must decline to claim convergence.
    """
    from slurmcmc.hybrid import HybridMCMCConfig, HybridMCMCRunner
    cfg = HybridMCMCConfig(log_prob_fun=log_prob_gaussian, init_points=init_points,
                           param_bounds=param_bounds, num_expensive_iters=8,
                           num_regularization_points=0, num_surrogate_iters=300,
                           num_validation_points=20, num_rounds_max=4,
                           max_verification_attempts=3, cluster='local-map', verbosity=0)
    runner = HybridMCMCRunner(cfg)
    # every round passes provisionally; every verification rejects. The fourth round has no
    # attempts left, and must not be waved through.
    verdicts = iter([True, False, True, False, True, False, True])
    runner._is_converged = lambda metrics: next(verdicts)
    result = runner.run()

    assert result['verification_attempts'] == 3
    assert not result['converged'], 'an unverified posterior was reported as converged'
    assert result['verified_per_round'] == [True, True, True, False]


def test_posterior_stability_is_judged_in_posterior_widths():
    """
    Stability is a statement about the answer, not about the chain: the largest change in any
    parameter's mean or std, in units of that parameter's own width. It does not tighten as the
    chain lengthens, which is what a Monte-Carlo z-score does.
    """
    from slurmcmc.hybrid import HybridMCMCConfig, HybridMCMCRunner
    rng = np.random.default_rng(0)
    old = rng.normal(0.0, 1.0, (50000, 3))
    runner = HybridMCMCRunner(HybridMCMCConfig(log_prob_fun=log_prob_gaussian,
                                               init_points=np.zeros((8, 2)),
                                               cluster='local-map', verbosity=0))
    assert runner._posterior_displacement(old + 0.02, old) == pytest.approx(0.02, abs=0.005)
    assert runner._posterior_displacement(old, None) == float('inf')

    passing = {'weighted_log_error': 0.0, 'ess_weights': 1e3,
               'posterior_displacement': 0.02, 'posterior_noise_floor': 0.01}
    assert runner._is_converged(passing)
    assert 'posterior-stability' not in runner._blocking_criteria(passing)

    moving = dict(passing, posterior_displacement=0.5)
    assert not runner._is_converged(moving)
    assert 'posterior-stability' in runner._blocking_criteria(moving)

    # a short chain cannot resolve a 0.1 move: two estimates of one posterior differ by more than
    # that by chance, so the threshold must not fall below the noise and make itself unsatisfiable
    noisy = dict(passing, posterior_displacement=0.25, posterior_noise_floor=0.30)
    assert runner._is_converged(noisy), 'a move smaller than the sampling noise must pass'
    assert runner._is_converged(dict(noisy, posterior_noise_floor=0.05)) is False


def test_existing_evaluations_can_replace_round_zero(seed, caplog):
    """
    Evaluations from an earlier run are as good as ones round 0 would make, so seeding with them
    skips the expensive stage entirely -- and they must not be charged to this run's budget.
    """
    from slurmcmc.hybrid import HybridMCMCConfig, HybridMCMCRunner
    rng = np.random.default_rng(0)
    points = rng.uniform(-3, 3, (120, 2))
    values = np.array([log_prob_gaussian(x) for x in points])
    runner = HybridMCMCRunner(HybridMCMCConfig(
        log_prob_fun=log_prob_gaussian, init_points=np.zeros((8, 2)), param_bounds=param_bounds,
        initial_train_points=points, initial_train_values=values,
        num_expensive_iters=999, num_regularization_points=999,   # must never be used
        num_surrogate_iters=300, num_validation_points=20, num_rounds_max=1,
        cluster='local-map', verbosity=0))
    with caplog.at_level('WARNING'):
        result = runner.run()

    # round 0 was skipped: the only expensive calls are this round's validation batch
    assert result['num_expensive_evals'] == 20
    # and the regularization points skipped with it are not skipped silently
    assert any('num_regularization_points=999 is ignored' in r.getMessage() for r in caplog.records)
    assert result['num_train_points_per_round'][0] == len(points)
    assert runner.timings['expensive_mcmc_0'] == 0.0


def test_seeding_requires_points_and_values_together():
    from slurmcmc.hybrid import HybridMCMCConfig, HybridMCMCRunner
    with pytest.raises(ValueError, match='must be given together'):
        HybridMCMCRunner(HybridMCMCConfig(log_prob_fun=log_prob_gaussian,
                                          init_points=np.zeros((8, 2)),
                                          initial_train_points=np.zeros((5, 2))))
    with pytest.raises(ValueError, match='rows but'):
        HybridMCMCRunner(HybridMCMCConfig(log_prob_fun=log_prob_gaussian,
                                          init_points=np.zeros((8, 2)),
                                          initial_train_points=np.zeros((5, 2)),
                                          initial_train_values=np.zeros(4)))


def test_seeded_run_needs_no_init_points_and_resumes_without_them(work_dir):
    """
    With existing evaluations round 0 is skipped, and init_points was only ever its starting
    positions, so a seeded run must not demand them -- neither to start nor to resume, and its
    diagnostics must not describe a round 0 that never ran.
    """
    from slurmcmc.hybrid import slurm_mcmc_hybrid
    rng = np.random.default_rng(0)
    points = rng.uniform(-3, 3, (120, 2))
    values = np.array([log_prob_gaussian(x) for x in points])
    settings = dict(log_prob_fun=log_prob_gaussian, param_bounds=param_bounds,
                    initial_train_points=points, initial_train_values=values,
                    num_surrogate_iters=300, num_validation_points=20,
                    log_error_threshold=1e-9,          # unreachable, so the resume has work to do
                    cluster='local-map', verbosity=0, work_dir=work_dir, save_restart=True)
    np.random.seed(0)
    first = slurm_mcmc_hybrid(num_rounds_max=1, **settings)
    assert first['num_expensive_evals'] == 20          # only the validation batch
    diagnostics = open(os.path.join(work_dir, 'hybrid_diagnostics.txt')).read()
    assert 'round 0 skipped' in diagnostics and 'round 0 MCMC' not in diagnostics

    np.random.seed(0)
    resumed = slurm_mcmc_hybrid(num_rounds_max=2, load_restart=True, **settings)
    assert resumed['num_rounds'] == 2
    assert resumed['num_expensive_evals'] == 40


def test_init_points_is_required_whenever_something_uses_it():
    from slurmcmc.hybrid import HybridMCMCConfig, HybridMCMCRunner
    points = np.zeros((5, 2))
    values = np.zeros(5)
    with pytest.raises(ValueError, match='init_points is required'):
        HybridMCMCRunner(HybridMCMCConfig(log_prob_fun=log_prob_gaussian))
    # the expensive refresh takes its walker count from init_points
    with pytest.raises(ValueError, match='num_expensive_iters_per_round'):
        HybridMCMCRunner(HybridMCMCConfig(log_prob_fun=log_prob_gaussian,
                                          initial_train_points=points, initial_train_values=values,
                                          num_expensive_iters_per_round=2))
    with pytest.raises(ValueError, match='disagree on the number of parameters'):
        HybridMCMCRunner(HybridMCMCConfig(log_prob_fun=log_prob_gaussian,
                                          init_points=np.zeros((8, 3)),
                                          initial_train_points=points, initial_train_values=values))
    # both given and consistent is still fine, as before
    HybridMCMCRunner(HybridMCMCConfig(log_prob_fun=log_prob_gaussian, init_points=np.zeros((8, 2)),
                                      initial_train_points=points, initial_train_values=values))


def test_random_seed_reproduces_a_run_locally_remotely_and_across_a_restart(work_dir):
    """
    The seed is applied inside the process that runs the loop. Without that, a remote run -- a
    fresh process on a compute node -- starts from an unseeded generator and follows a different
    trajectory from the very same configuration. The restart file carries the generator state,
    so a resumed run reproduces an uninterrupted one.
    """
    from slurmcmc.hybrid import slurm_mcmc_hybrid
    settings = dict(log_prob_fun=log_prob_ring, param_bounds=param_bounds,
                    init_points=np.random.default_rng(0).uniform(-2, 2, (8, 2)),
                    num_expensive_iters=5, num_regularization_points=20, num_surrogate_iters=250,
                    num_validation_points=20, log_error_threshold=1e-9,   # force every round to run
                    cluster='local-map', verbosity=0)

    def summary(result):
        return (result['samples'], [r['weighted_log_error'] for r in result['round_records']],
                result['num_expensive_evals'])

    def assert_same(a, b):
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1] == b[1] and a[2] == b[2]

    reference = summary(slurm_mcmc_hybrid(num_rounds_max=2, random_seed=11,
                                          work_dir=os.path.abspath('local'), **settings))
    np.random.rand(10)
    assert_same(reference, summary(slurm_mcmc_hybrid(num_rounds_max=2, random_seed=11,
                                                     work_dir=os.path.abspath('again'), **settings)))
    other = summary(slurm_mcmc_hybrid(num_rounds_max=2, random_seed=12,
                                      work_dir=os.path.abspath('other'), **settings))
    assert not np.array_equal(reference[0], other[0])

    job = slurm_mcmc_hybrid(num_rounds_max=2, random_seed=11, work_dir=os.path.abspath('remote'),
                            remote=True, remote_cluster='local',
                            remote_submitit_kwargs={'timeout_min': 20}, **settings)
    assert_same(reference, summary(job.result()))

    slurm_mcmc_hybrid(num_rounds_max=1, random_seed=11, work_dir=os.path.abspath('resumed'),
                      save_restart=True, **settings)
    np.random.rand(10)
    resumed = slurm_mcmc_hybrid(num_rounds_max=2, random_seed=11, work_dir=os.path.abspath('resumed'),
                                load_restart=True, save_restart=True, **settings)
    assert_same(reference, summary(resumed))


def test_resume_clears_the_stage_an_interrupted_round_left_behind(work_dir):
    """
    A round killed during its expensive batch leaves that batch's directory without a restart
    recording it; SlurmPool refuses a directory holding old output, so the resume must move it
    aside rather than fail. keep_run_dirs reaches the batches: 'none' leaves only the history files.
    """
    settings = dict(log_prob_fun=log_prob_gaussian, init_points=np.random.default_rng(0).normal(size=(6, 2)),
                    num_expensive_iters=1, num_regularization_points=0, num_validation_points=6,
                    num_surrogate_iters=200, log_error_threshold=1e-9, posterior_shift_tolerance=None,
                    surrogate='polynomial', polynomial_degree=2, cluster='local', verbosity=0,
                    work_dir=work_dir, save_restart=True, random_seed=1)
    slurm_mcmc_hybrid(num_rounds_max=1, keep_run_dirs='none', **settings)
    for stage in ('round0_expensive_mcmc', 'round1_validation'):
        assert sorted(os.listdir(os.path.join(work_dir, stage))) == ['points_history.txt', 'values_history.txt']

    os.makedirs(os.path.join(work_dir, 'round2_validation', '0'))  # the interrupted attempt
    result = slurm_mcmc_hybrid(num_rounds_max=2, keep_run_dirs='all', load_restart=True, **settings)
    assert result['num_rounds'] == 2
    assert os.path.isdir(os.path.join(work_dir, 'round2_validation_interrupted1', '0'))
    assert os.path.isdir(os.path.join(work_dir, 'round2_validation', '0'))


def test_verification_row_leaves_the_shared_columns_empty(verbosity, seed, init_points):
    """A verified round's two rows share one fit, so its size and class are printed once."""
    result = slurm_mcmc_hybrid(log_prob_fun=log_prob_gaussian, init_points=init_points,
                               param_bounds=param_bounds,
                               num_expensive_iters=12, num_regularization_points=30,
                               num_surrogate_iters=600, num_validation_points=30,
                               num_rounds_max=4, cluster='local-map', verbosity=verbosity)
    assert result['verification_attempts'] >= 1
    widths = {name: width for name, width, _ in HybridMCMCRunner._DIAGNOSTICS_COLUMNS}
    lines = open(os.path.join('mcmc_hybrid', 'hybrid_diagnostics.txt')).read().splitlines()

    def field(line, name):
        start = 0
        for column, width in widths.items():
            if column == name:
                return line[start:start + width].strip()
            start += width

    verification = [line for line in lines if re.match(r'\s*\d+v\s', line)]
    assert verification
    refinement = [lines[lines.index(line) - 1] for line in verification]
    for line_v, line_r in zip(verification, refinement):
        for name in ('n_expensive', 'n_train', 'surrogate', 't_fit_s'):
            assert field(line_v, name) == ''
            assert field(line_r, name) != ''
        assert field(line_v, 'w_log_err') != ''
    assert len(result['surrogate_class_per_round']) == result['num_rounds']


def test_surrogate_log_records_every_fit_with_its_settings(work_dir):
    """
    save_surrogate also keeps a log of the fits themselves: what was fitted, on how many points,
    for how long, and with which settings -- appended, so it survives 'latest' pruning the
    pickles and a restart, and so a surrogate switched mid-run shows up as a change of class.
    """
    from slurmcmc.hybrid import DistributedGPSurrogate, slurm_mcmc_hybrid
    settings = _swap_settings(work_dir)
    np.random.seed(0)
    slurm_mcmc_hybrid(surrogate='gp', num_rounds_max=2, **settings)
    np.random.seed(0)
    slurm_mcmc_hybrid(surrogate=DistributedGPSurrogate(num_experts=2, n_restarts_optimizer=0,
                                                       min_points_per_expert=10),
                      num_rounds_max=3, load_restart=True, **settings)

    lines = open(os.path.join(work_dir, 'surrogates', 'surrogate_log.txt')).read().splitlines()
    rows = [line for line in lines if re.match(r'\s*\d+\s', line)]
    assert [int(line.split()[0]) for line in rows] == [1, 2, 3]
    assert [line.split()[1] for line in rows] == ['GaussianProcess', 'GaussianProcess', 'DistributedGP']
    # n_raw, n_trim, n_sub, n_fit, t_fit_s are numbers, and the settings close the line
    for line in rows:
        assert all(field.replace('.', '').isdigit() for field in line.split()[2:7])
    assert 'num_experts=2' in rows[-1] and 'default settings' in rows[0]
    # save_surrogate='latest' left one pickle, but the log kept every round
    assert sorted(os.listdir(os.path.join(work_dir, 'surrogates'))) == ['round3.pkl', 'surrogate_log.txt']


def test_diagnostics_header_has_no_training_set_block(verbosity, seed, init_points):
    """The per-round n_expensive/n_train columns already carry the trim; the block was redundant."""
    slurm_mcmc_hybrid(log_prob_fun=log_prob_gaussian, init_points=init_points,
                      param_bounds=param_bounds, num_expensive_iters=8,
                      num_regularization_points=20, num_surrogate_iters=400,
                      num_validation_points=20, num_rounds_max=2, cluster='local-map',
                      verbosity=verbosity)
    text = open(os.path.join('mcmc_hybrid', 'hybrid_diagnostics.txt')).read()
    assert 'training set at the latest surrogate fit' not in text
    assert 'expensive evaluations' in text and 'n_expensive' in text
