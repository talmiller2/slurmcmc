"""
Tests for the surrogate models themselves (slurmcmc/hybrid.py), independent of the pipeline.

Pipeline-level behaviour -- convergence criteria, restart, diagnostics -- lives in
test_hybrid.py. What is tested here is the surrogates in isolation: that they fit and predict,
that their predictive uncertainty behaves, and that the distributed ensemble's expert fitting
works serially, across cores, and across a cluster.
"""

import os

import numpy as np
import pytest

from slurmcmc.hybrid import GaussianProcessSurrogate, PolynomialSurrogate
from slurmcmc.slurm_utils import is_slurm_cluster
from tests.submitit_defaults import submitit_kwargs

# analytic 2D Gaussian: exactly quadratic, so the polynomial mean function explains it
# outright and the GP stage is skipped -- fine behaviour, useless for exercising a GP
mu = np.array([1.0, -1.0])
sigma = np.array([1.0, 0.5])
param_bounds = [[-5, 5], [-5, 5]]


def log_prob_gaussian(x):
    return float(-0.5 * np.sum(((np.asarray(x) - mu) / sigma) ** 2))


def log_prob_ring(x):
    """A shell: not polynomial at any degree, so the GP stage actually runs."""
    radius = np.sqrt(np.sum(np.asarray(x, dtype=float) ** 2))
    return float(-0.5 * ((radius - 2.0) / 0.6) ** 2)


def log_prob_rosenbrock(x):
    """
    The 3D Rosenbrock posterior the hybrid examples use. Its curved, sharply varying ridge
    drives the GP amplitude to its upper bound, which is what makes it the right target for
    testing an aggregation rule that has to compare variances against a prior.
    """
    x = np.asarray(x, dtype=float)
    return float(-sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
                      for i in range(len(x) - 1)) / 20.0)


def _rbcm_training_set(n=400, d=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, (n, d))
    return X, np.array([log_prob_ring(x) for x in X])


def _rosenbrock_training_set(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-4, 4, (n, 3))
    return X, np.array([log_prob_rosenbrock(x) for x in X])


def test_gp_surrogate_fit_predict(seed):
    X = np.random.uniform(-3, 3, (150, 2))
    y = np.array([log_prob_gaussian(x) for x in X])
    surrogate = GaussianProcessSurrogate()
    surrogate.fit(X, y)
    X_test = np.random.uniform(-2, 2, (30, 2))
    y_test = np.array([log_prob_gaussian(x) for x in X_test])
    y_pred = surrogate.predict(X_test)
    assert y_pred.shape == (30,)  # vectorized predict
    np.testing.assert_allclose(y_pred, y_test, atol=0.1)


def test_polynomial_surrogate_fit_predict(seed):
    X = np.random.uniform(-3, 3, (100, 2))
    y = np.array([log_prob_gaussian(x) for x in X])
    surrogate = PolynomialSurrogate(degree=2)  # the Gaussian log-prob is exactly quadratic
    surrogate.fit(X, y)
    X_test = np.random.uniform(-2, 2, (30, 2))
    y_test = np.array([log_prob_gaussian(x) for x in X_test])
    np.testing.assert_allclose(surrogate.predict(X_test), y_test, atol=1e-3)


def test_gp_surrogate_does_not_extrapolate_above_training_range(seed):
    """The polynomial mean function extrapolates hard outside the training cloud; predictions
    must stay within the observed log-prob range (plus a margin) so the surrogate cannot
    invent a probability peak higher than anything ever evaluated."""
    # training data confined to a small region of a much larger box
    X = np.random.uniform(-1, 1, (120, 3))
    y = np.array([log_prob_gaussian(x[:2]) for x in X])
    surrogate = GaussianProcessSurrogate()
    surrogate.fit(X, y)
    # predict far outside the training cloud
    X_far = np.random.uniform(-5, 5, (200, 3))
    pred = surrogate.predict(X_far)
    pad = max(0.1 * (y.max() - y.min()), 1.0)
    assert np.all(pred <= y.max() + pad + 1e-9)
    assert np.all(pred >= y.min() - pad - 1e-9)


def test_gp_surrogate_predict_std(seed):
    """predict_std reports the emulator's own uncertainty, larger away from training data."""
    X = np.random.uniform(-1, 1, (100, 2))
    y = np.array([log_prob_gaussian(x) for x in X])
    surrogate = GaussianProcessSurrogate()
    surrogate.fit(X, y)
    std_near = surrogate.predict_std(np.random.uniform(-1, 1, (50, 2)))
    assert std_near.shape == (50,)
    assert np.all(std_near >= 0)


def test_rbcm_expert_count_follows_max_points_per_expert():
    """
    A fixed expert count means each expert grows with the training set, which brings back the
    O(n^3) wall the ensemble exists to avoid. max_points_per_expert grows the *number* instead.
    """
    from slurmcmc.hybrid import DistributedGPSurrogate
    surrogate = DistributedGPSurrogate(max_points_per_expert=100, partition='random',
                                       min_points_per_expert=10)
    for n, expected in [(400, 4), (1000, 10)]:
        X, _ = _rbcm_training_set(n=n)
        assert len(surrogate._assign(X)) == expected

    # without it, the count is capped and each expert grows without bound
    fixed = DistributedGPSurrogate(num_experts=4, partition='random', min_points_per_expert=10)
    assert len(fixed._assign(_rbcm_training_set(n=400)[0])) == 4
    assert len(fixed._assign(_rbcm_training_set(n=4000)[0])) == 4


def test_rbcm_will_not_split_below_min_points_per_expert():
    """
    An expert fitted on a handful of points cannot identify its own ARD kernel: it learns a
    short length scale, is confident only on top of its own points, and contributes nothing
    anywhere else. Measured on a 3D Rosenbrock posterior, 51-point experts were an order of
    magnitude worse than a dense GP on the same data while 200-point experts matched it, so
    the ensemble refuses to split that finely and quietly uses fewer experts instead.
    """
    from slurmcmc.hybrid import DistributedGPSurrogate
    surrogate = DistributedGPSurrogate(num_experts=5, partition='random',
                                       min_points_per_expert=200)
    assert len(surrogate._assign(_rbcm_training_set(n=150)[0])) == 1
    assert len(surrogate._assign(_rbcm_training_set(n=600)[0])) == 3
    assert len(surrogate._assign(_rbcm_training_set(n=5000)[0])) == 5


def test_rbcm_partitioning_keeps_every_training_point():
    """
    k-means can leave a cluster too small to fit. Those points must be handed to the nearest
    surviving expert, not dropped: each one was paid for with an expensive evaluation.
    """
    from slurmcmc.hybrid import DistributedGPSurrogate
    rng = np.random.default_rng(0)
    # one tight blob plus three far-flung outliers, so k-means isolates sub-5-point clusters
    X = np.vstack([rng.normal(0, 0.05, (300, 2)),
                   np.array([[8.0, 8.0], [-8.0, 8.0], [8.0, -8.0]])])
    surrogate = DistributedGPSurrogate(num_experts=4, min_points_per_expert=10)
    groups = surrogate._assign(X)
    assigned = np.concatenate(groups)
    assert sorted(assigned.tolist()) == list(range(len(X))), 'a training point was dropped'
    assert len(assigned) == len(set(assigned.tolist())), 'a training point was double-counted'
    assert all(len(g) >= 5 for g in groups)


def test_rbcm_reduces_to_a_single_gp_when_there_is_one_expert():
    """With M=1 the aggregation must be a no-op, or the weighting is wrong."""
    from slurmcmc.hybrid import DistributedGPSurrogate, GaussianProcessSurrogate
    X, y = _rbcm_training_set(n=60)
    single = DistributedGPSurrogate(num_experts=1, n_restarts_optimizer=0)
    single.fit(X, y)
    assert len(single._experts) == 1
    dense = GaussianProcessSurrogate(n_restarts_optimizer=0)
    dense.fit(X, y)
    probe = np.array([[0.0, 0.0], [1.0, -1.0], [2.5, 2.0]])
    assert single.predict(probe) == pytest.approx(dense.predict(probe), abs=0.05)


class _FixedExpert:
    """A stand-in expert with a chosen posterior and a chosen prior variance, so the
    aggregation can be tested on its own without fitting anything."""

    class _Kernel:
        def __init__(self, prior_var):
            self.prior_var = prior_var

        def diag(self, X):
            return np.full(len(X), self.prior_var)

    def __init__(self, mean, std, prior_var):
        self.mean, self.std = mean, std
        self.kernel_ = self._Kernel(prior_var)

    def predict(self, X, return_std=False):
        mean = np.full(len(X), self.mean)
        return (mean, np.full(len(X), self.std)) if return_std else mean


def test_rbcm_weighs_each_expert_against_its_own_prior():
    """
    Regression test. The entropy weight beta_k = 0.5 (log sigma_k,prior^2 - log sigma_k^2) has
    to use the prior the expert actually fitted. Scoring it against a nominal 1 -- on the
    grounds that the residual was standardized -- clamps beta to zero for any expert whose
    learned amplitude exceeds 1, and the ensemble then returns a zero residual (its bare
    polynomial trend) with a fixed variance, while still looking well-behaved from outside.
    An amplitude far from 1 is not exotic: the amplitude/length-scale degeneracy routinely
    drives it to its bound.
    """
    from slurmcmc.hybrid import DistributedGPSurrogate
    surrogate = DistributedGPSurrogate(num_experts=1)
    # confident about a nonzero value, but with a prior variance far above 1
    surrogate._experts = [_FixedExpert(mean=2.0, std=5.0, prior_var=1000.0)]
    surrogate._expert_prior_vars = None
    mean, var = surrogate._combine(np.zeros((4, 2)))
    assert mean == pytest.approx(2.0), 'the expert was muted and the ensemble returned its trend'
    assert var == pytest.approx(25.0), 'the aggregate must reproduce a lone expert exactly'


def test_rbcm_aggregation_reproduces_a_lone_expert_exactly():
    """With M=1 the aggregation is a no-op by construction, whatever the expert believes."""
    from slurmcmc.hybrid import DistributedGPSurrogate
    for mean, std, prior in [(2.0, 5.0, 1000.0), (-3.0, 0.01, 1.0), (0.5, 0.9, 1.2)]:
        surrogate = DistributedGPSurrogate(num_experts=1)
        surrogate._experts = [_FixedExpert(mean, std, prior)]
        surrogate._expert_prior_vars = None
        got_mean, got_var = surrogate._combine(np.zeros((3, 2)))
        assert got_mean == pytest.approx(mean, rel=1e-12)
        assert got_var == pytest.approx(std ** 2, rel=1e-12)


def test_rbcm_lets_the_informed_expert_carry_the_prediction():
    """An expert sitting on data must outvote experts that are back at their own prior."""
    from slurmcmc.hybrid import DistributedGPSurrogate
    surrogate = DistributedGPSurrogate(num_experts=3)
    surrogate._experts = [_FixedExpert(mean=7.0, std=0.01, prior_var=4.0),   # knows this point
                          _FixedExpert(mean=0.0, std=2.0, prior_var=4.0),    # at its prior
                          _FixedExpert(mean=0.0, std=2.0, prior_var=4.0)]
    surrogate._expert_prior_vars = None
    mean, var = surrogate._combine(np.zeros((2, 2)))
    assert mean == pytest.approx(7.0, rel=1e-3)
    assert np.max(var) < 0.01, 'the confident expert must dominate the combined variance'


def test_rbcm_reverts_to_the_prior_when_no_expert_is_informed():
    """
    Where every expert is back at its own prior the weights are uninformative. The ensemble
    must then report the prior variance -- a large number that keeps the uncertainty penalty
    alive -- rather than a small one that would wave the surrogate MCMC through.
    """
    from slurmcmc.hybrid import DistributedGPSurrogate
    surrogate = DistributedGPSurrogate(num_experts=2)
    surrogate._experts = [_FixedExpert(mean=0.0, std=np.sqrt(9.0), prior_var=9.0),
                          _FixedExpert(mean=0.0, std=np.sqrt(9.0), prior_var=9.0)]
    surrogate._expert_prior_vars = None
    mean, var = surrogate._combine(np.zeros((2, 2)))
    assert mean == pytest.approx(0.0)
    assert var == pytest.approx(9.0), 'must revert to the prior variance, not to 1'


def test_rbcm_does_not_collapse_onto_its_polynomial_trend():
    """
    The end-to-end symptom of the bug above, on a target no polynomial can represent: the
    fitted surrogate predicted its own trend and reported a predictive std pinned at the
    residual scale, which is both wrong and reassuring-looking.
    """
    from slurmcmc.hybrid import DistributedGPSurrogate
    X, y = _rosenbrock_training_set(n=400)
    surrogate = DistributedGPSurrogate(num_experts=1, n_restarts_optimizer=2)
    surrogate.fit(X, y)
    assert surrogate._expert_prior_vars[0] > 1.0, (
        'this target is chosen because it drives the amplitude above 1, where an aggregation '
        'rule that assumes a unit prior mutes the expert; if it no longer does, the test has '
        'stopped guarding the regression it was written for')
    # on the data the GP must carry real signal beyond its polynomial mean function
    probe = X[:80]
    trend = np.ravel(surrogate._trend_model.predict(surrogate._scaler.transform(probe)))
    gp_error = np.sqrt(np.mean((surrogate.predict(probe) - y[:80]) ** 2))
    trend_error = np.sqrt(np.mean((trend - y[:80]) ** 2))
    assert gp_error < 0.2 * trend_error, 'the GP carries no signal beyond the trend'

    # Away from the data is where a muted expert shows itself. With every beta clamped to
    # zero the combination returns a zero residual with unit variance, so predict_std comes
    # back as exactly _residual_scale: a wrong answer wearing an unremarkable error bar, and
    # small enough that the surrogate MCMC's uncertainty penalty waves the walkers through.
    far = np.full((5, 3), 40.0)
    far_std = np.max(surrogate.predict_std(far))
    assert far_std > 2 * surrogate._residual_scale, (
        f'far-field std {far_std:.3f} vs residual scale {surrogate._residual_scale:.3f}: the '
        f'ensemble is not reverting to the prior it actually fitted')


def test_rbcm_matches_a_dense_gp_when_experts_are_large_enough():
    """
    The point of the ensemble is to buy back fitting cost without giving up accuracy. With
    enough points per expert its predictions must track a dense GP on the same data.
    """
    from slurmcmc.hybrid import DistributedGPSurrogate, GaussianProcessSurrogate
    X, y = _rbcm_training_set(n=900)
    probe, truth = _rbcm_training_set(n=120, seed=7)
    dense = GaussianProcessSurrogate(n_restarts_optimizer=2)
    dense.fit(X, y)
    ensemble = DistributedGPSurrogate(num_experts=4, n_restarts_optimizer=2,
                                      min_points_per_expert=200)
    ensemble.fit(X, y)
    assert len(ensemble._experts) == 4
    dense_error = np.sqrt(np.mean((dense.predict(probe) - truth) ** 2))
    ensemble_error = np.sqrt(np.mean((ensemble.predict(probe) - truth) ** 2))
    assert ensemble_error < max(2 * dense_error, dense_error + 0.05), (
        f'ensemble RMS {ensemble_error:.4f} vs dense GP {dense_error:.4f}')


def test_rbcm_variance_reverts_towards_the_prior_far_from_data():
    """
    The prior-correction term of the rBCM is what stops a product of experts becoming
    overconfident away from the data. Without it the variances multiply and shrink.
    """
    from slurmcmc.hybrid import DistributedGPSurrogate
    X, y = _rbcm_training_set(n=200)
    surrogate = DistributedGPSurrogate(num_experts=4, n_restarts_optimizer=0,
                                       min_points_per_expert=10)
    surrogate.fit(X, y)
    near = surrogate.predict_std(np.array([[0.0, 0.0]]))
    far = surrogate.predict_std(np.array([[4.9, 4.9]]))
    assert far > near, 'predictive std must grow away from the training data'


@pytest.mark.parametrize('backend', ['joblib', 'slurm'])
def test_rbcm_parallel_backends_agree_with_serial(backend, work_dir):
    """
    The expert fits are independent, so distributing them must not change the answer. Small
    differences are expected from threaded BLAS, not from the aggregation.
    """
    from slurmcmc.hybrid import DistributedGPSurrogate
    X, y = _rbcm_training_set(n=300)
    probe = np.array([[0.0, 0.0], [1.5, -1.0], [-2.0, 2.0]])

    serial = DistributedGPSurrogate(num_experts=3, n_restarts_optimizer=0, parallel='none',
                                     min_points_per_expert=10)
    serial.fit(X, y)

    kwargs = dict(cluster='local', work_dir=work_dir) if backend == 'slurm' else {}
    parallel = DistributedGPSurrogate(num_experts=3, n_restarts_optimizer=0,
                                      min_points_per_expert=10,
                                      parallel=backend, **kwargs)
    parallel.fit(X, y)

    assert len(parallel._experts) == len(serial._experts)
    assert parallel.predict(probe) == pytest.approx(serial.predict(probe), abs=1e-2)


def test_slurm_backend_removes_its_scratch_files(work_dir):
    """
    submitit uses the filesystem as its transport, and a fitted GP carries its Cholesky
    factor (~8 MB at 1000 points). Left alone that is gigabytes over a long run, on what is
    usually the slowest storage available.
    """
    from slurmcmc.hybrid import DistributedGPSurrogate
    X, y = _rbcm_training_set(n=300)
    surrogate = DistributedGPSurrogate(num_experts=3, n_restarts_optimizer=0,
                                       min_points_per_expert=10,
                                       parallel='slurm', cluster='local', work_dir=work_dir)
    surrogate.fit(X, y)
    surrogate.fit(X, y)  # a second round must not accumulate either
    leftovers = [os.path.join(root, f)
                 for root, _, files in os.walk(work_dir) for f in files]
    assert leftovers == [], f'scratch files left behind: {leftovers[:3]}'
    assert len(surrogate._experts) == 3  # and the experts survived in memory


def test_parallel_backend_falls_back_to_serial_on_failure(work_dir):
    """A scheduler problem must degrade performance, not kill a multi-hour run."""
    from slurmcmc.hybrid import DistributedGPSurrogate
    X, y = _rbcm_training_set(n=200)
    surrogate = DistributedGPSurrogate(num_experts=2, n_restarts_optimizer=0,
                                       min_points_per_expert=10,
                                       parallel='slurm', cluster='local',
                                       work_dir=work_dir,
                                       submitit_kwargs={'not_a_real_parameter': 1})
    surrogate.fit(X, y)  # must not raise
    assert len(surrogate._experts) == 2
    assert np.all(np.isfinite(surrogate.predict(np.zeros((1, 2)))))


def test_failed_expert_is_retried_then_fitted_locally(work_dir, monkeypatch):
    """
    A node failure or scheduler hiccup on one task must not discard the experts that did
    succeed. Failures are retried individually and, if still failing, fitted locally -- a
    surrogate short of a few experts is a slightly worse surrogate, not a crash.
    """
    from slurmcmc.hybrid import DistributedGPSurrogate

    surrogate = DistributedGPSurrogate(num_experts=3, n_restarts_optimizer=0,
                                       min_points_per_expert=10,
                                       parallel='slurm', cluster='local',
                                       work_dir=work_dir, max_fit_retries=2)
    calls = {'n': 0}
    original = DistributedGPSurrogate._collect_with_retry

    def flaky(self, executor, batches, args, indices, experts):
        """Fail expert 0 on the first submission only; the rest succeed normally."""
        calls['n'] += 1
        failed = original(self, executor, batches, args, indices, experts)
        if calls['n'] == 1 and 0 in indices:
            experts[0] = None
            failed = sorted(set(failed) | {0})
        return failed

    monkeypatch.setattr(DistributedGPSurrogate, '_collect_with_retry', flaky)
    X, y = _rbcm_training_set(n=300)
    surrogate.fit(X, y)

    assert calls['n'] >= 2, 'the failed expert was never retried'
    assert len(surrogate._experts) == 3
    assert all(e is not None for e in surrogate._experts), 'an expert slot was left empty'
    assert np.all(np.isfinite(surrogate.predict(np.array([[0.0, 0.0], [1.5, -1.0]]))))
    leftovers = [os.path.join(root, f)
                 for root, _, files in os.walk(work_dir) for f in files]
    assert leftovers == [], 'retry folders were not cleaned up'


def test_surrogate_inherits_work_dir_unless_set_explicitly():
    """
    A surrogate that distributes its own fitting writes scratch files, and on a cluster those
    have to be somewhere the compute nodes can read. Its default is relative to the driver's
    cwd, which is not that -- so the pipeline hands down its own work_dir, which is already
    required to be on a shared filesystem. An explicit choice by the caller still wins.
    """
    from slurmcmc.hybrid import DistributedGPSurrogate, HybridMCMCConfig, HybridMCMCRunner
    shared = dict(log_prob_fun=log_prob_ring, init_points=np.zeros((4, 2)),
                  param_bounds=param_bounds, work_dir='/shared/run17', verbosity=0)

    inherited = HybridMCMCRunner(HybridMCMCConfig(
        surrogate=DistributedGPSurrogate(parallel='slurm'), **shared))._create_surrogate()
    assert inherited.work_dir == '/shared/run17'

    explicit = HybridMCMCRunner(HybridMCMCConfig(
        surrogate=DistributedGPSurrogate(parallel='slurm', work_dir='/explicit'),
        **shared))._create_surrogate()
    assert explicit.work_dir == '/explicit'

    # the job name too, so the experts' jobs are recognisable in the queue
    assert inherited.job_name == 'mcmc_hybrid'
    named = HybridMCMCRunner(HybridMCMCConfig(
        surrogate=DistributedGPSurrogate(parallel='slurm', job_name='mine'),
        **shared))._create_surrogate()
    assert named.job_name == 'mine'


def test_slurm_expert_fits_are_named_and_kept_under_surrogates(work_dir, monkeypatch):
    """
    Expert job arrays carry the run's job name rather than submitit's default 'submitit', and
    their scratch folder is work_dir/surrogates/dgp_experts, removed (with its empty parents)
    once every expert is back.
    """
    import submitit
    from slurmcmc.hybrid import DistributedGPSurrogate
    seen = []
    real_executor = submitit.AutoExecutor

    class RecordingExecutor(real_executor):
        def __init__(self, folder, **kwargs):
            super().__init__(folder=folder, **kwargs)
            self._folder = folder

        def update_parameters(self, **kwargs):
            seen.append((self._folder, kwargs.get('slurm_job_name')))
            super().update_parameters(**kwargs)

    monkeypatch.setattr(submitit, 'AutoExecutor', RecordingExecutor)
    X, y = _rbcm_training_set(n=300)
    surrogate = DistributedGPSurrogate(num_experts=3, n_restarts_optimizer=0, min_points_per_expert=10,
                                       parallel='slurm', cluster='local', work_dir=work_dir,
                                       job_name='run17')
    surrogate.fit(X, y)
    surrogate.fit(X, y)

    assert [name for _, name in seen] == ['run17_dgp_fit1', 'run17_dgp_fit2']
    assert seen[0][0] == os.path.join(work_dir, 'surrogates', 'dgp_experts', 'fit1_3experts')
    assert os.listdir(work_dir) == []


@pytest.mark.skipif(not is_slurm_cluster(), reason="This test only runs on a Slurm cluster")
def test_distributed_gp_experts_on_slurm(work_dir):
    """
    Fit the rBCM's experts as a real Slurm job array.

    Deliberately tiny -- three experts on a few hundred points, no optimizer restarts -- so
    the wall time is one round of queue latency rather than compute. It exists to check the
    things a local submitit executor cannot: that sbatch accepts the parameters, that the
    compute nodes can import slurmcmc and read the scratch folder, that the fitted GPs
    survive the round trip as pickles, and that the scratch files are cleaned up afterwards.
    Speedup is not measured here; that needs a realistic workload.
    """
    from slurmcmc.hybrid import DistributedGPSurrogate

    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, (300, 2))
    y = np.array([log_prob_ring(point) for point in X])
    probe = np.array([[0.0, 0.0], [1.5, -1.0], [-2.0, 2.0]])

    serial = DistributedGPSurrogate(num_experts=3, n_restarts_optimizer=0, parallel='none',
                                     min_points_per_expert=10)
    serial.fit(X, y)

    distributed = DistributedGPSurrogate(num_experts=3, n_restarts_optimizer=0,
                                         min_points_per_expert=10,
                                         parallel='slurm', cluster='slurm',
                                         work_dir=work_dir, submitit_kwargs=submitit_kwargs)
    distributed.fit(X, y)

    assert len(distributed._experts) == 3, 'experts did not come back from the cluster'
    np.testing.assert_allclose(distributed.predict(probe), serial.predict(probe), atol=1e-2)
    leftovers = [os.path.join(root, f)
                 for root, _, files in os.walk(work_dir) for f in files]
    assert leftovers == [], f'scratch files left on the shared filesystem: {leftovers[:3]}'


def test_rbcm_expert_seeds_reach_every_backend(monkeypatch):
    """
    Each expert's hyperparameter restarts get an explicit seed drawn in the caller: fitted by
    joblib or submitit, an expert runs in another process, which a global seed never reaches.
    The seeds handed to the backend are the assertion, because they are exact: an expert fitted
    in a worker process runs its linear algebra with a different thread count, so the optimizer's
    last bits -- and with them the prediction -- are not reproducible to machine precision, and
    the predictions are only compared loosely.
    """
    from slurmcmc.general_utils import seed_random_generators
    from slurmcmc.hybrid import DistributedGPSurrogate
    X, y = _rbcm_training_set(n=300)
    probe, _ = _rbcm_training_set(n=40, seed=3)

    handed_out = []
    original = DistributedGPSurrogate._fit_experts

    def recording_fit_experts(self, batches):
        handed_out.append((self.parallel, [seed for _, _, seed in batches]))
        return original(self, batches)

    monkeypatch.setattr(DistributedGPSurrogate, '_fit_experts', recording_fit_experts)

    def fit_and_predict(parallel, seed):
        seed_random_generators(seed)
        surrogate = DistributedGPSurrogate(num_experts=3, min_points_per_expert=10,
                                           n_restarts_optimizer=2, parallel=parallel, n_jobs=3)
        surrogate.fit(X, y)
        return np.asarray(surrogate.predict(probe))

    serial = fit_and_predict('none', 5)
    parallel = fit_and_predict('joblib', 5)
    fit_and_predict('none', 6)

    assert [name for name, _ in handed_out] == ['none', 'joblib', 'none']
    assert handed_out[0][1] == handed_out[1][1], 'the backend changed the seeds the experts get'
    assert handed_out[2][1] != handed_out[0][1], 'the seeds ignore the global generator'
    np.testing.assert_allclose(parallel, serial, rtol=1e-4, atol=1e-6)
