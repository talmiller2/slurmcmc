"""
Surrogate models for the hybrid pipeline (see hybrid.py).

Anything with ``fit(X, y)`` and ``predict(X)`` works as a surrogate; ``predict_std(X)``
additionally enables the uncertainty penalty (the MCMC samples ``mean - kappa * std``).

* :class:`GaussianProcessSurrogate` -- the default, for up to a few thousand training points.
* :class:`DistributedGPSurrogate` -- an ensemble of GPs, for training sets too large to refit
  a single one every round.
* :class:`PolynomialSurrogate` -- a cheap fallback with no predictive uncertainty.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from scipy.optimize import minimize

from slurmcmc.slurm_utils import Cluster


def _lbfgs_with_more_iterations(obj_func, initial_theta, bounds):
    """
    L-BFGS-B for GaussianProcessRegressor with a raised iteration cap, which avoids sklearn's
    ABNORMAL_TERMINATION_IN_LNSRCH on ill-conditioned fits. Module-level so it pickles.
    """
    result = minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds,
                      options={"maxiter": 50000, "maxfun": 50000, "ftol": 1e-12, "gtol": 1e-8})
    return result.x, result.fun


class GaussianProcessSurrogate:
    """
    Gaussian-process surrogate (sklearn GaussianProcessRegressor with an ARD-RBF kernel).

    A polynomial mean function (quadratic by default, as a log-posterior is near its mode) is
    fitted first and the GP models only the residual; the GP is skipped when the trend already
    explains the data to within residual_tolerance nats. Inputs are standardized, and
    predictions are clamped to the observed range plus a margin.
    """

    def __init__(self, kernel=None, n_restarts_optimizer: int = 5, alpha: float = 1e-6,
                 normalize_inputs: bool = True,
                 trend_degree: Optional[int] = 2, residual_tolerance: float = 1e-3):
        self.kernel = kernel
        self.n_restarts_optimizer = n_restarts_optimizer
        self.alpha = alpha
        self.normalize_inputs = normalize_inputs
        self.trend_degree = trend_degree
        self.residual_tolerance = residual_tolerance
        self._gpr = None
        self._scaler = None
        self._trend_model = None
        self._residual_scale = 1.0
        self._predict_lo = None
        self._predict_hi = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # sklearn is an optional dependency (install with: pip install -e ".[hybrid]")
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler

        X = np.asarray(X, dtype=float)
        y = np.ravel(np.asarray(y, dtype=float))

        # clamp predictions to the observed range plus a margin: the polynomial trend can
        # extrapolate to a spurious peak far above anything observed, which the MCMC would chase
        y_lo, y_hi = float(np.min(y)), float(np.max(y))
        pad = max(0.1 * (y_hi - y_lo), 1.0)
        self._predict_lo, self._predict_hi = y_lo - pad, y_hi + pad

        if self.normalize_inputs:
            self._scaler = StandardScaler().fit(X)
            X_fit = self._scaler.transform(X)
        else:
            self._scaler = None
            X_fit = X

        # polynomial mean function
        if self.trend_degree is not None and len(y) > 0:
            self._trend_model = make_pipeline(PolynomialFeatures(degree=self.trend_degree),
                                              Ridge(alpha=1e-6))
            self._trend_model.fit(X_fit, y)
            residual = y - self._trend_model.predict(X_fit)
        else:
            self._trend_model = None
            residual = y

        # scaled explicitly rather than with normalize_y, which would inflate a residual that is
        # pure round-off up to unit variance and fit the GP to noise
        self._residual_scale = float(np.std(residual))
        if self._residual_scale <= self.residual_tolerance:
            self._gpr = None  # the trend explains everything; no GP correction needed
            return
        residual_scaled = residual / self._residual_scale

        kernel = self.kernel
        if kernel is None:
            # length scales may exceed the standardized data extent: flat directions want them long
            kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                      * RBF(np.ones(X_fit.shape[1]), (1e-2, 1e2)))

        self._gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=self.n_restarts_optimizer,
                                             alpha=self.alpha, normalize_y=False,
                                             optimizer=_lbfgs_with_more_iterations)
        self._gpr.fit(X_fit, residual_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self._scaler is not None:
            X = self._scaler.transform(X)
        prediction = np.zeros(len(X))
        if self._gpr is not None:
            prediction = np.ravel(self._gpr.predict(X)) * self._residual_scale
        if self._trend_model is not None:
            prediction = prediction + np.ravel(self._trend_model.predict(X))
        if self._predict_lo is not None:
            prediction = np.clip(prediction, self._predict_lo, self._predict_hi)
        return prediction

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """Predictive standard deviation of the surrogate, in nats."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self._gpr is None:
            return np.zeros(len(X))  # trend-only fit: no GP correction, hence no GP variance
        if self._scaler is not None:
            X = self._scaler.transform(X)
        _, std = self._gpr.predict(X, return_std=True)
        return np.ravel(std) * self._residual_scale


def _fit_expert(X: np.ndarray, y: np.ndarray, n_restarts_optimizer: int,
                alpha: float, random_state: Optional[int] = None) -> Any:
    """
    Fit one rBCM expert. Module-level so it pickles for joblib and submitit. random_state is
    explicit because those backends fit in another process, which a global seed never reaches.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel

    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * RBF(np.ones(X.shape[1]), (1e-2, 1e2)))
    expert = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer,
                                      alpha=alpha, normalize_y=False,
                                      optimizer=_lbfgs_with_more_iterations,
                                      random_state=random_state)
    expert.fit(X, y)
    return expert


class _DetrendedSurrogateBase:
    """Preprocessing shared with GaussianProcessSurrogate: standardize, detrend, scale, clamp."""

    def __init__(self, trend_degree=2, residual_tolerance: float = 1e-3):
        self.trend_degree = trend_degree
        self.residual_tolerance = residual_tolerance
        self._scaler = None
        self._trend_model = None
        self._residual_scale = 1.0
        self._predict_lo = None
        self._predict_hi = None
        self._fitted = False

    def _preprocess(self, X, y):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler

        X = np.asarray(X, dtype=float)
        y = np.ravel(np.asarray(y, dtype=float))
        y_lo, y_hi = float(np.min(y)), float(np.max(y))
        pad = max(0.1 * (y_hi - y_lo), 1.0)
        self._predict_lo, self._predict_hi = y_lo - pad, y_hi + pad

        self._scaler = StandardScaler().fit(X)
        X_fit = self._scaler.transform(X)

        if self.trend_degree is not None:
            self._trend_model = make_pipeline(PolynomialFeatures(degree=self.trend_degree),
                                              Ridge(alpha=1e-6))
            self._trend_model.fit(X_fit, y)
            residual = y - self._trend_model.predict(X_fit)
        else:
            self._trend_model = None
            residual = y

        self._residual_scale = float(np.std(residual))
        if self._residual_scale <= self.residual_tolerance:
            return X_fit, None  # the trend explains everything
        return X_fit, residual / self._residual_scale

    def _postprocess(self, X, residual_prediction):
        prediction = residual_prediction * self._residual_scale
        if self._trend_model is not None:
            prediction = prediction + np.ravel(self._trend_model.predict(X))
        return np.clip(prediction, self._predict_lo, self._predict_hi)


class DistributedGPSurrogate(_DetrendedSurrogateBase):
    """
    Robust Bayesian Committee Machine (Deisenroth & Ng 2015, with Cao & Fleet's 2014 weights).

    Splits the training set among independent GP experts, fitted in parallel, and recombines
    their predictions in closed form:

        beta_k   = 0.5 (log sigma_k,prior^2 - log sigma_k^2(x))
        w_k      = beta_k / sum_j beta_j
        sigma^-2 = sum_k w_k sigma_k^-2
        mu       = sigma^2 sum_k w_k sigma_k^-2 mu_k

    sigma_k,prior^2 is the prior variance each expert actually fitted, not a nominal 1: experts
    routinely learn amplitudes far from 1, and a nominal prior would zero every weight. The
    normalized weights make a one-expert ensemble reproduce the dense GP exactly, and fall back
    to uniform where no expert is informed, so the ensemble reverts to its experts' priors.
    """

    def __init__(self, num_experts: int = 8, partition: str = 'kmeans',
                 n_restarts_optimizer: int = 2, alpha: float = 1e-6,
                 max_points_per_expert: Optional[int] = None,
                 min_points_per_expert: int = 200,
                 parallel: Literal['none', 'joblib', 'slurm'] = 'none', n_jobs: int = -1,
                 cluster: Cluster = 'slurm', work_dir: Optional[str] = None,
                 job_name: Optional[str] = None,
                 submitit_kwargs: Optional[Dict] = None, max_fit_retries: int = 2,
                 trend_degree=2, residual_tolerance: float = 1e-3, random_state: int = 0):
        super().__init__(trend_degree=trend_degree, residual_tolerance=residual_tolerance)
        # 'none': serial; 'joblib': one process per core; 'slurm': one job array across nodes,
        # worth it only when a single expert fit is long compared with the queue wait
        self.parallel = parallel
        self.n_jobs = n_jobs
        # submitit backend of the 'slurm' mode; 'local' runs subprocesses on this machine
        self.cluster = cluster
        self.max_fit_retries = max_fit_retries
        self._fit_count = 0  # names each fit's scratch folder and Slurm job
        # 'slurm' mode: scratch folders go to <work_dir>/surrogates/dgp_experts, and the job arrays
        # are named <job_name>_dgp_fit<n>; slurm_mcmc_hybrid fills both in from its own when None
        self.work_dir = work_dir
        self.job_name = job_name
        self.submitit_kwargs = submitit_kwargs
        # grows the number of experts with the data instead of the size of each
        self.max_points_per_expert = max_points_per_expert
        # an expert on too few points cannot identify its own kernel; below this floor fewer
        # experts are used, down to one, which is exactly GaussianProcessSurrogate
        self.min_points_per_expert = min_points_per_expert
        self.num_experts = num_experts
        self.partition = partition
        self.n_restarts_optimizer = n_restarts_optimizer
        self.alpha = alpha
        self.random_state = random_state
        self._experts = []
        self._expert_prior_vars = np.ones(0)

    def _assign(self, X_fit):
        rng = np.random.default_rng(self.random_state)
        if self.max_points_per_expert:
            num_experts = max(1, int(np.ceil(len(X_fit) / self.max_points_per_expert)))
        else:
            num_experts = self.num_experts
        affordable = len(X_fit) // max(1, self.min_points_per_expert)
        num_experts = max(1, min(num_experts, affordable))
        if num_experts == 1:
            return [np.arange(len(X_fit))]
        if self.partition == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_experts, n_init=1,
                            random_state=self.random_state).fit(X_fit)
            labels = kmeans.labels_
            groups = [np.where(labels == k)[0] for k in range(num_experts)]
            return self._absorb_stragglers(groups, X_fit, kmeans.cluster_centers_)
        order = rng.permutation(len(X_fit))
        return [g for g in np.array_split(order, num_experts) if len(g) >= 5]

    @staticmethod
    def _absorb_stragglers(groups: List, X_fit, centers) -> List:
        """Hand the points of clusters too small to fit to the nearest surviving expert."""
        keep = [k for k, g in enumerate(groups) if len(g) >= 5]
        if len(keep) == len(groups):
            return groups
        if not keep:
            return [np.arange(len(X_fit))]
        orphans = np.concatenate([groups[k] for k in range(len(groups)) if k not in keep])
        distances = np.linalg.norm(X_fit[orphans][:, None, :] - centers[keep][None, :, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        return [np.concatenate([groups[k], orphans[nearest == i]]) for i, k in enumerate(keep)]

    def fit(self, X, y):
        X_fit, residual = self._preprocess(X, y)
        self._experts = []
        if residual is None:
            self._fitted = True
            return

        groups = self._assign(X_fit)
        # one seed per expert from the caller's generator: the same fits on every backend
        seeds = np.random.randint(0, 2 ** 31 - 1, size=len(groups))
        batches = [(X_fit[i], residual[i], int(seed)) for i, seed in zip(groups, seeds)]
        self._experts = self._fit_experts(batches)
        self._expert_prior_vars = self._prior_variances(self._experts, X_fit.shape[1])
        self._fitted = True

    @staticmethod
    def _prior_variances(experts: List, num_dims: int) -> np.ndarray:
        """Each expert's own prior variance, k(x, x) of its fitted kernel."""
        probe = np.zeros((1, num_dims))
        priors = []
        for expert in experts:
            kernel = getattr(expert, 'kernel_', None)
            priors.append(float(np.ravel(kernel.diag(probe))[0]) if kernel is not None else 1.0)
        return np.maximum(np.asarray(priors, dtype=float), 1e-12)

    def _fit_experts(self, batches: List) -> List:
        """Fit every expert, using the configured backend. Falls back to serial on failure."""
        args = (self.n_restarts_optimizer, self.alpha)
        if self.parallel == 'joblib' and len(batches) > 1:
            try:
                from joblib import Parallel, delayed
                return list(Parallel(n_jobs=self.n_jobs)(
                    delayed(_fit_expert)(Xb, yb, *args, sb) for Xb, yb, sb in batches))
            except Exception:
                logging.warning('joblib expert fitting failed; falling back to serial.',
                                exc_info=True)
        elif self.parallel == 'slurm' and len(batches) > 1:
            try:
                return self._fit_experts_slurm(batches, args)
            except Exception:
                logging.warning('slurm expert fitting failed; falling back to serial.',
                                exc_info=True)
        return [_fit_expert(Xb, yb, *args, sb) for Xb, yb, sb in batches]

    def _collect_with_retry(self, executor, batches: List, args, indices: List[int],
                            experts: List) -> List[int]:
        """Submit one job array for `indices`, collect it expert by expert, return the failed indices."""
        jobs = executor.map_array(_fit_expert,
                                  [batches[i][0] for i in indices],
                                  [batches[i][1] for i in indices],
                                  [args[0]] * len(indices), [args[1]] * len(indices),
                                  [batches[i][2] for i in indices])
        failed = []
        for index, job in zip(indices, jobs):
            try:
                experts[index] = job.result()
            except Exception:
                logging.warning(f'expert {index} failed on the cluster.', exc_info=True)
                failed.append(index)
        return failed

    def _fit_experts_slurm(self, batches: List, args) -> List:
        """
        One submitit job array over the experts. Failed experts are retried, then fitted locally;
        the scratch folder (pickled fits, megabytes each) is removed once all are in memory.
        """
        import shutil

        import submitit

        self._fit_count += 1
        folder = os.path.join(self.work_dir or '.', 'surrogates', 'dgp_experts',
                              f'fit{self._fit_count}_{len(batches)}experts')
        os.makedirs(folder, exist_ok=True)
        executor = submitit.AutoExecutor(folder=folder, cluster=self.cluster)
        # submitit's 5-minute default limit is short for a GP fit, and its default job name is
        # 'submitit'; submitit_kwargs still win
        job_name = f'{self.job_name}_dgp_fit{self._fit_count}' if self.job_name else f'dgp_fit{self._fit_count}'
        parameters = {'timeout_min': 120, 'slurm_job_name': job_name}
        parameters.update(dict(self.submitit_kwargs or {}))
        executor.update_parameters(**parameters)
        experts: List = [None] * len(batches)
        pending = self._collect_with_retry(executor, batches, args,
                                           list(range(len(batches))), experts)
        for attempt in range(1, self.max_fit_retries + 1):
            if not pending:
                break
            logging.warning(f'retrying {len(pending)} failed expert fit(s), '
                            f'attempt {attempt}/{self.max_fit_retries}.')
            retry_folder = os.path.join(folder, f'retry{attempt}')
            os.makedirs(retry_folder, exist_ok=True)
            retry_executor = submitit.AutoExecutor(folder=retry_folder, cluster=self.cluster)
            retry_executor.update_parameters(**{**parameters,
                                                'slurm_job_name': f"{parameters['slurm_job_name']}_retry{attempt}"})
            pending = self._collect_with_retry(retry_executor, batches, args, pending, experts)
        if pending:
            logging.warning(f'{len(pending)} expert(s) still failing after '
                            f'{self.max_fit_retries} retries; fitting them locally.')
            for index in pending:
                experts[index] = _fit_expert(batches[index][0], batches[index][1], *args,
                                             batches[index][2])

        # only once every result is safely in memory
        try:
            shutil.rmtree(folder, ignore_errors=True)
            for parent in (os.path.dirname(folder), os.path.dirname(os.path.dirname(folder))):
                if os.path.isdir(parent) and not os.listdir(parent):
                    os.rmdir(parent)  # dgp_experts, then surrogates, when nothing else is in them
        except OSError:
            logging.debug('could not remove the expert scratch folder.', exc_info=True)
        return experts

    def _combine(self, X_scaled):
        """rBCM mean and variance of the scaled residual."""
        if not self._experts:
            return np.zeros(len(X_scaled)), np.zeros(len(X_scaled))
        means, variances = [], []
        for expert in self._experts:
            mean, std = expert.predict(X_scaled, return_std=True)
            means.append(np.ravel(mean))
            variances.append(np.maximum(np.ravel(std) ** 2, 1e-12))
        means = np.asarray(means)
        variances = np.asarray(variances)

        # an uninformed expert has sigma_k^2 -> sigma_k,prior^2, so beta_k -> 0
        prior = self._prior_vars_for(means.shape[0], X_scaled.shape[1])[:, None]
        beta = np.maximum(0.5 * (np.log(prior) - np.log(variances)), 0.0)

        # normalized; uniform where no expert is informed, reverting to the experts' priors
        total = beta.sum(axis=0)
        uninformed = total <= 0.0
        if np.any(uninformed):
            beta[:, uninformed] = 1.0
            total = beta.sum(axis=0)
        weights = beta / total

        precision = np.sum(weights / variances, axis=0)
        combined_var = 1.0 / precision
        combined_mean = combined_var * np.sum(weights * means / variances, axis=0)
        return combined_mean, np.maximum(combined_var, 0.0)

    def _prior_vars_for(self, num_experts: int, num_dims: int) -> np.ndarray:
        """Cached expert priors, recomputed for surrogates pickled before they were stored."""
        priors = getattr(self, '_expert_prior_vars', None)
        if priors is None or len(priors) != num_experts:
            priors = self._prior_variances(self._experts, num_dims)
            self._expert_prior_vars = priors
        return priors

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        X_scaled = self._scaler.transform(X)
        mean, _ = self._combine(X_scaled)
        return self._postprocess(X_scaled, mean)

    def predict_std(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        X_scaled = self._scaler.transform(X)
        _, variance = self._combine(X_scaled)
        return np.sqrt(variance) * self._residual_scale


class PolynomialSurrogate:
    """
    Polynomial (ridge-regularized) surrogate: a cheap fallback for smooth, low-order posteriors.
    It extrapolates badly and has no predict_std, so outside param_bounds only the confinement
    envelope limits the walkers.
    """

    def __init__(self, degree: int = 3, ridge_alpha: float = 1e-6):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures

        self._model = make_pipeline(PolynomialFeatures(degree=degree), Ridge(alpha=ridge_alpha))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(np.asarray(X), np.ravel(y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ravel(self._model.predict(np.atleast_2d(X)))
