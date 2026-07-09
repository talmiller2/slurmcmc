from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


class BoTorchOptimizer:
    """
    Bayesian optimization backend for slurm_minimize, based on a Gaussian Process
    surrogate (SingleTaskGP) with the qLogExpectedImprovement acquisition function.

    Parameters
    ----------
    lower_bounds, upper_bounds : list[float]
        Per-parameter search-space bounds.
    num_workers : int
        Number of candidate points returned by each ask() call (the q in qEI).
    num_restarts : int
        Number of restarts for the acquisition-function optimization.
    raw_samples : int
        Number of raw samples used to seed the acquisition-function optimization.
    num_best_points : int or None
        If set, the GP is trained only on the num_best_points evaluations with
        the lowest loss values.  Keeps GP fitting tractable for long runs
        (GP training is O(n^3) in the number of points, so this matters once
        the history reaches several hundred points).
    options : dict or None
        Extra options forwarded to botorch's optimize_acqf.
    sequential : bool
        If True (default), the q=num_workers batch candidates are selected by
        greedy sequential optimization instead of a single joint optimization
        over the q*d-dimensional space.  Joint optimization is the dominant
        cost of botorch at moderate/large num_workers and blows up quickly;
        sequential greedy is the standard botorch recommendation and gives a
        large speedup (often 4-5x at num_workers~10) with equivalent quality.
    """

    def __init__(
        self,
        lower_bounds: List[float],
        upper_bounds: List[float],
        num_workers: int,
        num_restarts: int,
        raw_samples: int,
        num_best_points: Optional[int],
        options: Optional[Dict],
        sequential: bool = True,
    ) -> None:
        self.bounds_torch = torch.tensor([lower_bounds, upper_bounds], dtype=torch.float)
        self.num_workers = num_workers
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.num_best_points = num_best_points
        self.options = options
        self.sequential = sequential

    def ask(self, x_pts: np.ndarray, y_pts: np.ndarray) -> np.ndarray:
        """
        Propose num_workers new candidate points given the evaluation history.

        On the first call (empty history) returns quasi-random Sobol samples;
        afterwards fits a GP to (x_pts, y_pts) and maximizes qLogExpectedImprovement.
        """

        if len(x_pts) == 0:  # initial iteration
            points_torch = draw_sobol_samples(bounds=self.bounds_torch, n=1, q=self.num_workers).squeeze(0)

        else:
            # trim the training data to the num_best_points of lowest loss_fun values
            if self.num_best_points is not None and self.num_best_points < x_pts.shape[0]:
                indices_sorted = np.argsort(y_pts[:, 0])
                x_pts = x_pts[indices_sorted[0:self.num_best_points]]
                y_pts = y_pts[indices_sorted[0:self.num_best_points]]

            x_torch = torch.tensor(x_pts, dtype=torch.float64)  # botorch recommends float64
            y_torch = torch.tensor(y_pts[:, 0], dtype=torch.float64).unsqueeze(-1)
            y_torch *= -1  # botorch does maximization rather than minimization
            model = SingleTaskGP(x_torch, y_torch,
                                 input_transform=Normalize(d=x_pts.shape[1]),
                                 outcome_transform=Standardize(m=1),
                                 )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            acq_function = qLogExpectedImprovement(model=model, best_f=torch.max(y_torch))
            points_torch, _ = optimize_acqf(acq_function=acq_function, bounds=self.bounds_torch, q=self.num_workers,
                                            num_restarts=self.num_restarts, raw_samples=self.raw_samples,
                                            options=self.options, sequential=self.sequential)

        points_torch = points_torch.to(torch.float64)  # without this points are float32 which is not JSON serializable
        points = points_torch.numpy()  # convert to np.array
        return points
