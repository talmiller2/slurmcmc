import warnings

import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.exceptions import InputDataWarning, OptimizationWarning
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

# suppress some of the botorch warnings
warnings.filterwarnings("ignore", category=InputDataWarning)
warnings.filterwarnings("ignore", category=OptimizationWarning)


class BoTorchOptimizer():
    def __init__(self, lower_bounds, upper_bounds, num_workers,
                 num_restarts=None, raw_samples=None, num_best_points=None):
        self.bounds_torch = torch.tensor([lower_bounds, upper_bounds], dtype=torch.float)
        self.num_workers = num_workers
        if num_restarts is not None:
            self.num_restarts = num_restarts
        else:
            self.num_restarts = num_workers

        if raw_samples is not None:
            self.raw_samples = raw_samples
        else:
            self.raw_samples = num_workers

        self.num_best_points = num_best_points

    def ask(self, x_pts, y_pts):

        if len(x_pts) == 0:  # initial iteration
            botorch_points = draw_sobol_samples(bounds=self.bounds_torch, n=1, q=self.num_workers).squeeze(0)

        else:
            # trim the training data to the num_best_points of lowest loss_fun values
            if self.num_best_points is not None and self.num_best_points > x_pts.shape[0]:
                indices_sorted = np.argsort(y_pts[:, 0])
                x_pts = x_pts[indices_sorted[0:self.num_best_points]]
                y_pts = y_pts[indices_sorted[0:self.num_best_points]]

            x_torch = torch.tensor(x_pts, dtype=torch.float64)
            y_torch = torch.tensor(y_pts[:, 0], dtype=torch.float64).unsqueeze(-1)
            y_torch *= -1  # botorch does maximization rather than minimization
            model = SingleTaskGP(x_torch, y_torch)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            acq_function = qExpectedImprovement(model=model, best_f=torch.max(y_torch))
            botorch_points, _ = optimize_acqf(acq_function=acq_function, bounds=self.bounds_torch, q=self.num_workers,
                                              num_restarts=self.num_restarts, raw_samples=self.raw_samples)

        points = list(np.asarray(botorch_points))
        return points
