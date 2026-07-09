"""
History — evaluation-history tracking for SlurmPool.

Decoupled from SlurmPool so that history data can be inspected, serialised,
and reasoned about independently of the pool's job-submission machinery.

SlurmPool exposes every attribute here as a read-only @property for full
backward compatibility (existing code using slurm_pool.points_history etc.
continues to work unchanged).
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np

from slurmcmc.general_utils import point_to_tuple


class History:
    """
    Stores the full record of every function evaluation made by a SlurmPool.

    Attributes
    ----------
    points_history : np.ndarray, shape (n, dim_input)
        All evaluated input points, in submission order.
    values_history : np.ndarray, shape (n, dim_output)
        Corresponding function outputs (failed points store job_fail_value).
    inds_success_points : list[int]
        Row indices into points_history of successful evaluations.
    inds_failed_points : list[int]
        Row indices into points_history of failed evaluations.
    evaluated_points_set : set
        Set of point tuples for O(1) membership testing.
    point_loc_dict : dict
        Maps each point tuple to (num_calls, ind_point) — the on-disk
        sub-directory where the evaluation result lives.
    num_evaluated_points : int
        Total number of points evaluated (successes + failures).
    """

    def __init__(self, dim_input: int, dim_output: int) -> None:
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_evaluated_points: int = 0
        self.points_history = []          # grows into np.ndarray on first record()
        self.values_history = []
        self.inds_success_points: List[int] = []
        self.inds_failed_points: List[int] = []
        self.evaluated_points_set: set = set()
        self.point_loc_dict: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _vstack(history, new_data: np.ndarray, dim: int) -> np.ndarray:
        """Append rows of new_data to history, reshaping scalars to (n,1)."""
        x = np.array(new_data)
        if dim == 1:
            x = x.reshape(-1, 1)
        if len(history) == 0:
            return x
        return np.vstack([history, x])

    # ------------------------------------------------------------------
    # Two-phase update API (matches SlurmPool's submit-then-collect flow)
    # ------------------------------------------------------------------

    def record_point_locations(self, points, num_calls: int) -> None:
        """
        Phase 1 — called during job *submission*, before results are available.

        Registers the on-disk location (num_calls, ind_point) for every point
        that has not been seen before.  Points already in evaluated_points_set
        (re-evaluated points) are skipped so point_loc_dict always holds the
        location of the *first* evaluation of each point.
        """
        for ind, point in enumerate(points):
            pt = point_to_tuple(point)
            if pt not in self.evaluated_points_set:
                self.point_loc_dict[pt] = (num_calls, ind)

    def record(self, points, results, check_failed_fn: Callable) -> None:
        """
        Phase 2 — called after results are collected (in map_chunk).

        Updates evaluated_points_set, success/failure index lists, and the
        points_history / values_history arrays.
        """
        # Mark every point in this batch as evaluated
        for point in points:
            self.evaluated_points_set.add(point_to_tuple(point))

        # Classify each result
        base = self.num_evaluated_points
        self.inds_failed_points  += [base + i for i, v in enumerate(results) if     check_failed_fn(v)]
        self.inds_success_points += [base + i for i, v in enumerate(results) if not check_failed_fn(v)]

        # Extend history arrays
        self.points_history = self._vstack(self.points_history, np.array(points),  self.dim_input)
        self.values_history = self._vstack(self.values_history, np.array(results), self.dim_output)
        self.num_evaluated_points += len(results)
