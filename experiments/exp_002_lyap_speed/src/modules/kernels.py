"""Faithful split of the package kernel for native stage attribution."""

import numpy as np
from numba import njit


@njit(cache=True)
def neighbours_original(y, min_tsep, trajectory_len):
    """Retain the original all-pairs search including its array temporaries."""
    n_traj = y.shape[0] - trajectory_len + 1
    neighbour = np.empty(n_traj, dtype=np.int64)
    for i in range(n_traj):
        best, best_j = np.inf, -1
        for j in range(n_traj):
            if abs(i - j) <= min_tsep:
                continue
            d = np.sum((y[i] - y[j]) ** 2)
            if d < best:
                best, best_j = d, j
        neighbour[i] = best_j
    return neighbour


@njit(cache=True)
def follow_divergence(y, neighbour, trajectory_len):
    """Keep original arithmetic and ascending-row reduction order unchanged."""
    n_traj = len(neighbour)
    log_div = np.full(trajectory_len, -np.inf)
    for k in range(trajectory_len):
        total, count = 0.0, 0
        for i in range(n_traj):
            d = np.sqrt(np.sum((y[i + k] - y[neighbour[i] + k]) ** 2))
            if d > 0:
                total += np.log(d)
                count += 1
        if count > 0:
            log_div[k] = total / count
    return log_div
