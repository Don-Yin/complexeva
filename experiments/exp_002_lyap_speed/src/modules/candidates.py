"""Named speed candidates; retain all methods because comparison is the deliverable."""

import numpy as np
from numba import njit, prange
from scipy.spatial import KDTree

from .kernels import follow_divergence


def _scalar_search(y, min_tsep, trajectory_len):
    """Scan in ascending index order without allocating a distance array."""
    n_traj = y.shape[0] - trajectory_len + 1
    neighbour = np.empty(n_traj, dtype=np.int64)
    for i in prange(n_traj):
        best, best_j = np.inf, -1
        for j in range(n_traj):
            if abs(i - j) <= min_tsep:
                continue
            d = 0.0
            for dimension in range(y.shape[1]):
                delta = y[i, dimension] - y[j, dimension]
                d += delta * delta
            if d < best:
                best, best_j = d, j
        neighbour[i] = best_j
    return neighbour


# Avoid cache collisions between differently compiled views of one source function.
# Compile the same numerical loop twice to isolate parallel scheduling cost.
neighbours_scalar = njit(cache=False)(_scalar_search)
neighbours_parallel = njit(cache=False, parallel=True)(_scalar_search)


@njit(cache=True)
def _select_candidates(y, rows, indices, distances, min_tsep, n_traj, complete):
    """Recompute exact squared distances and resolve every tie by lowest index."""
    selected = np.empty(len(rows), dtype=np.int64)
    finished = np.empty(len(rows), dtype=np.bool_)
    margin = 1.0 + 8 * y.shape[1] * np.finfo(np.float64).eps
    for row in range(len(rows)):
        i = rows[row]
        best, best_j = np.inf, -1
        for j in indices[row]:
            # KDTree uses tree.n for unavailable/overflowed-distance neighbours.
            if j == n_traj or abs(i - j) <= min_tsep:
                continue
            d = 0.0
            for dimension in range(y.shape[1]):
                delta = y[i, dimension] - y[j, dimension]
                d += delta * delta
            if d < best or (d == best and j < best_j):
                best, best_j = d, j
        selected[row] = best_j
        # A strict gap ensures tied neighbours were not truncated by k.
        # The rounding guard also admits candidates at a near-equal radius.
        finished[row] = complete or (best_j >= 0 and distances[row, -1] > np.sqrt(best) * margin)
    return selected, finished


def neighbours_kdtree(y, min_tsep, trajectory_len):
    """Expand exact KD queries until exclusion and deterministic ties are resolved."""
    n_traj = y.shape[0] - trajectory_len + 1
    tree = KDTree(y[:n_traj], leafsize=16)
    remaining = np.arange(n_traj, dtype=np.int64)
    neighbours = np.full(n_traj, -1, dtype=np.int64)
    k = min(8, n_traj)
    while len(remaining):
        unresolved = []
        # Bound query matrices to O(N + 65536), not O(N * k).
        batch_size = max(1, 65536 // k)
        for start in range(0, len(remaining), batch_size):
            rows = remaining[start : start + batch_size]
            distances, indices = tree.query(y[rows], k=k, eps=0, workers=1)
            selected, finished = _select_candidates(y, rows, indices, distances, min_tsep, n_traj, k == n_traj)
            neighbours[rows[finished]] = selected[finished]
            unresolved.append(rows[~finished])
        remaining = np.concatenate(unresolved)
        k = min(2 * k, n_traj)
    return neighbours


SEARCHES = {
    "scalar": neighbours_scalar,
    "parallel": neighbours_parallel,
    "kdtree": neighbours_kdtree,
}


def prepare_embedding(data, tau=1, m=2, dt=1.0, min_tsep=None, trajectory_len=20, fit_offset=0):
    """Include current public validation so end-to-end timings are comparable."""
    from .benchmark import lyap

    x = lyap.as_finite_1d(data, "data")
    if len(x) < 2:
        raise ValueError(f"data must contain at least 2 samples, got len(data)={len(x)}")
    tau = lyap.check_positive_int(tau, "tau")
    m = lyap.check_positive_int(m, "m")
    trajectory_len = lyap.check_positive_int(trajectory_len, "trajectory_len", minimum=2)
    fit_offset = lyap.check_positive_int(fit_offset, "fit_offset", minimum=0)
    if fit_offset > trajectory_len - 2:
        raise ValueError(f"fit_offset must be <= {trajectory_len - 2}, got {fit_offset}")
    if (
        isinstance(dt, (bool, np.bool_))
        or not isinstance(dt, (int, float, np.integer, np.floating))
        or not np.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError(f"dt must be finite and > 0, got {dt!r}")
    if min_tsep is None:
        min_tsep = min(lyap._mean_period(x), len(x) // 4)
    else:
        min_tsep = lyap.check_positive_int(min_tsep, "min_tsep", minimum=0)
    n_vectors = len(x) - (m - 1) * tau
    n_traj = n_vectors - trajectory_len + 1
    if n_traj < 2 * min_tsep + 2:
        raise ValueError(f"series too short: {n_traj} trajectories for min_tsep={min_tsep}")
    y = np.stack([x[i * tau : i * tau + n_vectors] for i in range(m)], axis=1)
    return y, min_tsep, trajectory_len


def fit_exponent(log_div, dt=1.0, fit_offset=0):
    """Use the package's existing finite-mask, fit and sampling-time arithmetic."""
    ks = np.arange(len(log_div))[fit_offset:]
    log_div = log_div[fit_offset:]
    finite = np.isfinite(log_div)
    if finite.sum() < 2:
        return -np.inf
    return float(np.polyfit(ks[finite], log_div[finite], 1)[0] / dt)


def estimate(data, method, **parameters):
    """Benchmark only the replacement kernel with original surrounding arithmetic."""
    y, window, length = prepare_embedding(data, **parameters)
    neighbours = SEARCHES[method](y, window, length)
    divergence = follow_divergence(y, neighbours, length)
    return fit_exponent(divergence, parameters.get("dt", 1.0), parameters.get("fit_offset", 0))
