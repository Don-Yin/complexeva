import numpy as np
from numba import njit


@njit(cache=True)
def _divergence(Y, min_tsep, trajectory_len):
    """mean log distance between each orbit point and its nearest neighbour after k steps (Rosenstein 1993)"""
    n_traj = Y.shape[0] - trajectory_len + 1
    neighbour = np.empty(n_traj, dtype=np.int64)
    for i in range(n_traj):
        best, best_j = np.inf, -1
        for j in range(n_traj):
            if abs(i - j) <= min_tsep:  # Theiler window: skip temporally close points
                continue
            d = np.sum((Y[i] - Y[j]) ** 2)
            if d < best:
                best, best_j = d, j
        neighbour[i] = best_j
    log_div = np.full(trajectory_len, -np.inf)
    for k in range(trajectory_len):
        total, count = 0.0, 0
        for i in range(n_traj):
            d = np.sqrt(np.sum((Y[i + k] - Y[neighbour[i] + k]) ** 2))
            if d > 0:
                total += np.log(d)
                count += 1
        if count > 0:
            log_div[k] = total / count
    return log_div


def _mean_period(x):
    """1 / power-weighted mean frequency, the default Theiler window"""
    psd = np.abs(np.fft.rfft(x - x.mean())) ** 2
    freqs = np.fft.rfftfreq(len(x))
    return int(np.ceil(np.sum(psd[1:]) / np.sum(freqs[1:] * psd[1:])))


def lyapunov_exponent(data, tau=1, m=2, dt=1.0, min_tsep=None, trajectory_len=20, fit_offset=0):
    """
    Largest Lyapunov exponent by the Rosenstein et al. (1993) method.

    The series is delay-embedded, each point is paired with its nearest neighbour outside a Theiler window,
    and the exponent is the slope of the mean log divergence of those pairs over `trajectory_len` steps.

    Args:
        data: 1D array-like time series
        tau: embedding delay in samples
        m: embedding dimension
        dt: sampling interval; the result is in units of 1/dt
        min_tsep: Theiler window in samples; defaults to the mean period of the signal
        trajectory_len: number of steps over which divergence is followed and fitted
        fit_offset: number of initial steps excluded from the line fit

    Returns:
        float: largest Lyapunov exponent (positive suggests chaos, about 0 or below for regular signals)
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"data must be 1D, got shape {x.shape}")
    if min_tsep is None:
        min_tsep = min(_mean_period(x), len(x) // 4)
    n_vectors = len(x) - (m - 1) * tau
    n_traj = n_vectors - trajectory_len + 1
    if n_traj < 2 * min_tsep + 2:
        raise ValueError(f"series too short: {n_traj} trajectories for min_tsep={min_tsep}")
    Y = np.stack([x[i * tau : i * tau + n_vectors] for i in range(m)], axis=1)
    log_div = _divergence(Y, min_tsep, trajectory_len)
    ks = np.arange(trajectory_len)[fit_offset:]
    log_div = log_div[fit_offset:]
    finite = np.isfinite(log_div)
    if finite.sum() < 2:
        return -np.inf
    return float(np.polyfit(ks[finite], log_div[finite], 1)[0] / dt)
