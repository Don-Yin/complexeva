import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._validation import as_finite_1d, check_positive_int


@njit(cache=True)
def _divergence(Y: NDArray[np.float64], min_tsep: int, trajectory_len: int) -> NDArray[np.float64]:
    """Compute the Rosenstein mean log distance at each trajectory step.

    Args:
        Y: Delay-embedded orbit, with shape (number of vectors, embedding dimension).
        min_tsep: Nonnegative Theiler window in samples; pairs at this distance or closer are excluded.
        trajectory_len: Number of future steps followed for each nearest-neighbour pair.

    Returns:
        Float64 array of length trajectory_len containing the mean natural-log distance at each step.
        A step with no positive distances is represented by negative infinity.
    """
    n_traj = Y.shape[0] - trajectory_len + 1
    neighbour = np.empty(n_traj, dtype=np.int64)
    for i in range(n_traj):
        best, best_j = np.inf, -1
        for j in range(n_traj):
            if abs(i - j) <= min_tsep:  # Theiler window: skip temporally close points
                continue
            d = 0.0
            for dimension in range(Y.shape[1]):
                delta = Y[i, dimension] - Y[j, dimension]
                d += delta * delta
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


def _mean_period(x: NDArray[np.float64]) -> int:
    """Estimate the period used for the default Theiler window.

    Args:
        x: Nonconstant, finite, one-dimensional float64 signal with at least two samples.

    Returns:
        Reciprocal of the power-weighted mean positive FFT frequency, rounded up to whole samples.

    Raises:
        ValueError: If the signal has zero spectral power, so a default window is undefined.
    """
    psd = np.abs(np.fft.rfft(x - x.mean())) ** 2
    freqs = np.fft.rfftfreq(len(x))
    power = np.sum(psd[1:])
    if power == 0:
        raise ValueError(f"min_tsep cannot be inferred from constant data with value {x[0]!r}; set min_tsep explicitly")
    return int(np.ceil(power / np.sum(freqs[1:] * psd[1:])))


def lyapunov_exponent(
    data: ArrayLike,
    tau: int | np.integer = 1,
    m: int | np.integer = 2,
    dt: float = 1.0,
    min_tsep: int | np.integer | None = None,
    trajectory_len: int | np.integer = 20,
    fit_offset: int | np.integer = 0,
) -> float:
    """Estimate the largest Lyapunov exponent by Rosenstein et al. (1993).

    The series is delay-embedded, each point is paired with its nearest neighbour outside a Theiler window,
    and the exponent is the slope of the mean log divergence of those pairs over trajectory_len steps.

    Args:
        data: One-dimensional, finite, real-valued samples, converted to float64. The resulting number
            of trajectories, len(data) - (m - 1) * tau - trajectory_len + 1, must be at least
            2 * min_tsep + 2.
        tau: Positive integer embedding delay in samples. Defaults to 1.
        m: Positive integer embedding dimension. Defaults to 2.
        dt: Finite, strictly positive sampling interval. Defaults to 1.0; the result has units of 1/dt.
        min_tsep: Nonnegative integer Theiler window in samples. Pairs at this separation or closer
            are excluded. None estimates the ceil of the reciprocal power-weighted mean frequency,
            capped at len(data) // 4. Constant data requires an explicit value.
        trajectory_len: Integer number of future steps followed per pair, at least 2. Defaults to 20.
        fit_offset: Nonnegative integer number of initial steps excluded from the line fit. Must be
            at most trajectory_len - 2 so that two steps remain. Defaults to 0.

    Returns:
        Largest Lyapunov exponent per unit time. Positive values suggest chaos; values near zero
        or below indicate regular dynamics. Returns negative infinity when fewer than two fitted
        steps have finite mean log distances, including constant data with an explicit min_tsep.

    Raises:
        ValueError: If data is not a finite one-dimensional signal, a parameter is outside its
            stated range, the series is too short, or a default window cannot be inferred.
    """
    x = as_finite_1d(data, "data")
    if len(x) < 2:
        raise ValueError(f"data must contain at least 2 samples, got len(data)={len(x)}")
    tau = check_positive_int(tau, "tau")
    m = check_positive_int(m, "m")
    trajectory_len = check_positive_int(trajectory_len, "trajectory_len", minimum=2)
    fit_offset = check_positive_int(fit_offset, "fit_offset", minimum=0)
    if fit_offset > trajectory_len - 2:
        raise ValueError(f"fit_offset must be <= trajectory_len - 2 = {trajectory_len - 2}, got {fit_offset}")
    if (
        isinstance(dt, (bool, np.bool_))
        or not isinstance(dt, (int, float, np.integer, np.floating))
        or not np.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError(f"dt must be finite and > 0, got {dt!r}")
    if min_tsep is None:
        min_tsep = min(_mean_period(x), len(x) // 4)
    else:
        min_tsep = check_positive_int(min_tsep, "min_tsep", minimum=0)
    n_vectors = len(x) - (m - 1) * tau
    n_traj = n_vectors - trajectory_len + 1
    if n_traj < 2 * min_tsep + 2:
        raise ValueError(
            f"series too short: len(data)={len(x)} gives {n_traj} trajectories for m={m}, tau={tau}, "
            f"trajectory_len={trajectory_len}, min_tsep={min_tsep}; require at least {2 * min_tsep + 2}"
        )
    Y = np.stack([x[i * tau : i * tau + n_vectors] for i in range(m)], axis=1)
    log_div = _divergence(Y, min_tsep, trajectory_len)
    ks = np.arange(trajectory_len)[fit_offset:]
    log_div = log_div[fit_offset:]
    finite = np.isfinite(log_div)
    if finite.sum() < 2:
        return -np.inf
    return float(np.polyfit(ks[finite], log_div[finite], 1)[0] / dt)
