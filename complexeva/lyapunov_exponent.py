import numpy
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._validation import as_finite_1d, check_finite_real, check_positive_int


@njit(cache=True)
def _nearest_neighbours(Y: NDArray[numpy.float64], min_tsep: int, n_traj: int) -> NDArray[numpy.int64]:
    """Find the exact nearest neighbour of each of the first n_traj vectors outside the Theiler window.

    Vectors are sorted by their first coordinate and scanned outward from each query. Consecutive sorted
    ranks form blocks of about sqrt(n_traj); a block is skipped whole when all its time indices lie inside
    the window or its bounding box is farther than the best distance, so a window that covers many close
    sorted neighbours, as in a trending series, costs one test per block instead of one per point. The
    sorted copy of the vectors makes the scan contiguous in memory.

    Args:
        Y: Delay-embedded orbit, with shape (number of vectors, embedding dimension).
        min_tsep: Nonnegative Theiler window in samples; pairs at this distance or closer are excluded.
        n_traj: Number of leading vectors that are both queries and candidates.

    Returns:
        Int64 array of neighbour indices; distance ties resolve to the lowest index.
    """
    m = Y.shape[1]
    shift = (int(numpy.log2(n_traj)) + 1) // 2
    size = 1 << shift
    n_blocks = (n_traj + size - 1) >> shift
    order = numpy.argsort(Y[:n_traj, 0])
    points = Y[:n_traj][order]
    first_index = numpy.full(n_blocks, n_traj)
    last_index = numpy.full(n_blocks, -1)
    box_low = numpy.full((n_blocks, m), numpy.inf)
    box_high = numpy.full((n_blocks, m), -numpy.inf)
    for rank in range(n_traj):
        block = rank >> shift
        first_index[block] = min(first_index[block], order[rank])
        last_index[block] = max(last_index[block], order[rank])
        for dimension in range(m):
            box_low[block, dimension] = min(box_low[block, dimension], points[rank, dimension])
            box_high[block, dimension] = max(box_high[block, dimension], points[rank, dimension])
    neighbour = numpy.empty(n_traj, dtype=numpy.int64)
    for rank in range(n_traj):
        i = order[rank]
        # The first eligible index supplies a bound and resolves all-zero ties.
        first = 0 if i > min_tsep else i + min_tsep + 1
        best = 0.0
        for dimension in range(m):
            delta = Y[i, dimension] - Y[first, dimension]
            best += delta * delta
        best_j = first if best < numpy.inf else -1
        if best > 0:
            for direction in (-1, 1):
                position = rank + direction
                edge = 0 if direction == 1 else size - 1
                scanning = True
                while scanning and 0 <= position < n_traj:
                    block = position >> shift
                    if (position & (size - 1)) == edge:
                        skip = first_index[block] >= i - min_tsep and last_index[block] <= i + min_tsep
                        if not skip:
                            # Box distance is computed like d, so it never exceeds the distance of a member.
                            bound = 0.0
                            for dimension in range(m):
                                gap = max(
                                    box_low[block, dimension] - points[rank, dimension],
                                    points[rank, dimension] - box_high[block, dimension],
                                    0.0,
                                )
                                bound += gap * gap
                            skip = bound > best
                        if skip:
                            position += direction * size
                            continue
                    stop = min((block + 1) << shift, n_traj) if direction == 1 else (block << shift) - 1
                    while position != stop:
                        j = order[position]
                        delta = points[rank, 0] - points[position, 0]
                        # Every farther point has at least this squared distance.
                        if delta * delta > best:
                            scanning = False
                            break
                        if abs(i - j) > min_tsep:
                            d = 0.0
                            for dimension in range(m):
                                delta = points[rank, dimension] - points[position, dimension]
                                d += delta * delta
                            if d < best or (d == best and j < best_j):
                                best, best_j = d, j
                        position += direction
        neighbour[i] = best_j
    return neighbour


@njit(cache=True)
def _divergence(Y: NDArray[numpy.float64], min_tsep: int, trajectory_len: int) -> NDArray[numpy.float64]:
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
    neighbour = _nearest_neighbours(Y, min_tsep, n_traj)
    log_div = numpy.full(trajectory_len, -numpy.inf)
    for k in range(trajectory_len):
        total, count = 0.0, 0
        for i in range(n_traj):
            d = numpy.sqrt(numpy.sum((Y[i + k] - Y[neighbour[i] + k]) ** 2))
            if d > 0:
                total += numpy.log(d)
                count += 1
        if count > 0:
            log_div[k] = total / count
    return log_div


def _mean_period(x: NDArray[numpy.float64]) -> int:
    """Estimate the period used for the default Theiler window.

    Args:
        x: Nonconstant, finite, one-dimensional float64 signal with at least two samples.

    Returns:
        Reciprocal of the power-weighted mean positive FFT frequency, rounded up to whole samples.

    Raises:
        ValueError: If the signal has zero spectral power, so a default window is undefined.
    """
    psd = numpy.abs(numpy.fft.rfft(x - x.mean())) ** 2
    freqs = numpy.fft.rfftfreq(len(x))
    power = numpy.sum(psd[1:])
    if power == 0:
        raise ValueError(f"min_tsep cannot be inferred from constant data with value {x[0]!r}; set min_tsep explicitly")
    return int(numpy.ceil(power / numpy.sum(freqs[1:] * psd[1:])))


def lyapunov_exponent(
    x: ArrayLike,
    tau: int | numpy.integer = 1,
    m: int | numpy.integer = 2,
    dt: float = 1.0,
    min_tsep: int | numpy.integer | None = None,
    trajectory_len: int | numpy.integer = 20,
    fit_offset: int | numpy.integer = 0,
) -> float:
    """Estimate the largest Lyapunov exponent by Rosenstein et al. (1993).

    The series is delay-embedded, each point is paired with its nearest neighbour outside a Theiler window,
    and the exponent is the slope of the mean log divergence of those pairs over trajectory_len steps.

    Args:
        x: One-dimensional, finite, real-valued samples, converted to float64. The resulting number
            of trajectories, len(x) - (m - 1) * tau - trajectory_len + 1, must be at least
            2 * min_tsep + 2.
        tau: Positive integer embedding delay in samples. Defaults to 1.
        m: Positive integer embedding dimension. Defaults to 2.
        dt: Finite, strictly positive sampling interval. Defaults to 1.0; the result has units of 1/dt.
        min_tsep: Nonnegative integer Theiler window in samples. Pairs at this separation or closer
            are excluded. None estimates the ceil of the reciprocal power-weighted mean frequency,
            capped at len(x) // 4. Constant data requires an explicit value.
        trajectory_len: Integer number of future steps followed per pair, at least 2. Defaults to 20.
        fit_offset: Nonnegative integer number of initial steps excluded from the line fit. Must be
            at most trajectory_len - 2 so that two steps remain. Defaults to 0.

    Returns:
        Largest Lyapunov exponent per unit time. Positive values suggest chaos; values near zero
        or below indicate regular dynamics. Returns negative infinity when fewer than two fitted
        steps have finite mean log distances, including constant data with an explicit min_tsep.

    Raises:
        ValueError: If x is not a finite one-dimensional signal, a parameter is outside its
            stated range, the series is too short, or a default window cannot be inferred.
    """
    x = as_finite_1d(x, "x")
    if len(x) < 2:
        raise ValueError(f"x must contain at least 2 samples, got len(x)={len(x)}")
    tau = check_positive_int(tau, "tau")
    m = check_positive_int(m, "m")
    trajectory_len = check_positive_int(trajectory_len, "trajectory_len", minimum=2)
    fit_offset = check_positive_int(fit_offset, "fit_offset", minimum=0)
    if fit_offset > trajectory_len - 2:
        raise ValueError(f"fit_offset must be <= trajectory_len - 2 = {trajectory_len - 2}, got {fit_offset}")
    dt = check_finite_real(dt, "dt", allow_zero=False)
    if min_tsep is None:
        min_tsep = min(_mean_period(x), len(x) // 4)
    else:
        min_tsep = check_positive_int(min_tsep, "min_tsep", minimum=0)
    n_vectors = len(x) - (m - 1) * tau
    n_traj = n_vectors - trajectory_len + 1
    if n_traj < 2 * min_tsep + 2:
        raise ValueError(
            f"series too short: len(x)={len(x)} gives {n_traj} trajectories for m={m}, tau={tau}, "
            f"trajectory_len={trajectory_len}, min_tsep={min_tsep}; require at least {2 * min_tsep + 2}"
        )
    Y = numpy.stack([x[i * tau : i * tau + n_vectors] for i in range(m)], axis=1)
    log_div = _divergence(Y, min_tsep, trajectory_len)
    ks = numpy.arange(trajectory_len)[fit_offset:]
    log_div = log_div[fit_offset:]
    finite = numpy.isfinite(log_div)
    if finite.sum() < 2:
        return -numpy.inf
    return float(numpy.polyfit(ks[finite], log_div[finite], 1)[0] / dt)
