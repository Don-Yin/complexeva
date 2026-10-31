"""Baseline-centred fuzzy entropy with an exponential membership function."""

import numpy
from numba import get_num_threads, njit, prange
from numpy.typing import ArrayLike, NDArray

from ._validation import as_signals, check_finite_real, check_positive_int, float64_rows, unit_stride_rows

_PARALLEL_MIN_TEMPLATES = 4096  # Amortize the parallel scheduler on a single long signal.
_BATCH_COORDINATE_BUDGET = 60  # Bound concurrent template storage to 60 * N float64 values (12 default-order workers).


@njit(cache=True)
def _templates(x, order):
    """Centre each template separately, using the same extendible starts at both lengths."""
    count = len(x) - order
    short = numpy.empty((count, order))
    long = numpy.empty((count, order + 1))
    for i in range(count):
        mean = 0.0
        for k in range(order):
            mean += x[i + k]
        next_mean = (mean + x[i + order]) / (order + 1)
        mean /= order
        for k in range(order):
            short[i, k] = x[i + k] - mean
            long[i, k] = x[i + k] - next_mean
        long[i, order] = x[i + order] - next_mean
    return short, long


@njit(cache=True, inline="always")
def _membership(distance, r, exponent):
    """Keep the divisor outside the power, including the common square case."""
    if exponent == 2.0:
        power = distance * distance
    elif exponent == 1.0:
        power = distance
    else:
        power = distance**exponent
    return numpy.exp(-power / r)


@njit(cache=True)
def _anchor_sums(short, long, i, r, exponent):
    """Accumulate one upper-triangle row without a distance or membership matrix."""
    b, a = 0.0, 0.0
    order = short.shape[1]
    for j in range(i + 1, len(short)):
        dm, dm1 = 0.0, 0.0
        for k in range(order):
            dm = max(dm, abs(short[i, k] - short[j, k]))
            dm1 = max(dm1, abs(long[i, k] - long[j, k]))
        dm1 = max(dm1, abs(long[i, order] - long[j, order]))
        b += _membership(dm, r, exponent)
        a += _membership(dm1, r, exponent)
    return b, a


@njit(cache=True, inline="always")
def _finish(b, a):
    """Take the log ratio after cancelling the identical pair-count denominators."""
    if b == 0.0:
        return numpy.nan
    if a == 0.0:
        return numpy.inf
    return -numpy.log(a / b)


@njit(cache=True)
def _entropy_serial(x, order, r, exponent):
    """Use the same anchor grouping as the parallel single-signal kernel."""
    if r == 0.0:
        return numpy.nan
    short, long = _templates(x, order)
    b, a = 0.0, 0.0
    for i in range(len(short) - 1):
        bi, ai = _anchor_sums(short, long, i, r, exponent)
        b += bi
        a += ai
    return _finish(b, a)


@njit(cache=True, parallel=True)
def _entropy_parallel(x, order, r, exponent):
    """Parallelize anchor rows, then reduce in fixed order for reproducible bits."""
    if r == 0.0:
        return numpy.nan
    short, long = _templates(x, order)
    sums = numpy.empty((len(short) - 1, 2))
    # Pair long and short triangle rows so each scheduled iteration does equal work.
    for i in prange((len(sums) + 1) // 2):
        sums[i, 0], sums[i, 1] = _anchor_sums(short, long, i, r, exponent)
        j = len(sums) - 1 - i
        if j != i:
            sums[j, 0], sums[j, 1] = _anchor_sums(short, long, j, r, exponent)
    b, a = 0.0, 0.0
    for i in range(len(sums)):
        b += sums[i, 0]
        a += sums[i, 1]
    return _finish(b, a)


@njit(cache=True, parallel=True)
def _entropy_rows(values, direct, order, radii, exponent, chunks):
    """Read batch views in place, or convert one row per worker, never the whole batch."""
    n = values.shape[-1]
    out = numpy.empty(len(radii))
    buffers = numpy.empty((chunks if direct is None else 0, n))
    for chunk in prange(chunks):
        for i in range(len(radii) * chunk // chunks, len(radii) * (chunk + 1) // chunks):
            if direct is None:
                row = buffers[chunk]
                for j in range(n):
                    row[j] = values.flat[i * n + j]
            else:
                row = direct[i]
            out[i] = _entropy_serial(row, order, radii[i], exponent)
    return out


def _default_radius(row):
    """Keep exactly constant data at zero width, even when std has summation roundoff."""
    if numpy.min(row) == numpy.max(row):
        return 0.0
    return check_finite_real(float(0.2 * numpy.std(row, ddof=0)), "r", allow_zero=True)


def fuzzy_entropy(
    x: ArrayLike,
    order: int | numpy.integer = 2,
    r: float | numpy.floating | numpy.integer | None = None,
    n: float | numpy.floating | numpy.integer = 2,
) -> float | NDArray[numpy.float64]:
    """Fuzzy entropy with exponential membership exp(-distance**n / r), in nats.

    Args:
        x: Finite, real-valued samples with time on the last axis. Each signal requires at
            least order + 2 samples. Non-contiguous batches are converted one row at a time.
        order: Template length before extension by one sample, a positive integer. Defaults to 2.
        r: Nonnegative exponential divisor, not a distance threshold raised to n. None uses
            0.2 * std(row, ddof=0) independently for each signal. Zero gives NaN.
        n: Positive finite exponent of the Chebyshev distance. Defaults to 2.

    Returns:
        -log(A / B), where A and B are sums of memberships for baseline-centred templates of
        lengths order + 1 and order, respectively. Returns NaN when B is zero (including the
        default zero divisor of a constant signal), or infinity when only A is zero. A constant
        signal with positive r has entropy 0. A float for 1D input; a float64 array of shape
        x.shape[:-1] for a batch, bitwise equal to calling the function on every row separately.

    Raises:
        ValueError: If x has no time axis, is not finite and real, is too short, order is not a
            positive integer, r is negative or non-finite, or n is not positive and finite.
            Invalid samples are reported with their full index.

    Note:
        Each template has its own mean subtracted before measuring Chebyshev distance.
        Self-matches are excluded and both lengths use the same len(x) - order extendible
        templates. The membership divisor is r, not r**n. This is the Chen et al. (2007),
        IEEE TNSRE 15:266-272 definition (doi:10.1109/TNSRE.2007.897025), matching
        EntropyHub.FuzzEn(..., m=order, tau=1, r=(r, n))[0][-1]. pyentrp uses
        exp(-(distance / tolerance)**n), so pass tolerance=r**(1/n) to match.
        EntropyHub defaults to r=0.2 regardless of signal scale. neurokit2 0.2.13 always uses
        exponent 1 (its public n keyword is not forwarded), so it agrees only with n=1;
        pass tolerance=r to override its default sample-SD scaling. The paper does not
        prescribe a universal default width; its EMG example uses 0.3 * SD (section III.B).
        Computation is exact pair enumeration, O(N**2 * order) time and O(N * order) heap
        per active worker. Long single signals parallelize anchor rows. Batches parallelize
        signals when enough rows are available, otherwise reuse the anchor-parallel kernel
        for long rows (exp_033). All schedules use deterministic reductions and no fastmath.
        NUMBA_NUM_THREADS bounds
        the worker count. Batch template and conversion buffers share a 60-coordinate
        budget (12 default-order workers for direct float64 rows, and at least one worker
        for arbitrary order). No membership cutoff or sampled pairs are used.
    """
    values = as_signals(x)
    order = check_positive_int(order, "order")
    exponent = check_finite_real(n, "n", allow_zero=False)
    length = values.shape[-1]
    if length < order + 2:
        raise ValueError(f"x needs at least order + 2 = {order + 2} samples, got {length}")
    if r is not None:
        r = check_finite_real(r, "r", allow_zero=True)
    if values.ndim == 1:
        if r is None:
            r = _default_radius(values)
        kernel = _entropy_parallel if length - order >= _PARALLEL_MIN_TEMPLATES else _entropy_serial
        return float(kernel(values, order, r, exponent))
    rows = values.size // length
    if r is None:
        radii = numpy.fromiter(map(_default_radius, float64_rows(values)), numpy.float64)
    else:
        radii = numpy.full(rows, r)
    direct = unit_stride_rows(values)
    coordinates = 2 * order + 1 + (direct is None)
    capacity = min(get_num_threads(), max(1, _BATCH_COORDINATE_BUDGET // coordinates))
    if rows < capacity and length - order >= _PARALLEL_MIN_TEMPLATES:
        out = numpy.empty(rows)
        for i, row in enumerate(float64_rows(values)):
            out[i] = _entropy_parallel(row, order, radii[i], exponent)
    else:
        out = _entropy_rows(values, direct, order, radii, exponent, min(rows, capacity))
    return out.reshape(values.shape[:-1])
