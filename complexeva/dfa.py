import math

import numpy
from numba import get_num_threads, njit, prange
from numpy.typing import ArrayLike, NDArray

from ._validation import _first_nonfinite, as_array, as_finite_1d, as_kernel_readable, check_finite_rows

_RUN = 32  # squares summed plainly per run before the compensated combination
_SPLIT = 134217729.0  # 2**27 + 1: Veltkamp constant that splits a float64 into two 26-bit halves


@njit(cache=True)  # must stay non-recursive: Numba cannot reload a cached function that calls itself
def _fluctuations(values: NDArray, scales: NDArray[numpy.int64], offset: int = 0, n: int = 0) -> NDArray[numpy.float64]:
    """Stream the profile twice per window, retaining only O(scales) working storage (exp_029).

    Replay the same cumulative additions in the fit and residual passes to preserve every profile bit.
    Anchored fits, split slope products and compensated squares avoid catastrophic cancellation.
    A flat offset also reads irregular leading batch axes without materializing a row.
    """
    if n == 0:
        n = values.size
    # Convert before subtraction: native integer/bool batches must have float64 scalar semantics.
    origin = numpy.float64(values.flat[offset])
    mean = 0.0
    for j in range(n):
        mean += numpy.float64(values.flat[offset + j]) - origin
    mean /= n
    out = numpy.empty(len(scales))
    for i in range(len(scales)):
        size = scales[i]
        count = n // size
        half = (size - 1) / 2
        spread = size * (size * size - 1.0) / 12
        total = 0.0
        carry = 0.0
        profile = 0.0
        for k in range(count):
            start = offset + k * size
            before = profile
            anchor = before + ((numpy.float64(values.flat[start]) - origin) - mean)
            level = 0.0
            cross = 0.0
            for j in range(size):
                profile += (numpy.float64(values.flat[start + j]) - origin) - mean
                shifted = profile - anchor
                level += shifted
                cross += (j - half) * shifted
            level /= size
            slope = cross / spread
            split = _SPLIT * slope
            slope_hi = split - (split - slope)
            slope_lo = slope - slope_hi
            # Errors in level and slope reach the residual sum only at second order; the residuals
            # themselves are rounded once, and their squares are summed in runs combined with TwoSum.
            centre = anchor + level
            replay = before
            for first in range(0, size, _RUN):
                run = 0.0
                for j in range(first, min(first + _RUN, size)):
                    replay += (numpy.float64(values.flat[start + j]) - origin) - mean
                    residual = ((replay - centre) - slope_hi * (j - half)) - slope_lo * (j - half)
                    run += residual * residual
                new_total = total + run
                virtual = new_total - total
                carry += (total - (new_total - virtual)) + (run - virtual)
                total = new_total
        out[i] = numpy.sqrt((total + carry) / (count * size))
    return out


@njit(cache=True)
def _dfa_core(
    values: NDArray,
    scales: NDArray[numpy.int64],
    log_scales: NDArray[numpy.float64],
    offset: int = 0,
    n: int = 0,
) -> float:
    """One signal's streamed profile, stable residuals and centred two-pass slope."""
    fluctuations = _fluctuations(values, scales, offset, n)
    count = 0
    mean_x = 0.0
    mean_y = 0.0
    for i in range(len(scales)):
        if fluctuations[i] > 0:
            count += 1
            mean_x += log_scales[i]
            mean_y += math.log(fluctuations[i])
    if count < 2:
        return numpy.nan
    mean_x /= count
    mean_y /= count
    cross = 0.0
    spread = 0.0
    for i in range(len(scales)):
        if fluctuations[i] > 0:
            dx = log_scales[i] - mean_x
            cross += dx * (math.log(fluctuations[i]) - mean_y)
            spread += dx * dx
    return cross / spread


@njit(cache=True, parallel=True)
def _dfa_rows(
    values: NDArray,
    direct: NDArray,
    scales: NDArray[numpy.int64],
    log_scales: NDArray[numpy.float64],
    chunks: int,
    flat_rows: bool,
):
    """Read row views or flat logical offsets in place; never allocate a worker's row buffer."""
    n = values.shape[-1]
    rows = values.size // n
    out = numpy.empty(rows)
    bad = numpy.empty(rows, dtype=numpy.int64)
    for chunk in prange(chunks):
        for i in range(rows * chunk // chunks, rows * (chunk + 1) // chunks):
            if flat_rows:
                bad[i] = -1
                for j in range(n):
                    if not numpy.isfinite(values.flat[i * n + j]):
                        bad[i] = j
                        break
                if bad[i] < 0:
                    out[i] = _dfa_core(values, scales, log_scales, i * n, n)
            else:
                bad[i] = _first_nonfinite(direct[i])
                if bad[i] < 0:
                    out[i] = _dfa_core(direct[i], scales, log_scales)
    return out, bad


def dfa(x: ArrayLike, nvals: ArrayLike | None = None) -> float | NDArray[numpy.float64]:
    """First-order detrended fluctuation analysis along the last axis, with non-overlapping windows.

    Args:
        x: Finite, real samples with time on the last axis, converted to float64. One signal
            has shape (time,); batches have shape (..., time). At least 6 samples per signal
            are needed with explicit sizes; default sizes need at least 58. Batch signals run
            in parallel; set NUMBA_NUM_THREADS to limit threads.
        nvals: At least two distinct integer window sizes, each between 4 and x.shape[-1] - 1.
            None (default) uses floor(4 * 1.2**k) up to x.shape[-1] / 10, with duplicate sizes
            removed, as in antropy.detrended_fluctuation. Report the fitted window sizes with
            results.

    Returns:
        Slope of log RMS fluctuation versus log window size. White noise has an exponent near
        0.5 and Brownian motion near 1.5. Returns NaN when fewer than two scales have nonzero
        fluctuation, including constant input. This is a scaling exponent, not a fractal
        dimension. Returns a float for one signal, or a float64 array of shape x.shape[:-1]
        for a batch, bitwise equal to calling dfa on each signal separately.

    Raises:
        ValueError: If the input is scalar, not finite and real, is too short along the last
            axis, or the window sizes are invalid or insufficient for a slope fit. A non-finite
            error identifies the full index in a batch.

    Note:
        The demeaned series is cumulatively summed, then a least-squares linear
        trend is removed independently from each complete window. A trailing
        incomplete window is discarded at each scale. Squared residuals are
        averaged across all samples and windows before taking one square root.
        The log-log slope is an ordinary least-squares fit (not RANSAC).
        This is DFA-1 (Peng et al., 1994); it also matches nolds.dfa with the
        same nvals, overlap=False, order=1 and both fits set to "poly".
    """
    values = as_array(x)
    if values.ndim == 0:
        raise ValueError(f"x must have a time axis, got shape {values.shape}")
    values = as_finite_1d(values) if values.ndim == 1 else as_kernel_readable(values)
    n = values.shape[-1]
    if n < 6:
        raise ValueError(f"x needs at least 6 samples for two window sizes, got {n}")
    if nvals is None:
        max_power = math.floor(math.log(n / 40) / math.log(1.2))
        scales = numpy.array(sorted({int(4 * 1.2**k) for k in range(max_power + 1)}), dtype=numpy.int64)
    else:
        try:
            scales = numpy.asarray(nvals)
        except (TypeError, ValueError) as error:
            raise ValueError(f"nvals must be a 1D array of integers, got {nvals!r}") from error
    if scales.ndim != 1 or scales.dtype.kind not in "iu":
        raise ValueError(f"nvals must be a 1D array of integers, got {scales!r}")
    sizes = scales.tolist()
    if len(sizes) < 2 or len(set(sizes)) != len(sizes):
        raise ValueError(f"nvals must contain at least two distinct window sizes, got {scales!r} for len(x)={n}")
    if min(sizes) < 4 or max(sizes) >= n:
        raise ValueError(f"nvals must be between 4 and len(x) - 1 = {n - 1}, got {scales!r}")
    scales = scales.astype(numpy.int64)
    # Shifting the log coordinate leaves the slope unchanged; log1p avoids cancellation at adjacent sizes.
    smallest = scales.min()
    log_scales = numpy.log1p((scales - smallest) / smallest)
    if values.ndim == 1:
        return float(_dfa_core(values, scales, log_scales))
    rows = values.size // n
    flat_rows = values.ndim > 2 and not values.flags.c_contiguous
    if values.ndim == 2:
        direct = values
    else:
        direct = numpy.empty((0, 0)) if flat_rows else values.reshape(rows, n)
    out, bad = _dfa_rows(values, direct, scales, log_scales, min(rows, get_num_threads()), flat_rows)
    check_finite_rows(values, bad)
    return out.reshape(values.shape[:-1])
