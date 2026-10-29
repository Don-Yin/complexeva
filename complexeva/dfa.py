import math

import numpy
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._validation import as_finite_1d

_RUN = 32  # squares summed plainly per run before the compensated combination
_SPLIT = 134217729.0  # 2**27 + 1: Veltkamp constant that splits a float64 into two 26-bit halves


@njit(cache=True)  # must stay non-recursive: Numba cannot reload a cached function that calls itself
def _fluctuations(walk: NDArray[numpy.float64], scales: NDArray[numpy.int64]) -> NDArray[numpy.float64]:
    """Pooled RMS residual of per-window linear fits for every window size, in one compiled call.

    Sums are anchored at each window's first value and residuals are squared explicitly, so a large
    offset or steep trend cannot cancel catastrophically as it does in prefix-sum formulas. The product
    of the fitted slope and the (half-integer) time is formed exactly from a 26-bit split of the slope.
    """
    out = numpy.empty(len(scales))
    for i in range(len(scales)):
        size = scales[i]
        count = len(walk) // size
        half = (size - 1) / 2
        spread = size * (size * size - 1.0) / 12
        total = 0.0
        carry = 0.0
        for k in range(count):
            start = k * size
            anchor = walk[start]
            level = 0.0
            cross = 0.0
            for j in range(size):
                shifted = walk[start + j] - anchor
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
            for first in range(0, size, _RUN):
                run = 0.0
                for j in range(first, min(first + _RUN, size)):
                    residual = ((walk[start + j] - centre) - slope_hi * (j - half)) - slope_lo * (j - half)
                    run += residual * residual
                new_total = total + run
                virtual = new_total - total
                carry += (total - (new_total - virtual)) + (run - virtual)
                total = new_total
        out[i] = numpy.sqrt((total + carry) / (count * size))
    return out


def dfa(x: ArrayLike, nvals: ArrayLike | None = None) -> float:
    """First-order detrended fluctuation analysis with non-overlapping windows.

    Args:
        x: Finite, real 1D time series. At least 6 samples are needed with explicit window
            sizes; the default sizes need at least 58 samples.
        nvals: At least two distinct integer window sizes, each between 4 and len(x) - 1.
            None (default) uses floor(4 * 1.2**k) up to len(x) / 10, with duplicate sizes
            removed, as in antropy.detrended_fluctuation. Report the fitted window sizes with
            results.

    Returns:
        Slope of log RMS fluctuation versus log window size. White noise has an exponent near
        0.5 and Brownian motion near 1.5. Returns NaN when fewer than two scales have nonzero
        fluctuation, including constant input. This is a scaling exponent, not a fractal
        dimension.

    Raises:
        ValueError: If the input is not finite, real and 1D, is too short, or the window sizes
            are invalid or insufficient for a slope fit.

    Note:
        The demeaned series is cumulatively summed, then a least-squares linear
        trend is removed independently from each complete window. A trailing
        incomplete window is discarded at each scale. Squared residuals are
        averaged across all samples and windows before taking one square root.
        The log-log slope is an ordinary least-squares fit (not RANSAC).
        This is DFA-1 (Peng et al., 1994); it also matches nolds.dfa with the
        same nvals, overlap=False, order=1 and both fits set to "poly".
    """
    values = as_finite_1d(x)
    if len(values) < 6:
        raise ValueError(f"x needs at least 6 samples for two window sizes, got {len(values)}")
    if nvals is None:
        max_power = math.floor(math.log(len(values) / 40) / math.log(1.2))
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
        raise ValueError(f"nvals must contain at least two distinct window sizes, got {scales!r} for len(x)={len(values)}")
    if min(sizes) < 4 or max(sizes) >= len(values):
        raise ValueError(f"nvals must be between 4 and len(x) - 1 = {len(values) - 1}, got {scales!r}")
    # Subtracting an observed value first keeps constant signals exactly zero.
    walk = values - values[0]
    walk -= walk.mean()
    numpy.cumsum(walk, out=walk)
    fluctuations = _fluctuations(walk, scales.astype(numpy.int64))
    nonzero = fluctuations > 0
    if numpy.count_nonzero(nonzero) < 2:
        return float("nan")
    return float(numpy.polyfit(numpy.log(scales[nonzero]), numpy.log(fluctuations[nonzero]), 1)[0])
