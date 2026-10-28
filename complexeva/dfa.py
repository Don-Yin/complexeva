import numpy
from numpy.typing import ArrayLike

from ._validation import as_finite_1d


def dfa(x: ArrayLike, nvals: ArrayLike | None = None) -> float:
    """First-order detrended fluctuation analysis with non-overlapping windows.

    Parameters
    ----------
    x : array-like
        Finite, real 1D time series. At least 6 samples are needed with
        explicit window sizes; the default sizes need at least 58 samples.
    nvals : array-like of int or None, default None
        At least two distinct window sizes, each between 4 and ``len(x) - 1``.
        The default uses ``floor(4 * 1.2**k)`` up to ``len(x) / 10``, with
        duplicate sizes removed, as in ``antropy.detrended_fluctuation``.
        Report the fitted window sizes with results.

    Returns
    -------
    float
        Slope of log RMS fluctuation versus log window size. White noise has
        an exponent near 0.5 and Brownian motion near 1.5. Returns NaN when
        fewer than two scales have nonzero fluctuation, including constant
        input. This is a scaling exponent, not a fractal dimension.

    Raises
    ------
    ValueError
        If the input is not finite, real and 1D, is too short, or the window
        sizes are invalid or insufficient for a slope fit.

    Notes
    -----
    The demeaned series is cumulatively summed, then a least-squares linear
    trend is removed independently from each complete window. A trailing
    incomplete window is discarded at each scale. Squared residuals are
    averaged across all samples and windows before taking one square root.
    The log-log slope is an ordinary least-squares fit (not RANSAC).
    This is DFA-1 (Peng et al., 1994); it also matches ``nolds.dfa`` with the
    same nvals, ``overlap=False``, ``order=1`` and both fits set to ``"poly"``.
    """
    values = as_finite_1d(x)
    if len(values) < 6:
        raise ValueError(f"x needs at least 6 samples for two window sizes, got {len(values)}")
    if nvals is None:
        max_power = int(numpy.floor(numpy.log(len(values) / 40) / numpy.log(1.2)))
        scales = numpy.unique(numpy.floor(4 * 1.2 ** numpy.arange(max_power + 1)).astype(numpy.int64))
    else:
        try:
            scales = numpy.asarray(nvals)
        except (TypeError, ValueError) as error:
            raise ValueError(f"nvals must be a 1D array of integers, got {nvals!r}") from error
    if scales.ndim != 1 or scales.dtype.kind not in "iu":
        raise ValueError(f"nvals must be a 1D array of integers, got {scales!r}")
    if len(scales) < 2 or len(numpy.unique(scales)) != len(scales):
        raise ValueError(f"nvals must contain at least two distinct window sizes, got {scales!r} for len(x)={len(values)}")
    if numpy.any(scales < 4) or numpy.any(scales >= len(values)):
        raise ValueError(f"nvals must be between 4 and len(x) - 1 = {len(values) - 1}, got {scales!r}")
    # Subtracting an observed value first keeps constant signals exactly zero.
    centered = values - values[0]
    walk = numpy.cumsum(centered - centered.mean())
    fluctuations = numpy.empty(len(scales))
    for i, size in enumerate(scales):
        windows = walk[: len(walk) - len(walk) % size].reshape(-1, size)
        time = numpy.arange(size, dtype=numpy.float64) - (size - 1) / 2
        residuals = windows - windows.mean(axis=1, keepdims=True)
        slopes = (residuals @ time) / (time @ time)
        residuals -= slopes[:, None] * time
        fluctuations[i] = numpy.sqrt(numpy.mean(residuals**2))
    nonzero = fluctuations > 0
    if numpy.count_nonzero(nonzero) < 2:
        return float("nan")
    return float(numpy.polyfit(numpy.log(scales[nonzero]), numpy.log(fluctuations[nonzero]), 1)[0])
