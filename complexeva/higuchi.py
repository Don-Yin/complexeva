import numpy
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._hilbert import hilbert_envelope
from ._validation import as_signals, check_positive_int, unit_stride_rows


@njit(cache=True, fastmath={"reassoc"})
def _curve_length(x: NDArray[numpy.float64], m: int, k: int, n_max: int) -> float:
    """Sum nonnegative increments with reassociation, but without unsafe fast-math flags."""
    total = 0.0
    for i in range(1, n_max + 1):
        total += numpy.abs(x[m + i * k] - x[m + (i - 1) * k])
    return total


@njit(cache=True)
def _hfd_core(x: NDArray[numpy.float64], k_max: int, lengths: NDArray[numpy.float64] | None = None) -> float:
    """Compute the same centered regression with one reusable length array."""
    N = len(x)
    L = numpy.empty(k_max) if lengths is None else lengths
    sum_x = 0.0
    sum_L = 0.0
    for k in range(1, k_max + 1):
        Lk = 0.0
        for m in range(k):
            # Higuchi (1988): M = floor((N - m) / k) increments with 1-based m, i.e. (N - 1 - m) // k here
            n_max = (N - 1 - m) // k
            Lmk = _curve_length(x, m, k, n_max)
            Lmk *= (N - 1) / (n_max * k)  # normalisation factor
            Lk += Lmk / k  # curve length L_m(k); without /k the slope is D - 1
        L[k - 1] = numpy.log(Lk / k)
        sum_L += L[k - 1]
        sum_x += numpy.log(1.0 / k)
    mean_x = sum_x / k_max
    mean_L = sum_L / k_max
    numerator = 0.0
    denominator = 0.0
    for k in range(1, k_max + 1):
        dx = numpy.log(1.0 / k) - mean_x
        numerator += dx * (L[k - 1] - mean_L)
        denominator += dx * dx
    return numerator / denominator


@njit(cache=True)
def _hfd_rows(x: NDArray, direct: NDArray[numpy.float64] | None, k_max: int):
    """Read contiguous rows in place, or reuse one float64 conversion buffer."""
    n = x.shape[-1]
    rows = x.size // n
    out = numpy.empty(rows)
    lengths = numpy.empty(k_max)
    buffer = numpy.empty(n) if direct is None else None
    for i in range(rows):
        if direct is None:
            assert buffer is not None
            row = buffer
            for j in range(n):
                row[j] = x.flat[i * n + j]
        else:
            row = numpy.ascontiguousarray(direct[i])
        out[i] = _hfd_core(row, k_max, lengths)
    return out


def hfd(x: ArrayLike, k_max: int | numpy.integer = 10) -> float | NDArray[numpy.float64]:
    """Estimate the Higuchi fractal dimension of one signal, or of every signal along the last axis.

    Args:
        x: Finite, real-valued samples, converted to float64, with time on the last axis: one signal
            of shape (time,) or a batch such as (channels, time). Requires at least 2 * k_max samples.
            Batch rows share one reusable workspace instead of allocating one per worker.
        k_max: Largest integer time interval included in the regression. Must be between 2 and
            len(x) // 2, inclusive. Defaults to 10; report the chosen value with results.

    Returns:
        Slope of log curve length against log inverse interval, approximately 1 for smooth signals,
        1.5 for Brownian motion, and 2 for white noise. The dimension of a constant signal is
        undefined and is returned as NaN. A float for one signal; for a batch, a float64 array of
        shape x.shape[:-1] whose entries equal hfd on each signal.

    Raises:
        ValueError: If x is a scalar, not finite and real, or k_max is not an integer in the stated
            range. The error identifies the invalid value, its position, or the shape.
    """
    values = as_signals(x, "x")
    n = values.shape[-1]
    k_max = check_positive_int(k_max, "k_max", minimum=2)
    if k_max > n // 2:
        raise ValueError(f"k_max must be between 2 and x.shape[-1] // 2 = {n // 2}, got {k_max}")
    if values.ndim == 1:
        return float(_hfd_core(values, k_max))
    return _hfd_rows(values, unit_stride_rows(values), k_max).reshape(values.shape[:-1])


def hfd_hilbert(x: ArrayLike, k_max: int | numpy.integer = 10) -> float:
    """Estimate Higuchi dimension on the Hilbert envelope of a signal.

    Args:
        x: Nonempty, one-dimensional, finite, real-valued samples, converted to float64 before the
            Hilbert transform. Requires at least 2 * k_max samples.
        k_max: Largest integer time interval, between 2 and len(x) // 2 inclusive. Defaults to 10.

    Returns:
        Higuchi fractal dimension of abs(hilbert(x)), the instantaneous amplitude. A constant
        envelope has undefined dimension and produces NaN, as in hfd.

    Raises:
        ValueError: If x is empty or not a finite, real, one-dimensional signal, or k_max is outside
            the integer range accepted by hfd.
    """
    # The envelope is one signal, so hfd returns a float.
    return float(hfd(hilbert_envelope(x, "x"), k_max=k_max))
