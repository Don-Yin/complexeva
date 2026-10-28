import numpy
from numba import njit
from numpy.typing import ArrayLike, NDArray
from scipy.signal import hilbert

from ._validation import as_finite_1d, check_positive_int


@njit
def compute_L_x(X: NDArray[numpy.float64], k_max: int) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
    """Compute the log-coordinate pairs for the Higuchi regression.

    Args:
        X: Validated one-dimensional float64 signal with at least 2 * k_max samples.
        k_max: Largest integer interval, at least 2.

    Returns:
        Pair of float64 arrays of length k_max: log(1 / k) and log(mean_m L_m(k)), respectively.
    """
    N = len(X)
    L = numpy.zeros(k_max)
    x = numpy.zeros(k_max)
    for k in range(1, k_max + 1):
        Lk = numpy.zeros(k)
        for m in range(k):
            # Higuchi (1988): M = floor((N - m) / k) increments with 1-based m, i.e. (N - 1 - m) // k here
            Lmk = 0.0
            n_max = (N - 1 - m) // k
            for i in range(1, n_max + 1):
                Lmk += numpy.abs(X[m + i * k] - X[m + (i - 1) * k])
            Lmk *= (N - 1) / (n_max * k)  # normalisation factor
            Lk[m] = Lmk / k  # curve length L_m(k); without /k the slope is D - 1
        L[k - 1] = numpy.log(Lk.mean())
        x[k - 1] = numpy.log(1.0 / k)
    return x, L


@njit
def _hfd_core(X: NDArray[numpy.float64], k_max: int) -> float:
    """Least-squares slope of log L(k) on log(1/k), computed natively (faster than lstsq; exp_000_hfd_speed)."""
    x, L = compute_L_x(X, k_max)
    dx = x - x.mean()
    return numpy.sum(dx * (L - L.mean())) / numpy.sum(dx * dx)


def hfd(X: ArrayLike, k_max: int | numpy.integer = 10) -> float:
    """Estimate the Higuchi fractal dimension of a one-dimensional signal.

    Args:
        X: One-dimensional, finite, real-valued samples, converted to float64. Requires at least
            2 * k_max samples.
        k_max: Largest integer time interval included in the regression. Must be between 2 and
            len(X) // 2, inclusive. Defaults to 10; report the chosen value with results.

    Returns:
        Slope of log curve length against log inverse interval: approximately 1 for smooth signals,
        1.5 for Brownian motion, and 2 for white noise. The dimension of a constant signal is
        undefined and is returned as NaN.

    Raises:
        ValueError: If X is not a finite, real, one-dimensional signal, or k_max is not an integer
            in the stated range. The error identifies the invalid value or shape.
    """
    X = as_finite_1d(X, "X")
    k_max = check_positive_int(k_max, "k_max", minimum=2)
    if k_max > len(X) // 2:
        raise ValueError(f"k_max must be between 2 and len(X) // 2 = {len(X) // 2}, got {k_max}")
    return float(_hfd_core(X, k_max))


def hfd_hilbert(X: ArrayLike, k_max: int | numpy.integer = 10) -> float:
    """Estimate Higuchi dimension on the Hilbert envelope of a signal.

    Args:
        X: Nonempty, one-dimensional, finite, real-valued samples, converted to float64 before the
            Hilbert transform. Requires at least 2 * k_max samples.
        k_max: Largest integer time interval, between 2 and len(X) // 2 inclusive. Defaults to 10.

    Returns:
        Higuchi fractal dimension of abs(hilbert(X)), the instantaneous amplitude. A constant
        envelope has undefined dimension and produces NaN, as in hfd.

    Raises:
        ValueError: If X is empty or not a finite, real, one-dimensional signal, or k_max is outside
            the integer range accepted by hfd.
    """
    X = as_finite_1d(X, "X")
    if X.size == 0:
        raise ValueError("X must contain at least one sample for the Hilbert transform, got len(X)=0")
    return hfd(numpy.abs(numpy.asarray(hilbert(X))), k_max=k_max)
