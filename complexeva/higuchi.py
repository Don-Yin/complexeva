import numpy
from numba import njit
from scipy.signal import hilbert


@njit
def compute_L_x(X, k_max):
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


def hfd(X, k_max=10):
    """Higuchi fractal dimension of a 1D numpy array."""
    assert X.ndim == 1, f"X must be a 1d numpy array, got shape {X.shape}"
    if not 2 <= k_max <= len(X) // 2:
        raise ValueError(f"k_max must be between 2 and len(X) // 2 = {len(X) // 2}, got {k_max}")
    x, L = compute_L_x(X, k_max)
    A = numpy.column_stack((x, numpy.ones_like(x)))
    beta, _, _, _ = numpy.linalg.lstsq(A, L, rcond=None)
    return beta[0]


def hfd_hilbert(X, k_max=10):
    """Higuchi fractal dimension of the Hilbert envelope (instantaneous amplitude) of X."""
    assert X.ndim == 1, f"X must be a 1d numpy array, got shape {X.shape}"
    return hfd(numpy.abs(numpy.asarray(hilbert(X))), k_max=k_max)
