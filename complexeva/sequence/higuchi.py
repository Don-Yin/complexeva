#!/Users/donyin/miniconda3/envs/rotation-1/bin/python


import numpy
from numba import njit
from scipy.signal import hilbert


@njit
def compute_L_x(X, k_max=16):
    k_max = min(k_max, len(X) // 2) if k_max is None else k_max
    N = len(X)
    L = numpy.zeros(k_max)
    x = numpy.zeros(k_max)
    for k in range(1, k_max + 1):
        Lk = numpy.zeros(k)
        for m in range(k):
            Lmk = 0.0
            n_max = (N - m) // k
            for i in range(1, n_max):
                Lmk += numpy.abs(X[m + i * k] - X[m + (i - 1) * k])
            Lmk *= (N - 1) / (n_max * k)
            Lk[m] = Lmk
        L[k - 1] = numpy.log(Lk.mean())
        x[k - 1] = numpy.log(1.0 / k)
    return x, L


def hfd(X, k_max=16):
    """expect a 1d numpy array as input"""
    assert X.ndim == 1, f"X must be a 1d numpy array, got shape {X.shape}"
    x, L = compute_L_x(X, k_max)
    A = numpy.column_stack((x, numpy.ones_like(x)))
    beta, _, _, _ = numpy.linalg.lstsq(A, L, rcond=None)
    return beta[0]


def hfd_hilbert(X, k_max=16):
    """
    Calculate Higuchi Fractal Dimension on the instantaneous frequency
    obtained from Hilbert transform of the input signal.

    Args:
        X (numpy.ndarray): Input 1D signal
        k_max (int): Maximum k value for HFD calculation

    Returns:
        float: Higuchi Fractal Dimension of the instantaneous frequency
    """
    assert X.ndim == 1, f"X must be a 1d numpy array, got shape {X.shape}"
    analytic_signal = hilbert(X)
    instantaneous_phase = numpy.angle(analytic_signal)
    instantaneous_freq = numpy.diff(instantaneous_phase)
    instantaneous_freq = numpy.append(instantaneous_freq, instantaneous_freq[-1])
    return hfd(instantaneous_freq, k_max=k_max)


if __name__ == "__main__":
    import time

    signal = numpy.random.rand(1000)  # shape: (1000,)

    # Test plain HFD speed
    start_time = time.time()
    complexity_plain = hfd(signal)
    plain_time = time.time() - start_time

    # Test Hilbert HFD speed
    start_time = time.time()
    complexity_hilbert = hfd_hilbert(signal)
    hilbert_time = time.time() - start_time

    print(f"Plain HFD: {complexity_plain:.6f}, Time: {plain_time:.6f}s")
    print(f"Hilbert HFD: {complexity_hilbert:.6f}, Time: {hilbert_time:.6f}s")
    print(f"Speed ratio (Hilbert/Plain): {hilbert_time/plain_time:.2f}x")
