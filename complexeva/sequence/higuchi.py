#!/Users/donyin/miniconda3/envs/rotation-1/bin/python


import numpy
from numba import njit
from scipy.signal import hilbert


@njit
def compute_L_x(X, k_max=64):
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


def hfd(X, k_max=64):
    """expect a 1d numpy array as input"""
    assert X.ndim == 1, f"X must be a 1d numpy array, got shape {X.shape}"
    x, L = compute_L_x(X, k_max)
    A = numpy.column_stack((x, numpy.ones_like(x)))
    beta, _, _, _ = numpy.linalg.lstsq(A, L, rcond=None)
    return beta[0]


def hfd_hilbert_envelope(X, k_max=64):
    """
    Calculate HFD on the envelope (instantaneous amplitude) of the Hilbert transform.
    This measures complexity of amplitude modulations.
    """
    assert X.ndim == 1, f"X must be a 1d numpy array, got shape {X.shape}"
    analytic_signal = hilbert(X)
    envelope = numpy.abs(analytic_signal)
    return hfd(envelope, k_max=k_max)


def hfd_hilbert_transform(X, k_max=64):
    """
    Calculate HFD on the actual Hilbert transform (imaginary part of analytic signal).
    This applies fractal analysis to the 90-degree phase-shifted version of the signal.
    """
    assert X.ndim == 1, f"X must be a 1d numpy array, got shape {X.shape}"
    analytic_signal = hilbert(X)
    hilbert_transform = numpy.imag(analytic_signal)
    return hfd(hilbert_transform, k_max=k_max)


def hfd_hilbert_frequency(X, k_max=64):
    """
    Calculate HFD on the instantaneous frequency from Hilbert transform.
    This measures complexity of frequency modulations (your original implementation).
    """
    assert X.ndim == 1, f"X must be a 1d numpy array, got shape {X.shape}"
    analytic_signal = hilbert(X)
    instantaneous_phase = numpy.angle(analytic_signal)
    instantaneous_freq = numpy.diff(instantaneous_phase)
    instantaneous_freq = numpy.append(instantaneous_freq, instantaneous_freq[-1])
    return hfd(instantaneous_freq, k_max=k_max)


# Keep the original as an alias for backward compatibility
def hfd_hilbert(X, k_max=64):
    """
    Calculate Higuchi Fractal Dimension on the Hilbert-transformed BOLD time-series.

    Based on research literature context, this applies HFD to the envelope
    (instantaneous amplitude) of the Hilbert transform, which captures temporal
    brain activity patterns in the BOLD signal amplitude over time.

    Args:
        X (numpy.ndarray): Input 1D BOLD time-series
        k_max (int): Maximum k value for HFD calculation

    Returns:
        float: Higuchi Fractal Dimension of the Hilbert-transformed signal
    """
    return hfd_hilbert_envelope(X, k_max=k_max)


if __name__ == "__main__":
    import time

    signal = numpy.random.rand(1000)  # shape: (1000,)

    # Test plain HFD speed
    start_time = time.time()
    complexity_plain = hfd(signal)
    plain_time = time.time() - start_time

    # Test Hilbert HFD variants
    start_time = time.time()
    complexity_envelope = hfd_hilbert_envelope(signal)
    envelope_time = time.time() - start_time

    start_time = time.time()
    complexity_transform = hfd_hilbert_transform(signal)
    transform_time = time.time() - start_time

    start_time = time.time()
    complexity_frequency = hfd_hilbert_frequency(signal)
    frequency_time = time.time() - start_time

    print(f"Plain HFD: {complexity_plain:.6f}, Time: {plain_time:.6f}s")
    print(f"Hilbert Envelope HFD: {complexity_envelope:.6f}, Time: {envelope_time:.6f}s")
    print(f"Hilbert Transform HFD: {complexity_transform:.6f}, Time: {transform_time:.6f}s")
    print(f"Hilbert Frequency HFD: {complexity_frequency:.6f}, Time: {frequency_time:.6f}s")
    print(f"\nHilbert/Plain speed ratios:")
    print(f"  Envelope: {envelope_time/plain_time:.2f}x")
    print(f"  Transform: {transform_time/plain_time:.2f}x")
    print(f"  Frequency: {frequency_time/plain_time:.2f}x")
