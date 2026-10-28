import numpy
from numba import njit
from scipy.signal import hilbert


@njit
def lempel_ziv_core(binary_array):
    """Kaspar-Schuster count of components needed to reproduce a 0/1 integer array."""
    history_pos = 0  # position in history we're comparing against
    current_len = 1  # length of current pattern we're checking
    max_pattern_len = 1  # length of longest matching pattern found
    seq_pos = 1  # current position in sequence
    complexity = 1  # start with 1 as first character is always a new pattern
    seq_len = len(binary_array)

    while seq_pos + current_len <= seq_len:
        if binary_array[history_pos + current_len - 1] == binary_array[seq_pos + current_len - 1]:
            current_len += 1
        else:
            max_pattern_len = max(max_pattern_len, current_len)
            history_pos += 1
            if history_pos == seq_pos:
                complexity += 1
                seq_pos += max_pattern_len
                history_pos = 0
                current_len = 1
                max_pattern_len = 1
            else:
                current_len = 1
    if current_len > 1:  # sequence ended inside a copied component; Kaspar-Schuster counts it
        complexity += 1
    return complexity


def _binarize(time_series, method):
    """0/1 array: above median or mean, or increasing step for 'diff' (median follows doi:10.1101/2021.09.23.461002)."""
    vector = numpy.asarray(time_series, dtype=numpy.float64)
    if method == "diff":
        bits = numpy.diff(vector) > 0
    elif method == "mean":
        bits = vector > numpy.mean(vector)
    elif method == "median":
        bits = vector > numpy.median(vector)
    else:
        raise ValueError("Method must be 'mean', 'median', or 'diff'.")
    return bits.astype(numpy.int32)


def lempel_ziv(time_series, method="median", normalize=False):
    """
    Lempel-Ziv complexity as described in Kaspar and Schuster (1987), Phys. Rev. A.
    Counts the number of distinct patterns that need to be copied to reproduce the sequence.

    Args:
        time_series: Either a binary string (e.g., "1001110") or numeric array to be binarised
        method: Binarization method if time_series is numeric ('median', 'mean', 'diff')
        normalize: If True, return c * log2(n) / n for the binary sequence of length n (about 1 for random input)

    Returns: int (raw count) or float (normalized)
    """
    if isinstance(time_series, str) and all(c in "01" for c in time_series):
        binary_array = numpy.array([int(c) for c in time_series], dtype=numpy.int32)
    else:
        binary_array = _binarize(time_series, method)
    complexity = int(lempel_ziv_core(binary_array))
    if not normalize:
        return complexity
    n = len(binary_array)
    return float(complexity * numpy.log2(n) / n) if n > 1 else 0.0


def lempel_ziv_hilbert(time_series, method="median", normalize=False):
    """Lempel-Ziv complexity of the Hilbert envelope (instantaneous amplitude) of a 1D signal."""
    time_series = numpy.asarray(time_series)
    assert time_series.ndim == 1, f"time_series must be a 1d numpy array, got shape {time_series.shape}"
    envelope = numpy.abs(numpy.asarray(hilbert(time_series)))
    return lempel_ziv(envelope, method=method, normalize=normalize)
