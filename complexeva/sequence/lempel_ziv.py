#!/Users/donyin/miniconda3/envs/rotation-1/bin/python

import numpy
from scipy.signal import hilbert
import time
from numba import njit


@njit
def lempel_ziv_core(binary_array):
    """
    Core Lempel-Ziv complexity algorithm optimized with numba.
    Works on binary arrays of 0s and 1s (integers).

    Args:
        binary_array: numpy array of 0s and 1s (integers)

    Returns:
        int: The number of distinct patterns (complexity measure)
    """
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
            if current_len > max_pattern_len:
                max_pattern_len = current_len
            history_pos += 1
            if history_pos == seq_pos:
                complexity += 1
                seq_pos += max_pattern_len
                history_pos = 0
                current_len = 1
                max_pattern_len = 1
            else:
                current_len = 1
    return complexity


@njit
def time_series_to_binary_core(vector, threshold, method_idx):
    """
    Core binarization function optimized with numba.

    Args:
        vector: numpy array of numeric values
        threshold: threshold value for binarization
        method_idx: 0=mean, 1=median, 2=diff

    Returns:
        numpy array of 0s and 1s (integers)
    """
    if method_idx == 2:  # diff method
        diffs = numpy.diff(vector)
        return (diffs > 0).astype(numpy.int32)
    else:  # mean or median method
        return (vector > threshold).astype(numpy.int32)


def lempel_ziv(time_series, method="median", **kwargs):
    """
    - compared to other implementations, this implementation is makes the most sense

    Lempel-Ziv complexity as described in Kaspar and Schuster, Phys. Rev. A.
    Counts the number of distinct patterns that need to be copied to reproduce the sequence.

    Args:
        time_series: Either a binary string (e.g., "1001110") or numeric array to be converted
        method: Binarization method if time_series is numeric ('median', 'mean', 'diff')

    Returns: int: The number of distinct patterns (complexity measure)
    """
    # Handle binary string input
    if isinstance(time_series, str) and all(c in "01" for c in time_series):
        binary_array = numpy.array([int(c) for c in time_series], dtype=numpy.int32)
        return lempel_ziv_core(binary_array)

    # Handle numeric input
    vector = numpy.array(time_series, dtype=numpy.float64)

    if method == "diff":
        binary_array = time_series_to_binary_core(vector, 0.0, 2)  # threshold not used for diff
    elif method == "mean":
        threshold = numpy.mean(vector)
        binary_array = time_series_to_binary_core(vector, threshold, 0)
    elif method == "median":
        threshold = numpy.median(vector)
        binary_array = time_series_to_binary_core(vector, threshold, 1)
    else:
        raise ValueError("Method must be 'mean', 'median', or 'diff'.")

    return lempel_ziv_core(binary_array)


def time_series_to_binary(vector, method="median"):
    """
    - convert a time series to a binary sequence based on a threshold.
    - returns: str: A binary sequence as a string of '0's and '1's.
    - https://doi.org/10.1101/2021.09.23.461002 this paper used median, making median the default method.
    - diff also makes somewhat sense
    """

    # If input is already a binary string, return it as-is
    if isinstance(vector, str) and all(c in "01" for c in vector):
        return vector

    vector = numpy.array(vector)

    match method:

        case "mean":
            threshold = numpy.mean(vector)
            binary_seq = "".join(["1" if x > threshold else "0" for x in vector])

        case "median":
            threshold = numpy.median(vector)
            binary_seq = "".join(["1" if x > threshold else "0" for x in vector])

        case "diff":  # this measures whether the value is increasing or decreasing as compared to the previous value
            diffs = numpy.diff(vector)
            binary_seq = "".join(["1" if x > 0 else "0" for x in diffs])

        case _:
            raise ValueError("Method must be 'mean', 'median', or 'diff'.")

    return binary_seq


def lempel_ziv_hilbert_envelope(time_series, method="median", **kwargs):
    """
    Calculate Lempel-Ziv complexity on the envelope (instantaneous amplitude) of the Hilbert transform.
    This measures complexity of amplitude modulations.

    Args:
        time_series (numpy.ndarray): Input 1D signal
        method (str): Binarization method ('median', 'mean', 'diff')
        **kwargs: Additional arguments

    Returns:
        int: Lempel-Ziv complexity of the Hilbert-transformed signal
    """
    time_series = numpy.array(time_series)
    assert time_series.ndim == 1, f"time_series must be a 1d numpy array, got shape {time_series.shape}"
    analytic_signal = hilbert(time_series)
    envelope = numpy.abs(analytic_signal)
    return lempel_ziv(envelope, method=method, **kwargs)


def lempel_ziv_hilbert_transform(time_series, method="median", **kwargs):
    """
    Calculate Lempel-Ziv complexity on the actual Hilbert transform (imaginary part of analytic signal).
    This applies complexity analysis to the 90-degree phase-shifted version of the signal.

    Args:
        time_series (numpy.ndarray): Input 1D signal
        method (str): Binarization method ('median', 'mean', 'diff')
        **kwargs: Additional arguments

    Returns:
        int: Lempel-Ziv complexity of the Hilbert transform
    """
    time_series = numpy.array(time_series)
    assert time_series.ndim == 1, f"time_series must be a 1d numpy array, got shape {time_series.shape}"
    analytic_signal = hilbert(time_series)
    hilbert_transform = numpy.imag(analytic_signal)
    return lempel_ziv(hilbert_transform, method=method, **kwargs)


def lempel_ziv_hilbert_frequency(time_series, method="median", **kwargs):
    """
    Calculate Lempel-Ziv complexity on the instantaneous frequency from Hilbert transform.
    This measures complexity of frequency modulations.

    Args:
        time_series (numpy.ndarray): Input 1D signal
        method (str): Binarization method ('median', 'mean', 'diff')
        **kwargs: Additional arguments

    Returns:
        int: Lempel-Ziv complexity of the instantaneous frequency
    """
    time_series = numpy.array(time_series)
    assert time_series.ndim == 1, f"time_series must be a 1d numpy array, got shape {time_series.shape}"
    analytic_signal = hilbert(time_series)
    instantaneous_phase = numpy.angle(analytic_signal)
    instantaneous_freq = numpy.diff(instantaneous_phase)
    instantaneous_freq = numpy.append(instantaneous_freq, instantaneous_freq[-1])
    return lempel_ziv(instantaneous_freq, method=method, **kwargs)


def lempel_ziv_hilbert(time_series, method="median", **kwargs):
    """
    Calculate Lempel-Ziv complexity on the Hilbert-transformed BOLD time-series.

    Based on research literature context, this applies LZ complexity to the envelope
    (instantaneous amplitude) of the Hilbert transform, which captures temporal
    brain activity patterns in the BOLD signal amplitude over time.

    Args:
        time_series (numpy.ndarray): Input 1D BOLD time-series
        method (str): Binarization method ('median', 'mean', 'diff')
        **kwargs: Additional arguments

    Returns:
        int: Lempel-Ziv complexity of the Hilbert-transformed signal
    """
    return lempel_ziv_hilbert_envelope(time_series, method=method, **kwargs)


if __name__ == "__main__":
    # Test with synthetic signal
    signal = numpy.random.rand(1000)

    print("Testing Lempel-Ziv complexity methods:")
    print("=" * 50)

    # Test plain LZ
    start_time = time.time()
    lz_plain = lempel_ziv(signal, method="median")
    plain_time = time.time() - start_time

    # Test Hilbert LZ variants
    start_time = time.time()
    lz_envelope = lempel_ziv_hilbert_envelope(signal, method="median")
    envelope_time = time.time() - start_time

    start_time = time.time()
    lz_transform = lempel_ziv_hilbert_transform(signal, method="median")
    transform_time = time.time() - start_time

    start_time = time.time()
    lz_frequency = lempel_ziv_hilbert_frequency(signal, method="median")
    frequency_time = time.time() - start_time

    print(f"Plain LZ: {lz_plain}, Time: {plain_time:.6f}s")
    print(f"Hilbert Envelope LZ: {lz_envelope}, Time: {envelope_time:.6f}s")
    print(f"Hilbert Transform LZ: {lz_transform}, Time: {transform_time:.6f}s")
    print(f"Hilbert Frequency LZ: {lz_frequency}, Time: {frequency_time:.6f}s")
    print(f"\nHilbert/Plain speed ratios:")
    print(f"  Envelope: {envelope_time/plain_time:.2f}x")
    print(f"  Transform: {transform_time/plain_time:.2f}x")
    print(f"  Frequency: {frequency_time/plain_time:.2f}x")

    # Test original functionality
    print(f"\nOriginal test:")
    sequence = "1001111011000010"
    print(f"LZ of binary sequence '{sequence}': {lempel_ziv(sequence)}")

    # Additional benchmark: test the same signal multiple times to see compiled performance
    print(f"\nPerformance after compilation (running same signal 5 times):")
    times = []
    for i in range(5):
        start_time = time.time()
        lz_result = lempel_ziv(signal, method="median")
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"  Run {i+1}: {lz_result}, Time: {elapsed:.6f}s")

    print(f"Average compiled time: {numpy.mean(times):.6f}s")
    print(f"Min compiled time: {numpy.min(times):.6f}s")
