import numpy
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._hilbert import hilbert_envelope
from ._validation import as_finite_1d


@njit(cache=True)
def _count_components(binary_array: NDArray[numpy.bool_]) -> int:
    """Count Kaspar-Schuster components with a linear-time suffix automaton.

    Args:
        binary_array: One-dimensional array containing only 0/False and 1/True.

    Returns:
        Integer number of components, including a final unfinished copied component. The initial
        component contributes 1 even for an empty array, preserving the raw-count convention.
    """
    n = len(binary_array)
    if n < 2:
        return 1
    if n >= 2**30:
        raise ValueError("binary sequence too long for int32 automaton states")
    # int32 state ids: 2n states, 16n bytes of transitions plus 8n of links and lengths.
    transitions = numpy.full((2 * n, 2), -1, dtype=numpy.int32)
    link = numpy.full(2 * n, -1, dtype=numpy.int32)
    length = numpy.zeros(2 * n, dtype=numpy.int32)
    last, size, count, start = 0, 1, 1, 1
    for position in range(n):
        symbol = int(binary_array[position])
        current = size
        size += 1
        length[current] = length[last] + 1
        previous = last
        while previous >= 0 and transitions[previous, symbol] < 0:
            transitions[previous, symbol] = current
            previous = link[previous]
        if previous < 0:
            link[current] = 0
        else:
            target = transitions[previous, symbol]
            if length[previous] + 1 == length[target]:
                link[current] = target
            else:
                clone = size
                size += 1
                length[clone] = length[previous] + 1
                link[clone] = link[target]
                transitions[clone, 0] = transitions[target, 0]
                transitions[clone, 1] = transitions[target, 1]
                while previous >= 0 and transitions[previous, symbol] == target:
                    transitions[previous, symbol] = clone
                    previous = link[previous]
                link[target] = clone
                link[current] = clone
        last = current
        # The longest suffix seen earlier has length[link[last]]; a component ends once its
        # own text exceeds it, which keeps self-overlapping copies.
        if position + 1 - start > length[link[last]]:
            count += 1
            start = position + 1
    return count + (start < n)


def _binarize(time_series: ArrayLike, method: str) -> NDArray[numpy.bool_]:
    """Convert numeric samples to binary symbols for Lempel-Ziv counting.

    Args:
        time_series: One-dimensional, finite, real-valued samples, converted to float64.
        method: "median" or "mean" marks samples strictly above the threshold; "diff" marks
            strictly increasing adjacent samples. Median thresholding follows doi:10.1101/2021.09.23.461002.

    Returns:
        Boolean array of symbols. Its length is len(time_series), or max(len(time_series) - 1, 0)
        for "diff". An empty signal produces an empty array.

    Raises:
        ValueError: If the input is not a finite, real, one-dimensional signal or method is unknown.
    """
    vector = as_finite_1d(time_series, "time_series")
    if vector.size == 0 and method in ("diff", "mean", "median"):
        return numpy.empty(0, dtype=numpy.bool_)
    if method == "diff":
        bits = numpy.diff(vector) > 0
    elif method == "mean":
        bits = vector > numpy.mean(vector)
    elif method == "median":
        bits = vector > numpy.median(vector)
    else:
        raise ValueError(f"method must be 'mean', 'median', or 'diff', got {method!r}")
    return bits


def lempel_ziv(time_series: ArrayLike | str, method: str = "median", normalize: bool = False) -> int | float:
    """Compute Lempel-Ziv complexity using Kaspar and Schuster (1987).

    Args:
        time_series: A binary string containing only "0" and "1", used without binarization, or
            one-dimensional, finite, real-valued samples, converted to float64 before binarization.
        method: Numeric binarization rule: "median" (default) or "mean" marks samples strictly above
            that threshold, and "diff" marks strictly increasing adjacent samples. Ignored for
            binary-string input, which is already encoded.
        normalize: False returns the integer count c. True returns c * log2(n) / n, where n is the
            binary sequence length (one fewer numeric sample for "diff"). Defaults to False.

    Returns:
        Raw integer component count, or a normalized float that is approximately 1 for a long
        random binary sequence and need not be bounded by 1. For n <= 1, normalized complexity
        is 0.0; the raw count of an empty or single-symbol sequence is 1.

    Raises:
        ValueError: If a string contains nonbinary characters, numeric input is not a finite,
            real, one-dimensional signal, or the numeric binarization method is unknown.
    """
    if isinstance(time_series, str):
        if any(c not in "01" for c in time_series):
            raise ValueError(f"time_series must be a binary string containing only '0' and '1', got {time_series!r}")
        binary_array = numpy.array([c == "1" for c in time_series], dtype=numpy.bool_)
    else:
        binary_array = _binarize(time_series, method)
    complexity = int(_count_components(binary_array))
    if not normalize:
        return complexity
    n = len(binary_array)
    return float(complexity * numpy.log2(n) / n) if n > 1 else 0.0


def lempel_ziv_hilbert(time_series: ArrayLike, method: str = "median", normalize: bool = False) -> int | float:
    """Compute Lempel-Ziv complexity on the Hilbert envelope of a signal.

    Args:
        time_series: Nonempty, one-dimensional, finite, real-valued samples, converted to float64
            before taking the Hilbert transform. Binary strings are not accepted by this variant.
        method: Envelope binarization rule: "median" (default), "mean", or "diff" for rising steps.
            Threshold comparisons are strict, as in lempel_ziv.
        normalize: False returns the raw integer count. True returns c * log2(n) / n using the
            binarized envelope length n, or 0.0 for n <= 1. Defaults to False.

    Returns:
        Integer count or normalized float for abs(hilbert(time_series)), the instantaneous
        amplitude. Normalized values near 1 indicate random binary structure, not a strict bound.

    Raises:
        ValueError: If time_series is empty or not a finite, real, one-dimensional signal, or the
            binarization method is unknown.
    """
    return lempel_ziv(hilbert_envelope(time_series, "time_series"), method=method, normalize=normalize)
