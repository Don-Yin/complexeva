from math import factorial, log2

import numpy
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._validation import as_signals, check_positive_int, unit_stride_rows

_MAX_HISTOGRAM_ORDER = 7  # Bound the histogram to 7! int64 bins (40,320 bytes).


def _make_permutation_counter(order):
    """Specialize one stable Lehmer-counting loop for a fixed small order."""
    size = factorial(order)

    @njit(cache=True)
    def count(values, delay):
        """Count patterns without materializing windows, ranks, or per-window codes."""
        counts = numpy.zeros(size, dtype=numpy.int64)
        for start in range(len(values) - (order - 1) * delay):
            code = 0
            for left in range(order - 1):
                digit = 0
                value = values[start + left * delay]
                for right in range(left + 1, order):
                    # Equal later samples follow the current sample in a stable sort.
                    digit += value > values[start + right * delay]
                code = code * (order - left) + digit
            counts[code] += 1
        return counts

    return count


@njit(cache=True)
def _entropy_from_counts(counts, divisor):
    """Use identical non-reassociated arithmetic for a signal and each batch row."""
    total = counts.sum()
    entropy = 0.0
    for count in counts:
        if count:
            probability = count / total
            entropy += probability * numpy.log2(probability)
    return -entropy / divisor


def _make_permutation_rows(counter):
    """Read direct float64 rows, or reuse one conversion buffer without parallel-runtime overhead."""

    @njit(cache=True)
    def rows(values, direct, delay, divisor):
        n = values.shape[-1]
        count = values.size // n
        out = numpy.empty(count)
        buffer = numpy.empty(n) if direct is None else None
        for i in range(count):
            if direct is None:
                assert buffer is not None
                for j in range(n):
                    buffer[j] = values.flat[i * n + j]
                row = buffer
            else:
                row = direct[i]
            out[i] = _entropy_from_counts(counter(row, delay), divisor)
        return out

    return rows


_PERMUTATION_COUNTERS = {order: _make_permutation_counter(order) for order in range(2, _MAX_HISTOGRAM_ORDER + 1)}
_PERMUTATION_ROWS = {order: _make_permutation_rows(counter) for order, counter in _PERMUTATION_COUNTERS.items()}


def _large_order_entropy(values, order, delay, divisor):
    """Count arbitrary-order patterns without a bounded integer encoding."""
    window = (order - 1) * delay + 1
    embedded = numpy.lib.stride_tricks.sliding_window_view(values, window)[:, ::delay]
    patterns = numpy.argsort(embedded, axis=1, kind="stable")
    _, counts = numpy.unique(patterns, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-numpy.sum(probabilities * numpy.log2(probabilities))) / divisor


def permutation_entropy(
    x: ArrayLike, order: int | numpy.integer = 3, delay: int | numpy.integer = 1, normalize: bool = False
) -> float | NDArray[numpy.float64]:
    """Ordinal-pattern entropy of one signal, or of every signal along the last axis.

    Args:
        x: Finite, real-valued samples, converted to float64, with time on the last axis.
            Each signal requires at least (order - 1) * delay + 1 samples. Orders 2–7 use
            a compiled row loop with one reusable conversion buffer when needed.
        order: Number of samples per ordinal pattern, at least 2. Defaults to 3.
        delay: Positive sample spacing within each pattern. Pattern starting points advance by
            one sample, regardless of the delay. Defaults to 1.
        normalize: Divide the entropy in bits by log2(order!) to obtain a value between 0 and 1.
            Defaults to False.

    Returns:
        Entropy of the observed ordinal-pattern probabilities. A constant or strictly monotonic
        series has entropy 0. A float for one signal; for a batch, a float64 array of shape
        x.shape[:-1] whose entries equal permutation_entropy on each signal bit for bit.

    Raises:
        ValueError: If the input is scalar, not finite and real, is too short on the time axis,
            or order/delay is not an integer in its allowed range. Invalid samples include
            their full index in the error.

    Note:
        Equal values retain their temporal order (stable sorting); no random
        jitter is added. Orders up to 7 use collision-free factorial codes and a
        bounded histogram; larger orders count patterns directly, without an
        integer hash that can overflow. This implements Bandt and Pompe (2002)
        permutation entropy, as in antropy.perm_entropy.
    """
    values = as_signals(x)
    order = check_positive_int(order, "order", minimum=2)
    delay = check_positive_int(delay, "delay")
    window = (order - 1) * delay + 1
    n = values.shape[-1]
    if n < window:
        raise ValueError(f"x needs at least (order - 1) * delay + 1 = {window} samples, got {n}")
    divisor = log2(factorial(order)) if normalize else 1.0
    if values.ndim == 1:
        if order <= _MAX_HISTOGRAM_ORDER:
            return float(_entropy_from_counts(_PERMUTATION_COUNTERS[order](values, delay), divisor))
        return _large_order_entropy(values, order, delay, divisor)
    if order <= _MAX_HISTOGRAM_ORDER:
        out = _PERMUTATION_ROWS[order](values, unit_stride_rows(values), delay, divisor)
    else:
        # NumPy's arbitrary-order counting remains serial, with only a single row converted at a time.
        out = numpy.empty(values.size // n)
        for i, index in enumerate(numpy.ndindex(values.shape[:-1])):
            row = numpy.ascontiguousarray(values[index], dtype=numpy.float64)
            out[i] = _large_order_entropy(row, order, delay, divisor)
    return out.reshape(values.shape[:-1])
