from math import factorial, log2

import numpy
from numba import njit
from numpy.typing import ArrayLike

from ._validation import as_finite_1d, check_positive_int

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


_PERMUTATION_COUNTERS = {order: _make_permutation_counter(order) for order in range(2, _MAX_HISTOGRAM_ORDER + 1)}


def permutation_entropy(
    x: ArrayLike, order: int | numpy.integer = 3, delay: int | numpy.integer = 1, normalize: bool = False
) -> float:
    """Shannon entropy of ordinal patterns, in bits or normalized to [0, 1].

    Args:
        x: Finite, real 1D time series with at least (order - 1) * delay + 1 samples.
        order: Number of samples per ordinal pattern, at least 2. Defaults to 3.
        delay: Positive sample spacing within each pattern. Pattern starting points advance by
            one sample, regardless of the delay. Defaults to 1.
        normalize: Divide the entropy in bits by log2(order!) to obtain a value between 0 and 1.
            Defaults to False.

    Returns:
        Entropy of the observed ordinal-pattern probabilities. A constant or strictly monotonic
        series has entropy 0.

    Raises:
        ValueError: If the input is not finite, real and 1D, is too short, or order/delay is not
            an integer in its allowed range.

    Note:
        Equal values retain their temporal order (stable sorting); no random
        jitter is added. Orders up to 7 use collision-free factorial codes and a
        bounded histogram; larger orders count patterns directly, without an
        integer hash that can overflow. This implements Bandt and Pompe (2002)
        permutation entropy, as in antropy.perm_entropy.
    """
    values = as_finite_1d(x)
    order = check_positive_int(order, "order", minimum=2)
    delay = check_positive_int(delay, "delay")
    window = (order - 1) * delay + 1
    if len(values) < window:
        raise ValueError(f"x needs at least (order - 1) * delay + 1 = {window} samples, got {len(values)}")
    if order <= _MAX_HISTOGRAM_ORDER:
        counts = _PERMUTATION_COUNTERS[order](values, delay)
        counts = counts[counts != 0]
    else:
        embedded = numpy.lib.stride_tricks.sliding_window_view(values, window)[:, ::delay]
        patterns = numpy.argsort(embedded, axis=1, kind="stable")
        _, counts = numpy.unique(patterns, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = float(-numpy.sum(probabilities * numpy.log2(probabilities)))
    if normalize:
        entropy /= log2(factorial(order))
    return entropy
