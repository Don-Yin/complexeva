from math import factorial, log2

import numpy
from numpy.typing import ArrayLike

from ._validation import as_finite_1d, check_positive_int


def permutation_entropy(
    x: ArrayLike, order: int | numpy.integer = 3, delay: int | numpy.integer = 1, normalize: bool = False
) -> float:
    """Shannon entropy of ordinal patterns, in bits or normalized to [0, 1].

    Parameters
    ----------
    x : array-like
        Finite, real 1D time series with at least ``(order - 1) * delay + 1``
        samples.
    order : int, default 3
        Number of samples per ordinal pattern; must be at least 2.
    delay : int, default 1
        Positive sample spacing within each pattern. Pattern starting points
        advance by one sample, regardless of the delay.
    normalize : bool, default False
        Divide the entropy in bits by ``log2(order!)`` to obtain a value
        between 0 and 1.

    Returns
    -------
    float
        Entropy of the observed ordinal-pattern probabilities. A constant or
        strictly monotonic series has entropy 0.

    Raises
    ------
    ValueError
        If the input is not finite, real and 1D, is too short, or order/delay
        is not an integer in its allowed range.

    Notes
    -----
    Equal values retain their temporal order (stable sorting); no random
    jitter is added. Patterns are counted directly, without an integer hash
    that can overflow for large orders. This implements Bandt and Pompe
    (2002) permutation entropy, as in ``antropy.perm_entropy``.
    """
    values = as_finite_1d(x)
    order = check_positive_int(order, "order", minimum=2)
    delay = check_positive_int(delay, "delay")
    window = (order - 1) * delay + 1
    if len(values) < window:
        raise ValueError(f"x needs at least (order - 1) * delay + 1 = {window} samples, got {len(values)}")
    embedded = numpy.lib.stride_tricks.sliding_window_view(values, window)[:, ::delay]
    patterns = numpy.argsort(embedded, axis=1, kind="stable")
    _, counts = numpy.unique(patterns, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = float(-numpy.sum(probabilities * numpy.log2(probabilities)))
    if normalize:
        entropy /= log2(factorial(order))
    return entropy
