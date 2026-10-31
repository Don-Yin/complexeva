"""Pincus approximate entropy with exact, self-inclusive Chebyshev counts."""

import numpy
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._sampen_tree import _build
from ._validation import as_signals, check_finite_real, check_positive_int, unit_stride_rows
from .sample_entropy import _batch_tolerances

_TREE_MIN_TEMPLATES = 4096  # Short diagonal scans avoid constructing two spatial trees.
_TREE_MAX_ORDER = 4  # Keep spatial pruning and workspace bounded in useful dimensions.


@njit(cache=True)
def _phi(counts, n):
    """Average log probabilities with compensated summation, independent of row scheduling."""
    total, correction = 0.0, 0.0
    for i in range(n):
        term = numpy.log(counts[i] / n) - correction
        value = total + term
        correction = (value - total) - term
        total = value
    return total / n


@njit(cache=True)
def _diagonal_entropy(x, order, r):
    """Update both sets of template counts in one allocation-free comparison scan."""
    n = len(x)
    nm, nm1 = n - order + 1, n - order
    counts = numpy.ones((2, nm), dtype=numpy.int64)
    for offset in range(1, nm):
        run = 0
        for i in range(n - offset):
            run = run + 1 if abs(x[i] - x[i + offset]) <= r else 0
            if run >= order:
                first = i - order + 1
                counts[0, first] += 1
                counts[0, first + offset] += 1
            if run > order:
                first = i - order
                counts[1, first] += 1
                counts[1, first + offset] += 1
    return _phi(counts[0], nm) - _phi(counts[1], nm1)


@njit(cache=True)
def _tree_phi(x, dimension, r):
    """Count individual inclusive neighbourhoods with lazy, exact dual-tree updates."""
    n = len(x) - dimension + 1
    points, low, high, bounds, children, starts, stops = _build(x, dimension, n, 8)
    counts = numpy.zeros(n, dtype=numpy.int64)
    increments = numpy.zeros(len(starts), dtype=numpy.int64)
    depth, nodes = 1, len(starts)
    while nodes > 1:
        depth += 1
        nodes //= 2
    stack = numpy.empty((4 * depth, 2), dtype=numpy.int64)
    stack[0, 0], stack[0, 1] = 0, 0
    pending = 1
    while pending:
        pending -= 1
        a, b = stack[pending, 0], stack[pending, 1]
        inside, outside = True, False
        for k in range(dimension):
            if low[a, k] - high[b, k] > r or low[b, k] - high[a, k] > r:
                outside = True
                break
            if high[a, k] - low[b, k] > r or high[b, k] - low[a, k] > r:
                inside = False
        if outside:
            continue
        if inside:
            increments[a] += stops[b] - starts[b]
            if a != b:
                increments[b] += stops[a] - starts[a]
            continue
        leaf_a, leaf_b = children[a] == -1, children[b] == -1
        if leaf_a and leaf_b:
            for i in range(starts[a], stops[a]):
                start = i + 1 if a == b else starts[b]
                counts[i] += a == b
                for j in range(start, stops[b]):
                    match = True
                    for k in range(dimension):
                        match &= abs(points[i, k] - points[j, k]) <= r
                    counts[i] += match
                    counts[j] += match
            continue
        if a == b:
            left, right = children[a], children[a] + 1
            stack[pending, 0], stack[pending, 1] = left, left
            stack[pending + 1, 0], stack[pending + 1, 1] = left, right
            stack[pending + 2, 0], stack[pending + 2, 1] = right, right
            pending += 3
        elif leaf_b or (not leaf_a and bounds[a] >= bounds[b]):
            stack[pending, 0], stack[pending, 1] = children[a], b
            stack[pending + 1, 0], stack[pending + 1, 1] = children[a] + 1, b
            pending += 2
        else:
            stack[pending, 0], stack[pending, 1] = a, children[b]
            stack[pending + 1, 0], stack[pending + 1, 1] = a, children[b] + 1
            pending += 2
    node, active = 0, 1
    while node < active:
        if children[node] != -1:
            increments[children[node]] += increments[node]
            increments[children[node] + 1] += increments[node]
            active = max(active, children[node] + 2)
        else:
            for i in range(starts[node], stops[node]):
                counts[i] += increments[node]
        node += 1
    return _phi(counts, n)


@njit(cache=True)
def _entropy(x, order, r):
    """Choose by embedding size only, using the same mathematical definition."""
    if len(x) - order < _TREE_MIN_TEMPLATES or len(x) - order >= 2147483647 or order > _TREE_MAX_ORDER:
        return _diagonal_entropy(x, order, r)
    return _tree_phi(x, order, r) - _tree_phi(x, order + 1, r)


@njit(cache=True)
def _entropy_rows(values, direct, order, radii):
    """Reuse one row workspace; serial execution keeps heap below reference row loops."""
    n = values.shape[-1]
    out = numpy.empty(len(radii))
    buffer = numpy.empty(n) if direct is None else None
    for i in range(len(radii)):
        if direct is None:
            assert buffer is not None
            for j in range(n):
                buffer[j] = values.flat[i * n + j]
            row = buffer
        else:
            row = direct[i]
        out[i] = _entropy(row, order, radii[i])
    return out


def approximate_entropy(
    x: ArrayLike,
    order: int | numpy.integer = 2,
    r: float | numpy.floating | numpy.integer | None = None,
) -> float | NDArray[numpy.float64]:
    """Pincus approximate entropy with the Chebyshev distance, in nats.

    Args:
        x: Finite, real samples with time on the last axis. Each signal needs at least
            order + 1 samples. Native batch layouts need at most one float64 row buffer.
        order: Positive template length before extending by one sample. Defaults to 2.
        r: Nonnegative finite distance threshold in the units of x. None (default) uses
            0.2 * std(row, ddof=0) independently for each signal. A match includes distances
            equal to r and includes the template itself.

    Returns:
        Phi(order) - Phi(order + 1), where Phi(m) is the mean of log(C_i(m)) over all
        len(x) - m + 1 templates and C_i(m) is the fraction matching template i. A float for
        1D input, or a float64 array of shape x.shape[:-1] whose entries equal individual
        1D calls bit for bit. Constant signals return 0, including at the default r = 0:
        all templates match, so approximate entropy is defined. Finite-sample values can
        be negative; they are not clipped.

    Raises:
        ValueError: If x has no time axis, is not finite and real, is too short, order is
            not a positive integer, or r is not a nonnegative finite number. Invalid
            samples include their full index in the error.

    Note:
        Implements Pincus (1991), PNAS 88:2297-2301, equations 1, 2, 6 and 10,
        doi:10.1073/pnas.88.6.2297. Self-matches and the final length-order template are
        retained, unlike sample entropy. The default order and population-SD tolerance
        follow the usual Richman and Moorman (2000) / antropy convention. Short or
        high-order inputs use a diagonal scan; long inputs with order <= 4 use exact
        dual-tree neighbourhood counting. Both have quadratic worst-case time and
        O(N) auxiliary memory. Batch execution is serial and reuses one signal's
        workspace, plus O(number of signals) output and tolerance arrays.
    """
    values = as_signals(x)
    order = check_positive_int(order, "order")
    n = values.shape[-1]
    if n < order + 1:
        raise ValueError(f"x needs at least order + 1 = {order + 1} samples, got {n}")
    if r is not None:
        r = check_finite_real(r, "r", allow_zero=True)
    if values.ndim == 1:
        radius = check_finite_real(float(0.2 * numpy.std(values)) if r is None else r, "r", allow_zero=True)
        return float(_entropy(values, order, radius))
    radii = _batch_tolerances(values) if r is None else numpy.full(values.size // n, r)
    for radius in radii:
        if not numpy.isfinite(radius):
            check_finite_real(radius, "r", allow_zero=True)
    return _entropy_rows(values, unit_stride_rows(values), order, radii).reshape(values.shape[:-1])
