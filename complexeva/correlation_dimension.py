"""Grassberger–Procaccia correlation dimension without a pairwise distance matrix."""

from math import sqrt

import numpy
from numba import njit
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import KDTree

from ._validation import as_finite_1d, as_signals, check_positive_int, float64_rows

# nolds' geometric grid: 0.1 * sample SD, multiplied by 1.03, up to 0.5 * SD.
_RADIUS_FACTORS = 0.1 * 1.03 ** numpy.arange(55)


@njit(cache=True)
def _strict_pair_counts(embedded, radii):
    """Resolve radii near a pair distance with an explicit strict Euclidean test."""
    histogram = numpy.zeros(len(radii) + 1, dtype=numpy.int64)
    for i in range(len(embedded)):
        for j in range(i + 1, len(embedded)):
            squared = 0.0
            for k in range(embedded.shape[1]):
                delta = embedded[i, k] - embedded[j, k]
                squared += delta * delta
            distance = sqrt(squared)
            low, high = 0, len(radii)
            while low < high:
                middle = (low + high) // 2
                if radii[middle] <= distance:
                    low = middle + 1
                else:
                    high = middle
            histogram[low] += 2
    return numpy.cumsum(histogram[:-1])


def _correlation_counts(embedded, radii):
    """Count distinct pairs; resolve floating-point boundary bins by strict tests."""
    tree = KDTree(embedded, leafsize=16)
    # A preceding radius alone is insufficient: squaring sqrt(3)'s predecessor
    # can exclude a pair even when its rounded Euclidean norm is below r.
    # Bracket possible distance-rounding error, and trust a tree count only
    # when both ends agree. This also removes exact closed-ball boundary ties.
    guard = 8 * embedded.shape[1] * numpy.finfo(numpy.float64).eps
    bounds = numpy.empty((len(radii), 2))
    with numpy.errstate(over="ignore", under="ignore"):
        bounds[:, 0] = numpy.nextafter(radii * (1 - guard), -numpy.inf)
        bounds[:, 1] = numpy.nextafter(radii * (1 + guard), numpy.inf)
    thresholds = bounds.ravel()
    permutation = numpy.argsort(thresholds, kind="stable")
    shells = tree.count_neighbors(tree, thresholds[permutation], cumulative=False)
    shells[0] -= len(embedded)
    counts = numpy.empty_like(shells)
    counts[permutation] = numpy.cumsum(shells)
    out = counts[::2].copy()
    uncertain = out != counts[1::2]
    if numpy.any(uncertain):
        out[uncertain] = _strict_pair_counts(embedded, radii[uncertain])
    return out


def _correlation_dimension_row(x, order, delay, radii):
    """Use one identical row calculation for scalar calls and every batch layout."""
    lower, upper = float(numpy.min(x)), float(numpy.max(x))
    if lower == upper:
        return numpy.nan
    # Only extreme magnitudes need rescaling to protect squared distances
    # and SD; ordinary inputs retain their exact floating-point boundaries.
    magnitude = max(abs(lower), abs(upper))
    scale = magnitude if magnitude < 1e-100 or magnitude > 1e100 else 1.0
    scaled = x if scale == 1.0 else x / scale
    if radii is None:
        row_radii = numpy.std(scaled, ddof=1) * _RADIUS_FACTORS
    else:
        with numpy.errstate(over="ignore", under="ignore"):
            row_radii = radii if scale == 1.0 else radii / scale
    window = (order - 1) * delay + 1
    embedded = sliding_window_view(scaled, window)[:, ::delay]
    counts = _correlation_counts(embedded, row_radii)
    nonzero = counts > 0
    if numpy.count_nonzero(nonzero) < 2:
        return numpy.nan
    n = len(embedded)
    correlation = counts[nonzero] / (n * (n - 1))
    # Use the supplied radii for the fit: their common rescaling only shifts
    # the intercept, and fitting in original units avoids subnormal logarithms.
    fit_radii = row_radii[nonzero] if radii is None else radii[nonzero]
    return float(numpy.polyfit(numpy.log(fit_radii), numpy.log(correlation), 1)[0])


def correlation_dimension(
    x: ArrayLike,
    order: int | numpy.integer = 2,
    delay: int | numpy.integer = 1,
    radii: ArrayLike | None = None,
) -> float | NDArray[numpy.float64]:
    """Grassberger–Procaccia correlation dimension along the last axis.

    Args:
        x: Finite, real-valued samples with time on the last axis. Each signal
            requires at least (order - 1) * delay + 2 samples.
        order: Positive delay-embedding dimension. Defaults to 2.
        delay: Positive sample spacing within each embedded vector. Defaults to 1.
        radii: At least two finite, positive, strictly increasing Euclidean
            radii in the units of x. The same supplied radii apply to every row.
            None uses 55 radii, 0.1 * std(x, ddof=1) * 1.03**arange(55), for
            each signal separately, matching nolds' radius grid.

    Returns:
        Ordinary least-squares slope of log(C(r)) against log(r), retaining
        only radii with nonzero correlation sums. A float for one signal, or
        a float64 array of shape x.shape[:-1] for a batch. Every batch entry
        equals its 1D call bit for bit. Constant signals and fits with fewer
        than two nonzero correlation sums return NaN.

    Raises:
        ValueError: If x is scalar, non-real, non-finite, or too short; order
            or delay is not a positive integer; or radii is not a 1D sequence
            of at least two finite, positive, strictly increasing values.
            Invalid samples are reported with their full index.

    Note:
        Implements Grassberger and Procaccia (1983), Physica D 9, 189–208,
        sections 1 and 5 and Appendix A, doi:10.1016/0167-2789(83)90298-1.
        C(r) counts distinct embedded vectors with Euclidean distance < r;
        self-matches are excluded and no Theiler window is applied. Its
        finite-sample normalization is 1 / (M * (M - 1)); replacing it with
        the paper's 1 / M**2 changes only the intercept, not the dimension.
        A dual k-d-tree traversal stores O(M * order + len(radii)) values,
        never an M by M distance matrix. Batches run serially to retain just
        one row's workspace. Radii near a pair distance are resolved with a
        compiled strict-distance scan to avoid squared-distance roundoff.
        Worst-case counting time remains quadratic.

        With matched radii this agrees with neurokit2.fractal_correlation.
        nolds.corr_dim(fit="poly") uses <= and inadvertently retains the zero
        diagonal of its distance matrix. Its self-match bias is not reproduced.
        The default RANSAC regression of nolds is also not used. A finite-radius
        slope need not equal the limiting attractor dimension: choose radii,
        delay, and embedding dimension for the signal's scaling region.
    """
    values = as_signals(x)
    order = check_positive_int(order, "order")
    delay = check_positive_int(delay, "delay")
    minimum = (order - 1) * delay + 2
    n = values.shape[-1]
    if n < minimum:
        raise ValueError(f"x needs at least (order - 1) * delay + 2 = {minimum} samples, got {n}")
    if radii is not None:
        radii = as_finite_1d(radii, "radii")
        if len(radii) < 2 or radii[0] <= 0 or numpy.any(numpy.diff(radii) <= 0):
            raise ValueError(f"radii must contain at least two positive, strictly increasing values, got {radii!r}")
    if values.ndim == 1:
        return _correlation_dimension_row(values, order, delay, radii)
    out = numpy.fromiter((_correlation_dimension_row(row, order, delay, radii) for row in float64_rows(values)), numpy.float64)
    return out.reshape(values.shape[:-1])
