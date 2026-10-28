from numbers import Real

import numpy
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._validation import as_finite_1d, check_positive_int


@njit
def _match_counts(x: NDArray[numpy.float64], order: int, tolerance: float) -> tuple[int, int]:
    """Count unordered matching template pairs, excluding self-matches."""
    count_m, count_m1 = 0, 0
    n_templates = len(x) - order
    for i in range(n_templates - 1):
        for j in range(i + 1, n_templates):
            for k in range(order):
                if abs(x[i + k] - x[j + k]) >= tolerance:
                    break
            else:
                count_m += 1
                if abs(x[i + order] - x[j + order]) < tolerance:
                    count_m1 += 1
    return count_m, count_m1


def sample_entropy(
    x: ArrayLike,
    order: int | numpy.integer = 2,
    tolerance: float | numpy.floating | numpy.integer | None = None,
) -> float:
    """Sample entropy using the Chebyshev distance, in natural-log units (nats).

    Parameters
    ----------
    x : array-like
        Finite, real 1D time series with at least ``order + 2`` samples.
    order : int, default 2
        Length of the templates before extending them by one sample.
    tolerance : float or None, default None
        Nonnegative, finite distance threshold in the units of ``x``. The
        default is ``0.2 * std(x, ddof=0)``. Matches require distance strictly
        less than this threshold, not less than or equal to it.

    Returns
    -------
    float
        ``-log(A / B)``, where B counts matching pairs of length ``order`` and
        A counts those that still match at length ``order + 1``. Returns NaN
        if B is zero (including zero tolerance), and infinity if only A is zero.
        A constant series has NaN at the default zero tolerance, or 0 at a
        positive tolerance.

    Raises
    ------
    ValueError
        If the input is not finite, real and 1D, is too short, or a parameter
        is outside its allowed range.

    Notes
    -----
    Self-matches are excluded. Both counts use the same ``len(x) - order``
    templates, omitting the last length-order template because it cannot be
    extended. Pair counting uses O(N**2 * order) time and O(1) auxiliary memory.
    This is the Richman and Moorman (2000) sample entropy definition, with the
    same default tolerance as ``antropy.sample_entropy``.
    """
    values = as_finite_1d(x)
    order = check_positive_int(order, "order")
    if len(values) < order + 2:
        raise ValueError(f"x needs at least order + 2 = {order + 2} samples, got {len(values)}")
    if tolerance is None:
        tolerance = float(0.2 * numpy.std(values, ddof=0))
    if isinstance(tolerance, (bool, numpy.bool_)) or not isinstance(tolerance, Real):
        # All invalid public inputs use ValueError, including invalid types.
        raise ValueError(f"tolerance must be a finite nonnegative real number, got {tolerance!r}")  # noqa: TRY004
    try:
        tolerance = float(tolerance)
    except OverflowError as error:
        raise ValueError(f"tolerance must be finite, got {tolerance!r}") from error
    if not numpy.isfinite(tolerance) or tolerance < 0:
        raise ValueError(f"tolerance must be a finite nonnegative real number, got {tolerance!r}")
    if tolerance == 0:
        return float("nan")
    count_m, count_m1 = _match_counts(values, order, tolerance)
    if count_m == 0:
        return float("nan")
    if count_m1 == 0:
        return float("inf")
    return float(-numpy.log(count_m1 / count_m))
