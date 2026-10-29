import numpy
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ._sampen_tree import _tree_match_count
from ._validation import as_finite_1d, check_finite_real, check_positive_int

_TREE_MIN_TEMPLATES = 4096  # An allocation-free scan beats tree setup on short series.
_TREE_MAX_ORDER = 4  # Bound tree storage and avoid ineffective high-dimensional pruning.


@njit(cache=True)
def _diagonal_match_counts(x: NDArray[numpy.float64], order: int, tolerance: float) -> tuple[int, int]:
    """Count matching windows along offset diagonals without working arrays."""
    count_m, count_m1 = 0, 0
    for offset in range(1, len(x) - order):
        run = 0
        for i in range(len(x) - offset):
            run = run + 1 if abs(x[i] - x[i + offset]) < tolerance else 0
            count_m += run >= order
            count_m1 += run > order
        count_m -= run >= order  # The final window has no later template to pair with.
    return count_m, count_m1


def _match_counts(x: NDArray[numpy.float64], order: int, tolerance: float) -> tuple[int, int]:
    """Use bounded-memory spatial counting only for long, low-order inputs."""
    n_templates = len(x) - order
    if n_templates < _TREE_MIN_TEMPLATES or order > _TREE_MAX_ORDER:
        return _diagonal_match_counts(x, order, tolerance)
    return (
        _tree_match_count(x, order, n_templates, tolerance),
        _tree_match_count(x, order + 1, n_templates, tolerance),
    )


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
    extended. Long inputs with order <= 4 use exact dual-tree pair counting;
    other inputs use an allocation-free diagonal scan. Worst-case counting is
    quadratic, with O(N) auxiliary memory for the bounded-dimensional tree
    and O(1) auxiliary memory for diagonal counting.
    This is the Richman and Moorman (2000) sample entropy definition, with the
    same default tolerance as ``antropy.sample_entropy``.
    """
    values = as_finite_1d(x)
    order = check_positive_int(order, "order")
    if len(values) < order + 2:
        raise ValueError(f"x needs at least order + 2 = {order + 2} samples, got {len(values)}")
    if tolerance is None:
        tolerance = float(0.2 * numpy.std(values, ddof=0))
    tolerance = check_finite_real(tolerance, "tolerance", allow_zero=True)
    if tolerance == 0:
        return float("nan")
    count_m, count_m1 = _match_counts(values, order, tolerance)
    if count_m == 0:
        return float("nan")
    if count_m1 == 0:
        return float("inf")
    return float(-numpy.log(count_m1 / count_m))
