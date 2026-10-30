import numpy
from numba import get_num_threads, njit, prange
from numpy.typing import ArrayLike, NDArray

from ._sampen_tree import _tree_match_count, _tree_workspace_bytes
from ._validation import (
    _first_nonfinite,
    as_array,
    as_finite_1d,
    as_kernel_readable,
    check_finite_real,
    check_finite_rows,
    check_positive_int,
)

_TREE_MIN_TEMPLATES = 4096  # An allocation-free scan beats tree setup on short series.
_TREE_MAX_ORDER = 4  # Bound tree storage and avoid ineffective high-dimensional pruning.
_TREE_BATCH_BYTES = 12 * 1024 * 1024  # Bound concurrent row trees, not the process's thread pool.


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


@njit(cache=True)
def _match_counts(x: NDArray[numpy.float64], order: int, tolerance: float) -> tuple[int, int]:
    """Use bounded-memory spatial counting only for long, low-order inputs."""
    n_templates = len(x) - order
    if n_templates < _TREE_MIN_TEMPLATES or order > _TREE_MAX_ORDER:
        return _diagonal_match_counts(x, order, tolerance)
    return (
        _tree_match_count(x, order, n_templates, tolerance),
        _tree_match_count(x, order + 1, n_templates, tolerance),
    )


@njit(cache=True, parallel=True)
def _match_counts_rows(
    values: NDArray,
    direct: NDArray[numpy.float64],
    order: int,
    tolerances: NDArray[numpy.float64],
    chunks: int,
    count_dtype,
    copy_rows: bool,
):
    """Use direct float64 rows when possible; otherwise reuse one buffer per worker (exp_024)."""
    n = values.shape[-1]
    rows = len(tolerances)
    counts = numpy.zeros((rows, 2), dtype=count_dtype)
    # Size scratch before prange: Numba can hoist a conditional allocation inside the loop.
    buffers = numpy.empty((chunks if copy_rows else 0, n), dtype=numpy.float64)
    for chunk in prange(chunks):
        for i in range(rows * chunk // chunks, rows * (chunk + 1) // chunks):
            if copy_rows:
                row = buffers[chunk]
                for j in range(n):
                    row[j] = values.flat[i * n + j]
            else:
                row = direct[i]
            bad = _first_nonfinite(row)
            if bad >= 0:
                counts[i, 0] = -bad - 1  # Counts are nonnegative, so invalid positions need no extra array.
            elif tolerances[i] != 0 and numpy.isfinite(tolerances[i]):
                counts[i, 0], counts[i, 1] = _match_counts(row, order, tolerances[i])
    return counts


def _batch_tolerances(values: NDArray) -> NDArray[numpy.float64]:
    """Reduce contiguous float64 rows exactly as 1D; leave invalid rows for the full-index error."""
    tolerances = numpy.empty(int(numpy.prod(values.shape[:-1], dtype=numpy.intp)))
    for i in range(len(tolerances)):
        # ndindex materializes each range as a tuple; avoid its O(rows) Python-integer pool.
        index = i if values.ndim == 2 else numpy.unravel_index(i, values.shape[:-1])
        row = numpy.ascontiguousarray(values[index], dtype=numpy.float64)
        tolerances[i] = 0.2 * numpy.std(row, ddof=0) if _first_nonfinite(row) < 0 else numpy.nan
    return tolerances


def _entropy_from_counts(count_m: int, count_m1: int) -> float:
    """Keep the same scalar NumPy logarithm for a single signal and every batch row."""
    if count_m == 0:
        return float("nan")
    if count_m1 == 0:
        return float("inf")
    return float(-numpy.log(count_m1 / count_m))


def sample_entropy(
    x: ArrayLike,
    order: int | numpy.integer = 2,
    tolerance: float | numpy.floating | numpy.integer | None = None,
) -> float | NDArray[numpy.float64]:
    """Sample entropy using the Chebyshev distance, in natural-log units (nats).

    Args:
        x: Finite, real samples with time on the last axis, with at least order + 2 samples per
            signal. A batch is computed in parallel over signals; set NUMBA_NUM_THREADS to limit
            threads.
        order: Length of the templates before extending them by one sample. Defaults to 2.
        tolerance: Nonnegative, finite distance threshold in the units of x. None (default) uses
            0.2 * std(row, ddof=0) independently for each signal. Matches require distance strictly
            less than this threshold, not less than or equal to it.

    Returns:
        -log(A / B), where B counts matching pairs of length order and A counts those that still
        match at length order + 1. Returns NaN if B is zero (including zero tolerance), and
        infinity if only A is zero. A constant series has NaN at the default zero tolerance, or 0
        at a positive tolerance. Returns a float for 1D input, or a float64 array of shape
        x.shape[:-1] whose entries equal the 1D calls bit for bit.

    Raises:
        ValueError: If the input has no time axis, is not finite and real, is too short, or a
            parameter is outside its allowed range. Non-finite errors include the full index.

    Note:
        Self-matches are excluded. Both counts use the same len(x) - order
        templates, omitting the last length-order template because it cannot be
        extended. Long inputs with order <= 4 use exact dual-tree pair counting;
        other inputs use an allocation-free diagonal scan. Worst-case counting is
        quadratic, with O(N) auxiliary memory for the bounded-dimensional tree
        and O(1) auxiliary memory for diagonal counting, per active worker. Batch bookkeeping
        uses O(number of signals) memory. C-contiguous float64 batches and 2D float64 views
        use rows directly; other native layouts and dtypes need one float64 row buffer per
        active worker. Default tolerances still use contiguous float64 rows to match 1D std.
        Concurrent trees share a 12 MiB array budget (at least one tree); this limits active
        row workers without changing the Numba thread pool or the exact counting method.
        This is the Richman and Moorman (2000) sample entropy definition, with the
        same default tolerance as antropy.sample_entropy.
    """
    values = as_array(x)
    if values.ndim == 0:
        raise ValueError(f"x must have a time axis, got shape {values.shape}")
    values = as_finite_1d(values) if values.ndim == 1 else as_kernel_readable(values)
    order = check_positive_int(order, "order")
    n = values.shape[-1]
    if n < order + 2:
        raise ValueError(f"x needs at least order + 2 = {order + 2} samples, got {n}")
    if values.ndim == 1:
        if tolerance is None:
            tolerance = float(0.2 * numpy.std(values, ddof=0))
        tolerance = check_finite_real(tolerance, "tolerance", allow_zero=True)
        if tolerance == 0:
            return float("nan")
        return _entropy_from_counts(*_match_counts(values, order, tolerance))
    rows = values.size // n
    if tolerance is None:
        tolerances = _batch_tolerances(values)
    else:
        tolerances = numpy.full(rows, check_finite_real(tolerance, "tolerance", allow_zero=True), dtype=numpy.float64)
    copy_rows = values.dtype != numpy.float64 or (values.ndim != 2 and not values.flags.c_contiguous)
    direct = numpy.empty((0, 0)) if copy_rows else values.reshape(rows, n)
    chunks = min(rows, get_num_threads())
    if n - order >= _TREE_MIN_TEMPLATES and order <= _TREE_MAX_ORDER:
        chunks = min(chunks, max(1, _TREE_BATCH_BYTES // _tree_workspace_bytes(n - order, order + 1)))
    templates = n - order
    count_dtype = numpy.int32 if n <= 2147483647 and templates * (templates - 1) // 2 <= 2147483647 else numpy.int64
    counts = _match_counts_rows(values, direct, order, tolerances, chunks, count_dtype, copy_rows)
    if rows and numpy.min(counts[:, 0]) < 0:
        check_finite_rows(values, numpy.where(counts[:, 0] < 0, -counts[:, 0] - 1, -1))
    invalid_tolerances = tolerances[~numpy.isfinite(tolerances)]
    if invalid_tolerances.size:
        check_finite_real(float(invalid_tolerances[0]), "tolerance", allow_zero=True)
    del tolerances, invalid_tolerances
    result = numpy.empty(rows)
    for i, (b, a) in enumerate(counts):
        result[i] = _entropy_from_counts(int(b), int(a))
    return result.reshape(values.shape[:-1])
