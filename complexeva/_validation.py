import math
from numbers import Real

import numpy
from numba import njit
from numpy.typing import ArrayLike, NDArray


@njit(cache=True)
def _first_nonfinite(rows: NDArray) -> int:
    """Locate the first invalid sample of 2D rows in logical order, in any strides, without a boolean mask."""
    n = rows.shape[1]
    for i in range(rows.shape[0]):
        row = rows[i]
        for j in range(n):
            if not numpy.isfinite(row[j]):
                return i * n + j
    return -1


def as_array(x: ArrayLike, name: str = "x") -> NDArray:
    """Return x as an array, or raise ValueError when it is ragged or not array-like."""
    try:
        return numpy.asarray(x)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real numeric array, got {x!r}") from error


def as_real_float64(values: NDArray, name: str = "x") -> NDArray[numpy.float64]:
    """Return a C-contiguous float64 copy or view of a real numeric array, or raise ValueError."""
    if values.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numbers, got dtype {values.dtype} with values {values!r}")
    return numpy.ascontiguousarray(values, dtype=numpy.float64)


def float64_rows(values: NDArray):
    """Yield each signal along the last axis as a contiguous float64 row, one at a time, in C order."""
    lead = values.shape[:-1]
    for i in range(math.prod(lead)):
        # ndindex keeps a tuple per row; unravel_index keeps memory constant in the row count.
        yield numpy.ascontiguousarray(values[numpy.unravel_index(i, lead)], dtype=numpy.float64)


def as_kernel_readable(values: NDArray, name: str = "x") -> NDArray:
    """Return values that Numba kernels can read in place, without copying the batch.

    Native bool, int, uint, float32 and float64 keep their dtype and strides. Numba cannot read
    float16, longdouble or non-native byte order, so those are converted to float64.
    """
    if values.dtype.isnative and (values.dtype.kind in "biu" or values.dtype.kind == "f" and values.dtype.itemsize in (4, 8)):
        return values
    return as_real_float64(values, name)


def as_signals(x: ArrayLike, name: str = "x") -> NDArray:
    """Return finite, real samples with time on the last axis; leave length checks to the caller.

    One signal becomes a contiguous float64 array. A batch keeps its layout (see as_kernel_readable),
    so kernels read it in place. A non-finite sample raises with its full index.
    """
    values = as_array(x, name)
    if values.ndim == 0:
        raise ValueError(f"{name} must have a time axis, got shape {values.shape}")
    values = as_real_float64(values, name) if values.ndim == 1 else as_kernel_readable(values, name)
    if values.dtype.kind != "f" or values.size == 0:
        return values
    # Scan 2D row views; only a non-contiguous array with more than two axes loops over its leading axes.
    leading = values.shape[:-2] if values.ndim > 2 and not values.flags.c_contiguous else ()
    for lead in numpy.ndindex(leading):
        block = values[lead]
        index = _first_nonfinite(block.reshape(-1, values.shape[-1]))
        if index >= 0:
            where = lead + tuple(int(j) for j in numpy.unravel_index(index, block.shape))
            where = where[0] if values.ndim == 1 else where
            raise ValueError(f"{name} must contain only finite values, got {values[where]!r} at index {where}")
    return values


def as_finite_1d(x: ArrayLike, name: str = "x") -> NDArray[numpy.float64]:
    """Return a real, finite 1D float64 array; leave length checks to the caller."""
    values = as_array(x, name)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {values.shape}")
    return as_signals(values, name)


def unit_stride_rows(values: NDArray) -> NDArray[numpy.float64] | None:
    """Return float64 rows a kernel can read in place with the same arithmetic as 1D, else None.

    Rows must be contiguous along time: a reassociated sum vectorizes differently over strided rows.
    """
    n = values.shape[-1]
    contiguous = values.flags.c_contiguous or values.ndim == 2 and values.strides[-1] == values.itemsize
    return values.reshape(values.size // n, n) if values.dtype == numpy.float64 and contiguous else None


def check_positive_int(value: int | numpy.integer, name: str, minimum: int = 1) -> int:
    """Return an integer parameter or raise ValueError with its name and value."""
    if isinstance(value, (bool, numpy.bool_)) or not isinstance(value, (int, numpy.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def check_finite_real(value: float | numpy.floating | numpy.integer, name: str, *, allow_zero: bool) -> float:
    """Return a finite real parameter that is positive (or nonnegative) as a float, or raise ValueError."""
    bound = ">= 0" if allow_zero else "> 0"
    if isinstance(value, (bool, numpy.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number {bound}, got {value!r}")  # noqa: TRY004
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite real number {bound}, got {value!r}") from error
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise ValueError(f"{name} must be a finite real number {bound}, got {value!r}")
    return number
