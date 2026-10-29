import math
from numbers import Real

import numpy
from numba import njit
from numpy.typing import ArrayLike, NDArray


@njit(cache=True)
def _first_nonfinite(values: NDArray[numpy.float64]) -> int:
    """Locate the first invalid sample without allocating a boolean mask."""
    for index in range(len(values)):
        if not numpy.isfinite(values[index]):
            return index
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


def as_finite_1d(x: ArrayLike, name: str = "x") -> NDArray[numpy.float64]:
    """Return a real, finite 1D float64 array; leave length checks to the caller."""
    values = as_array(x, name)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {values.shape}")
    values = as_real_float64(values, name)
    index = _first_nonfinite(values)
    if index >= 0:
        raise ValueError(f"{name} must contain only finite values, got {values[index]!r} at index {index}")
    return values


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
