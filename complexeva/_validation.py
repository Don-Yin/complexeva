import numpy
from numpy.typing import ArrayLike, NDArray


def as_finite_1d(x: ArrayLike, name: str = "x") -> NDArray[numpy.float64]:
    """Return a real, finite 1D float64 array; leave length checks to the caller."""
    try:
        values = numpy.asarray(x)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a 1D real numeric array, got {x!r}") from error
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {values.shape}")
    if values.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numbers, got dtype {values.dtype} with values {values!r}")
    values = numpy.asarray(values, dtype=numpy.float64)
    invalid = numpy.flatnonzero(~numpy.isfinite(values))
    if invalid.size:
        index = int(invalid[0])
        raise ValueError(f"{name} must contain only finite values, got {values[index]!r} at index {index}")
    return values


def check_positive_int(value: int | numpy.integer, name: str, minimum: int = 1) -> int:
    """Return an integer parameter or raise ValueError with its name and value."""
    if isinstance(value, (bool, numpy.bool_)) or not isinstance(value, (int, numpy.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)
